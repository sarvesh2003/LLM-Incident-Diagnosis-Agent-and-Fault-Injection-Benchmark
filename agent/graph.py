"""LangGraph investigation loop: baseline checks, tool-calling turns, and a grounded final report."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Literal, TypedDict, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from agent.prompts import SYSTEM_PROMPT
from agent.tools import observe, usable_evidence

# Counts the two baseline observations and duplicate attempts.
TOOL_CALL_BUDGET = 15
# "confirmed" needs cited evidence from at least this many distinct tools.
CORROBORATION_THRESHOLD = 2


class EvidenceEntry(TypedDict):
    tool_name: str
    args: dict
    result: str
    observed_at: str


class PendingAction(TypedDict):
    type: Literal["tool_call", "conclude"]
    name: str | None
    args: dict | None


class InvestigationState(TypedDict):
    trigger: str
    evidence_log: list[EvidenceEntry]
    calls_used: int
    pending_action: PendingAction | None
    conclusion: dict | None
    trace: list[dict]


def trace_event(event: str, **data) -> dict:
    return {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **data}


class ConcludeOut(BaseModel):
    root_cause_id: str = Field(description="short kebab-case slug you invent for this root cause, e.g. 'redis1-maxmemory-misconfigured'. Use 'none' if inconclusive.")
    summary: str = Field(
        description="2-3 sentences. Every service name, number, and timestamp must appear "
        "verbatim in the evidence log; if the evidence never named which specific instance, "
        "write 'a redis node' rather than guessing one"
    )
    confidence: Literal["confirmed", "best_guess", "inconclusive"]
    reasoning: str
    cited_evidence_indices: list[int] = Field(
        description="ALL evidence-log indices that support this conclusion - cite every "
        "independent piece of evidence that backs it up, not just one, so the real amount of "
        "corroboration is visible"
    )


class ConcludeInvestigation(BaseModel):
    """Call this when there is sufficient evidence to conclude, or the tools available can no longer tell you anything new."""

    reason: str = Field(description="brief reason you're ready to conclude now")


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def llm_call_meta(response, t0: float) -> dict:
    """Timing, usage, cost and identity of one model reply (None or empty when the endpoint omits them)."""
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = response.response_metadata or {}
    cost = (metadata.get("token_usage") or {}).get("cost")
    generation_id = metadata.get("id")
    return {
        "llm_ms": (time.monotonic() - t0) * 1000,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": (usage.get("output_token_details") or {}).get("reasoning"),
        "cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
        "generation_ids": [generation_id] if generation_id else [],
        "served_model": metadata.get("model_name"),
        "n_tool_calls_emitted": len(getattr(response, "tool_calls", None) or []),
        "finish_reason": metadata.get("finish_reason"),
    }


class ProviderErrorResponse(RuntimeError):
    """The model endpoint answered without a completion (an HTTP 200 whose body is an `error`)."""


async def invoke_model(model, messages):
    """model.ainvoke, with langchain-openai's ValueError for an `error` body raised as ProviderErrorResponse."""
    try:
        return await model.ainvoke(messages)
    except ValueError as error:
        raise ProviderErrorResponse(f"model endpoint returned an error instead of a completion: {error}") from error


class ZeroCostViolation(RuntimeError):
    """A reply on a zero-cost (free) arm reported a cost above zero; the run stops here."""


def zero_cost_guard(meta: dict) -> None:
    """Raise at the first reply that reports a cost above zero (replies without a cost are audited later)."""
    if meta.get("cost_usd") is not None and meta["cost_usd"] > 0:
        raise ZeroCostViolation(f"reply {meta.get('generation_ids')} from {meta.get('served_model')!r} "
                                f"reported cost {meta['cost_usd']} USD on a zero-cost arm")


def merge_call_meta(wasted: dict, final: dict) -> dict:
    """One logical turn that took two attempts: sums usage and cost, keeps the final reply's identity."""
    def total(key):
        values = [m[key] for m in (wasted, final) if m[key] is not None]
        return sum(values) if values else None

    return {**final, "llm_ms": wasted["llm_ms"] + final["llm_ms"],
            "input_tokens": (wasted["input_tokens"] or 0) + (final["input_tokens"] or 0),
            "output_tokens": (wasted["output_tokens"] or 0) + (final["output_tokens"] or 0),
            "reasoning_tokens": total("reasoning_tokens"), "cost_usd": total("cost_usd"),
            "generation_ids": wasted["generation_ids"] + final["generation_ids"]}


def json_instruction(schema: type[BaseModel]) -> HumanMessage:
    return HumanMessage(content=(
        "Respond with ONLY a single valid JSON object matching this schema - "
        "no markdown code fences, no explanation, no text before or after it:\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    ))


def parse_json_reply(schema: type[SchemaT], content: str) -> SchemaT:
    """Parse a JSON reply, tolerating fences, surrounding prose, one wrapping key or a copied "properties" layout."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"no JSON object in the reply for {schema.__name__}") from None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(f"unparseable JSON for {schema.__name__}: {error}") from None

    candidates = [data]
    if isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), dict):
        candidates.append(next(iter(data.values())))
    if isinstance(data, dict) and isinstance(data.get("properties"), dict):
        candidates.append(data["properties"])
    for candidate in candidates:
        try:
            return schema.model_validate(candidate)
        except ValidationError:
            continue
    raise ValueError(f"JSON did not match {schema.__name__}: {data!r}")


def render_evidence(evidence_log: list[EvidenceEntry]) -> str:
    return "\n".join(
        f"{i}. [{e['observed_at']} {e['tool_name']}({e['args']})] {e['result']}"
        for i, e in enumerate(evidence_log)
    )


def build_graph(llm, tools: list[BaseTool], call_guard=None, system_prompt: str = SYSTEM_PROMPT):
    """Compile the graph; call_guard(meta), if given, sees every reply's llm_call_meta and may raise to stop the run."""
    tools_by_name = {t.name: t for t in tools}

    def checked_meta(response, t0: float) -> dict:
        meta = llm_call_meta(response, t0)
        if call_guard is not None:
            call_guard(meta)
        return meta

    async def baseline(state: InvestigationState) -> dict:
        events: list[dict] = []
        evidence: list[EvidenceEntry] = []

        async def run(name: str, args: dict) -> str:
            tool = tools_by_name.get(name)
            observation = await observe(tool, args)
            evidence.append({"tool_name": name, "args": args, "result": observation["result"],
                             "observed_at": observation["observed_at"]})
            events.append(trace_event("tool_call", node="baseline", tool_name=name, args=args,
                                      **observation))
            return observation["result"]

        await run("get_container_state", {})
        await run("get_metrics", {"promql": "up", "window": "1h"})

        return {
            "evidence_log": state["evidence_log"] + evidence,
            "calls_used": state["calls_used"] + len(events),
            "trace": state["trace"] + events,
        }

    def investigate_prompt(state: InvestigationState) -> str:
        evidence_text = render_evidence(state["evidence_log"]) or "(none gathered yet)"
        remaining = TOOL_CALL_BUDGET - state["calls_used"]

        prompt = (
            f"Operational assessment request: {state['trigger']}\n\n"
            f"Evidence gathered so far (entries 0+ are a counted baseline check "
            f"of container state and scrape health - read it before deciding your next move):\n"
            f"{evidence_text}\n\n"
            f"Tool calls remaining: {remaining} of {TOOL_CALL_BUDGET}.\n\n"
            f"You have read-only tools: {', '.join(sorted(tools_by_name))}. Investigate freely - call "
            "whatever tool would tell you something new about the current state. Test a suspected "
            "fault against alternatives; a healthy state is possible.\n"
            "Call ConcludeInvestigation when the evidence supports a report or the available tools "
            "cannot resolve the uncertainty."
        )
        return prompt

    async def investigate(state: InvestigationState) -> dict:
        prompt = investigate_prompt(state)
        router_llm = llm.bind_tools(tools + [ConcludeInvestigation])
        t0 = time.monotonic()
        response = await invoke_model(router_llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ])
        llm_meta = checked_meta(response, t0)

        # No tool call normally means "conclude", but two cases are retried once instead:
        # a reply cut off by the length limit, and a completely empty reply (an
        # infrastructure failure, not the model's choice). Prose without a tool call
        # is still a real decision to conclude.
        truncated_retry = False
        empty_retry = False
        initial_model_response = None
        if not response.tool_calls:
            if response.response_metadata.get("finish_reason") == "length":
                truncated_retry = True
            elif not str(response.content or "").strip():
                empty_retry = True
        if truncated_retry or empty_retry:
            initial_model_response = {"content": response.content, "tool_calls": response.tool_calls,
                                      "finish_reason": response.response_metadata.get("finish_reason")}
            wasted_meta = llm_meta
            t0 = time.monotonic()
            response = await invoke_model(router_llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ])
            llm_meta = merge_call_meta(wasted_meta, checked_meta(response, t0))

        if not response.tool_calls:
            action: PendingAction = {"type": "conclude", "name": None, "args": None}
        else:
            call = response.tool_calls[0]
            if call["name"] == "ConcludeInvestigation":
                action = {"type": "conclude", "name": None, "args": None}
            else:
                action = {"type": "tool_call", "name": call["name"], "args": call["args"]}

        event = trace_event(
            "action_selected", node="investigate", action=action,
            calls_used=state["calls_used"], truncated_retry=truncated_retry,
            empty_retry=empty_retry,
            attempts=2 if initial_model_response else 1,
            agent_visible_prompt=prompt,
            model_response={"content": response.content, "tool_calls": response.tool_calls,
                            "finish_reason": response.response_metadata.get("finish_reason")},
            initial_model_response=initial_model_response,
            **llm_meta,
        )
        return {"pending_action": action, "trace": state["trace"] + [event]}

    def route_after_investigate(state: InvestigationState) -> Literal["tool_call", "conclude"]:
        if state["calls_used"] >= TOOL_CALL_BUDGET:
            return "conclude"
        action = state["pending_action"]
        if action is not None and action["type"] == "tool_call":
            return "tool_call"
        return "conclude"

    def route_after_tool(state: InvestigationState) -> Literal["investigate", "conclude"]:
        return "conclude" if state["calls_used"] >= TOOL_CALL_BUDGET else "investigate"

    async def tool_call(state: InvestigationState) -> dict:
        action = state["pending_action"]

        dup_idx = next(
            (i for i, e in enumerate(state["evidence_log"])
             if e["tool_name"] == action["name"] and e["args"] == action["args"]),
            None,
        )
        tool_ms = 0.0
        observed_at = datetime.now(timezone.utc).isoformat()
        if dup_idx is not None:
            text = f"DUPLICATE: identical call already made - see evidence #{dup_idx} for its result, nothing new here"
        else:
            tool = tools_by_name.get(action["name"])
            observation = await observe(tool, action["args"])
            text = observation["result"]
            tool_ms = observation["tool_ms"]
            observed_at = observation["observed_at"]

        entry: EvidenceEntry = {"tool_name": action["name"], "args": action["args"],
                                "result": text, "observed_at": observed_at}
        calls_used = state["calls_used"] + 1
        event = trace_event(
            "tool_call", node="tool_call", tool_name=action["name"], args=action["args"],
            result=text, observed_at=observed_at, calls_used=calls_used, tool_ms=tool_ms,
        )
        return {
            "evidence_log": state["evidence_log"] + [entry],
            "calls_used": calls_used,
            "pending_action": None,
            "trace": state["trace"] + [event],
        }

    async def conclude(state: InvestigationState) -> dict:
        evidence_text = render_evidence(state["evidence_log"]) or "(no tool evidence gathered)"
        budget_note = (
            "Budget exhausted." if state["calls_used"] >= TOOL_CALL_BUDGET
            else f"{TOOL_CALL_BUDGET - state['calls_used']} tool calls remained unused."
        )

        final_prompt = (
                    f"Investigation is ending. {budget_note}\n\n"
                    f"Evidence log (the ONLY things you may cite as proof):\n{evidence_text}\n\n"
                    "Write the final incident report.\n"
                    "GROUNDING RULE: every specific identifier you write - service or container name "
                    "(redis1 vs redis2 vs redis3), IP, port, count, timestamp, config value - must "
                    "appear verbatim in the evidence log above. If the evidence log didn't say which "
                    "one, write the general form - 'a redis cluster node', 'one of the three masters' "
                    "- instead of picking a name. A correct vague report scores better than a specific "
                    "wrong one.\n"
                    "Only mark confidence='confirmed' if the evidence you cite spans at least 2 "
                    "distinct tools (e.g. get_metrics AND read_config, not two query_logs calls) - "
                    "cite every piece of evidence that genuinely backs your conclusion, not just one."
        )
        # A failed model call propagates (the run is recorded as crashed); only an unusable reply is a parse failure.
        t0 = time.monotonic()
        response = await invoke_model(llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=final_prompt),
            json_instruction(ConcludeOut),
        ])
        llm_meta = checked_meta(response, t0)
        raw_response = str(response.content)
        try:
            result = parse_json_reply(ConcludeOut, raw_response)
        except ValueError as e:
            conclusion = {
                "root_cause_id": "PARSE_FAILURE",
                "summary": "",
                "confidence": "inconclusive",
                "model_self_reported_confidence": "inconclusive",
                "reasoning": f"conclude() failed: {e}",
                "cited_evidence_indices": [],
                "distinct_supporting_tools": [],
                "calls_used": state["calls_used"],
                "budget": TOOL_CALL_BUDGET,
                "downgrade_reason": None,
            }
            event = trace_event(
                "conclusion", node="conclude", conclusion=conclusion, parse_failure=True,
                agent_visible_prompt=final_prompt,
                response_schema=ConcludeOut.model_json_schema(),
                model_response=raw_response, **llm_meta,
            )
            return {"conclusion": conclusion, "trace": state["trace"] + [event]}

        valid_indices = sorted({i for i in result.cited_evidence_indices if 0 <= i < len(state["evidence_log"])})
        supporting_indices = [i for i in valid_indices if usable_evidence(state["evidence_log"][i])]
        distinct_supporting_tools = sorted({state["evidence_log"][i]["tool_name"] for i in supporting_indices})
        baseline_count = sum(e.get("node") == "baseline" and e.get("event") == "tool_call" for e in state["trace"])
        has_agent_observation = any(i >= baseline_count for i in supporting_indices)

        corroboration_shortfall = result.confidence == "confirmed" and (
            len(distinct_supporting_tools) < CORROBORATION_THRESHOLD or not has_agent_observation
        )

        enforced_confidence = result.confidence
        downgrade_reason = None
        if corroboration_shortfall:
            enforced_confidence = "best_guess"
            downgrade_reason = "corroboration"

        conclusion = {
            "root_cause_id": result.root_cause_id,
            "summary": result.summary,
            "confidence": enforced_confidence,
            "model_self_reported_confidence": result.confidence,
            "reasoning": result.reasoning,
            "cited_evidence_indices": valid_indices,
            "distinct_supporting_tools": distinct_supporting_tools,
            "calls_used": state["calls_used"],
            "budget": TOOL_CALL_BUDGET,
            "downgrade_reason": downgrade_reason,
        }
        event = trace_event("conclusion", node="conclude", conclusion=conclusion,
                            agent_visible_prompt=final_prompt,
                            response_schema=ConcludeOut.model_json_schema(),
                            model_response=raw_response, **llm_meta)
        return {"conclusion": conclusion, "trace": state["trace"] + [event]}

    graph = StateGraph(InvestigationState)
    graph.add_node("baseline", baseline)
    graph.add_node("investigate", investigate)
    graph.add_node("tool_call", tool_call)
    graph.add_node("conclude", conclude)

    graph.add_edge(START, "baseline")
    graph.add_edge("baseline", "investigate")
    graph.add_conditional_edges("investigate", route_after_investigate, {
        "tool_call": "tool_call",
        "conclude": "conclude",
    })
    graph.add_conditional_edges("tool_call", route_after_tool, {
        "investigate": "investigate", "conclude": "conclude",
    })
    graph.add_edge("conclude", END)

    return graph.compile()
