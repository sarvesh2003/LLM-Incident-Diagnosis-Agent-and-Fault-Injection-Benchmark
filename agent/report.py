"""Build the result.json summary (schema 2.2) from one investigation's final state and trace."""

import hashlib
import json
from urllib.parse import urlparse

from agent.config import AgentConfig
from agent.graph import TOOL_CALL_BUDGET


def build_result(
    config: AgentConfig, trigger: str, final_state: dict | None, wall_clock_s: float,
    status: str, error_type: str = "", traceback_tail: str = "",
    vram_pre: dict | None = None, vram_post: dict | None = None,
    tool_specs: list[dict] | None = None,
) -> dict:
    result: dict = {
        "schema_version": "2.2",
        "run_id": config.run_tag,
        "env_label": config.env_label,
        "provider": config.provider,
        "model_id": config.model_id,
        "endpoint_host": urlparse(config.base_url).hostname,
        "trigger": trigger,
        "status": status,
        "error_type": error_type,
        "traceback_tail": traceback_tail,
        "wall_clock_s": wall_clock_s,
        "vram_pre": vram_pre,
        "vram_post": vram_post,
        "model_settings": config.model_settings.as_dict(),
        "tool_schema_sha256": hashlib.sha256(
            json.dumps(tool_specs or [], sort_keys=True).encode("utf-8")).hexdigest(),
    }

    if final_state is None:
        result.update({
            "calls_used": None, "budget_exhausted": None,
            "llm_calls": 0, "llm_attempts": 0, "llm_ms_total": 0.0, "tool_ms_total": 0.0,
            "input_tokens_total": 0, "output_tokens_total": 0,
            "reasoning_tokens_total": 0, "cost_usd_total": None, "cost_reported_calls": 0,
            "generation_ids": [], "served_models": [], "tool_calls_dropped": 0,
            "truncated_retries": 0, "empty_retries": 0, "baseline_evidence_count": 0,
            "tool_call_sequence": [],
            "distinct_tools_used_total": [], "duplicate_calls": 0,
            "tool_errors": 0, "hallucinated_tool_calls": 0, "empty_result_calls": 0,
            "root_cause_id": None, "summary": "", "reasoning": "",
            "confidence": None, "model_self_reported_confidence": None,
            "confidence_was_downgraded": None, "downgrade_reason": None, "cited_evidence_indices": [],
            "distinct_supporting_tools": [], "n_cited": 0,
            "cited_baseline_fraction": None,
        })
        return result

    trace = final_state.get("trace", [])
    evidence_log = final_state.get("evidence_log", [])
    conclusion = final_state.get("conclusion") or {}

    llm_events = [e for e in trace if e.get("event") in ("action_selected", "conclusion") and "llm_ms" in e]
    tool_events = [e for e in trace if e.get("event") == "tool_call"]
    baseline_events = [e for e in trace if e.get("node") == "baseline" and e.get("event") == "tool_call"]

    tool_call_sequence = [e["tool_name"] for e in trace if e.get("event") == "tool_call" and e.get("node") == "tool_call"]
    duplicate_calls = sum(1 for e in evidence_log if str(e.get("result", "")).startswith("DUPLICATE:"))
    tool_errors = sum(1 for e in evidence_log if str(e.get("result", "")).startswith("TOOL ERROR:")
                      and "requested tool is unavailable" not in str(e.get("result", "")))
    hallucinated_tool_calls = sum(1 for e in evidence_log if "requested tool is unavailable" in str(e.get("result", "")))
    empty_result_calls = sum(1 for e in evidence_log if str(e.get("result", "")).strip().startswith(("(no", "(0 of")))
    truncated_retries = sum(1 for e in trace if e.get("event") == "action_selected" and e.get("truncated_retry"))
    # Kept apart from truncated_retries: an empty reply is an infrastructure failure, not truncation.
    empty_retries = sum(1 for e in trace if e.get("event") == "action_selected" and e.get("empty_retry"))

    valid_indices = conclusion.get("cited_evidence_indices", [])
    baseline_count = len(baseline_events)
    cited_baseline_fraction = (
        sum(1 for i in valid_indices if i < baseline_count) / len(valid_indices)
        if valid_indices else None
    )

    summary = conclusion.get("summary", "") or ""
    costs = [e["cost_usd"] for e in llm_events if e.get("cost_usd") is not None]

    result.update({
        "calls_used": final_state.get("calls_used"),
        "budget_exhausted": final_state.get("calls_used", 0) >= TOOL_CALL_BUDGET,
        "llm_calls": len(llm_events),
        # A retried turn counts twice.
        "llm_attempts": sum(e.get("attempts", 1) for e in llm_events),
        "llm_ms_total": sum(e.get("llm_ms", 0) or 0 for e in llm_events),
        "tool_ms_total": sum(e.get("tool_ms", 0) or 0 for e in tool_events),
        "input_tokens_total": sum(e.get("input_tokens") or 0 for e in llm_events),
        "output_tokens_total": sum(e.get("output_tokens") or 0 for e in llm_events),
        "reasoning_tokens_total": sum(e.get("reasoning_tokens") or 0 for e in llm_events),
        # None when no reply reported a cost (non-OpenRouter endpoints).
        "cost_usd_total": sum(costs) if costs else None,
        "cost_reported_calls": len(costs),
        "generation_ids": [gid for e in llm_events for gid in e.get("generation_ids", [])],
        "served_models": sorted({e["served_model"] for e in llm_events if e.get("served_model")}),
        # Tool calls beyond the first in one reply, which the agent ignores.
        "tool_calls_dropped": sum(max(0, (e.get("n_tool_calls_emitted") or 0) - 1)
                                  for e in llm_events if e.get("event") == "action_selected"),
        "truncated_retries": truncated_retries,
        "empty_retries": empty_retries,
        "baseline_evidence_count": baseline_count,
        "tool_call_sequence": tool_call_sequence,
        "distinct_tools_used_total": sorted({e["tool_name"] for e in evidence_log}),
        "duplicate_calls": duplicate_calls,
        "tool_errors": tool_errors,
        "hallucinated_tool_calls": hallucinated_tool_calls,
        "empty_result_calls": empty_result_calls,
        "root_cause_id": conclusion.get("root_cause_id"),
        "summary": summary,
        "reasoning": conclusion.get("reasoning", ""),
        "confidence": conclusion.get("confidence"),
        "model_self_reported_confidence": conclusion.get("model_self_reported_confidence"),
        "confidence_was_downgraded": conclusion.get("confidence") != conclusion.get("model_self_reported_confidence"),
        "downgrade_reason": conclusion.get("downgrade_reason"),
        "cited_evidence_indices": valid_indices,
        "distinct_supporting_tools": conclusion.get("distinct_supporting_tools", []),
        "n_cited": len(valid_indices),
        "cited_baseline_fraction": cited_baseline_fraction,
    })
    return result


