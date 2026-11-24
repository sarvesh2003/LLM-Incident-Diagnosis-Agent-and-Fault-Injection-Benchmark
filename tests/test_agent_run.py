"""Artifact persistence and setup for one agent invocation, with a fake model and tools."""

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage

from agent.config import AgentConfig
from agent.prompts import SYSTEM_PROMPT, V2_ADDENDUM, prompt_sha256, system_prompt_for
from agent.run import REQUIRED_TOOLS, build_llm, investigate, mcp_server_env
from agent.tools import MAX_OBSERVATION_CHARS
from tests.agent_fakes import CONCLUDE, FakeTools, ScriptedModel, report, tool_call


class RunArtifactTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.config = AgentConfig.from_env({
            "MODEL_BASE_URL": "http://127.0.0.1:9/v1", "MODEL_ID": "fake-model",
            "RUN_TAG": "unit_run", "RESULTS_DIR": self.folder.name})

    def run_with(self, model, tools=None):
        tools = tools or FakeTools()

        async def connect(config):
            return model, tools.build()

        return asyncio.run(investigate(self.config, "Assess current health", connect))

    def artifacts(self):
        root = Path(self.folder.name)
        result = json.loads((root / "unit_run.result.json").read_text(encoding="utf-8"))
        trace = [json.loads(line) for line in (root / "unit_run.trace.jsonl").read_text(encoding="utf-8").splitlines()]
        return result, trace

    def test_completed_run_writes_versioned_result_and_trace(self):
        self.run_with(ScriptedModel([tool_call("query_logs", {"service": "go-server", "since": "5m"}), CONCLUDE],
                                    report([0, 2])))
        result, trace = self.artifacts()
        self.assertEqual((result["status"], result["schema_version"]), ("completed", "2.2"))
        self.assertEqual(result["confidence"], "confirmed")
        self.assertEqual(result["calls_used"], 3)
        self.assertEqual(result["llm_calls"], 3)       # two investigate turns + conclusion
        self.assertEqual(result["llm_attempts"], 3)
        self.assertEqual(result["model_settings"]["temperature"], 0.2)
        self.assertEqual(len(result["tool_schema_sha256"]), 64)
        self.assertEqual(trace[0]["event"], "investigation_started")
        self.assertEqual({spec["name"] for spec in trace[0]["tool_specs"]}, REQUIRED_TOOLS)
        self.assertIn("system_prompt", trace[0])
        for key in ("fault_id", "fault_name", "fault_family"):
            self.assertNotIn(key, trace[0])

    def test_cost_usage_and_generation_ids_are_totalled(self):
        def reply(gid, cost, calls):
            return AIMessage(content="", tool_calls=calls, usage_metadata={
                "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                "output_token_details": {"reasoning": 4}},
                response_metadata={"finish_reason": "tool_calls", "id": gid, "model_name": "vendor/model",
                                   "token_usage": {"cost": cost}})
        two_calls = [{"name": "query_logs", "args": {"service": "go-server", "since": "5m"}, "id": "a"},
                     {"name": "read_config", "args": {"service": "go-server"}, "id": "b"}]
        conclude = [{"name": "ConcludeInvestigation", "args": {"reason": "done"}, "id": "c"}]
        self.run_with(ScriptedModel([reply("gen-1", 0.001, two_calls), reply("gen-2", 0.002, conclude)]))
        result, trace = self.artifacts()
        self.assertEqual(result["generation_ids"], ["gen-1", "gen-2"])  # the scripted conclusion has no id
        self.assertAlmostEqual(result["cost_usd_total"], 0.003)
        self.assertEqual((result["cost_reported_calls"], result["reasoning_tokens_total"]), (2, 8))
        self.assertEqual(result["served_models"], ["vendor/model"])
        self.assertEqual(result["tool_calls_dropped"], 1)
        self.assertEqual(trace[3]["n_tool_calls_emitted"], 2)

    def test_retried_turn_merges_cost_and_both_generation_ids(self):
        empty = AIMessage(content="", usage_metadata={"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
                          response_metadata={"finish_reason": "stop", "id": "gen-empty", "token_usage": {"cost": 0.5}})
        retried = AIMessage(content="", tool_calls=[{"name": "ConcludeInvestigation", "args": {"reason": "x"}, "id": "c"}],
                            response_metadata={"finish_reason": "tool_calls", "id": "gen-ok", "token_usage": {"cost": 0.25}})
        self.run_with(ScriptedModel([empty, retried]))
        result, trace = self.artifacts()
        turn = next(e for e in trace if e.get("event") == "action_selected")
        self.assertEqual((turn["attempts"], turn["generation_ids"], turn["cost_usd"]), (2, ["gen-empty", "gen-ok"], 0.75))
        self.assertEqual(result["cost_usd_total"], 0.75)

    def free_config(self, require_zero_cost="true"):
        return AgentConfig.from_env({
            "MODEL_BASE_URL": "https://openrouter.ai/api/v1", "MODEL_ID": "vendor/model:free",
            "RUN_TAG": "unit_run", "RESULTS_DIR": self.folder.name, "MODEL_MAX_PRICE": "0,0",
            "MODEL_REQUIRE_ZERO_COST": require_zero_cost})

    def test_zero_cost_guard_stops_a_free_run_at_the_first_charged_reply(self):
        def reply(gid, cost):
            return AIMessage(content="", tool_calls=[{"name": "query_logs", "args": {"service": "go-server",
                                                                                     "since": "5m"}, "id": gid}],
                             response_metadata={"finish_reason": "tool_calls", "id": gid, "model_name": "vendor/model:free",
                                                "token_usage": {"cost": cost}})
        self.config = self.free_config()
        model = ScriptedModel([reply("gen-1", 0.0), reply("gen-2", 0.0004), reply("gen-3", 0.0)])
        self.run_with(model)
        result, _ = self.artifacts()
        self.assertEqual((result["status"], result["error_type"]), ("crashed", "ZeroCostViolation"))
        self.assertIn("gen-2", result["traceback_tail"])
        self.assertEqual(model.calls, 2)  # nothing is sent after the charged reply
        # Without the guard the same replies run to the end.
        self.config = self.free_config("false")
        self.run_with(ScriptedModel([reply("gen-1", 0.0), reply("gen-2", 0.0004)]))
        self.assertEqual(self.artifacts()[0]["status"], "completed")

    def test_free_arm_routing_sends_max_price_zero(self):
        llm = build_llm(self.free_config())
        self.assertEqual(llm.extra_body, {"provider": {"allow_fallbacks": True,
                                                       "max_price": {"prompt": 0.0, "completion": 0.0}}})
        with self.assertRaisesRegex(ValueError, "MODEL_MAX_PRICE"):
            AgentConfig.from_env({"MODEL_BASE_URL": "http://x/v1", "MODEL_ID": "m", "MODEL_MAX_PRICE": "0"})
        with self.assertRaisesRegex(ValueError, "MODEL_REQUIRE_ZERO_COST"):
            AgentConfig.from_env({"MODEL_BASE_URL": "http://x/v1", "MODEL_ID": "m", "MODEL_REQUIRE_ZERO_COST": "yes"})

    def test_endpoint_without_cost_reports_none(self):
        self.run_with(ScriptedModel([CONCLUDE]))
        result, _ = self.artifacts()
        self.assertEqual((result["cost_usd_total"], result["cost_reported_calls"], result["generation_ids"]),
                         (None, 0, []))

    def test_model_failure_still_persists_trace_and_crash(self):
        self.run_with(ScriptedModel([ConnectionError("endpoint refused")]))
        result, trace = self.artifacts()
        self.assertEqual((result["status"], result["error_type"]), ("crashed", "ConnectionError"))
        self.assertIn("endpoint refused", result["traceback_tail"])
        self.assertEqual([e["node"] for e in trace[1:]], ["baseline", "baseline"])

    def test_setup_failure_is_recorded(self):
        async def connect(config):
            raise RuntimeError("MCP server did not provide the five required read-only tools")

        asyncio.run(investigate(self.config, "Assess current health", connect))
        result, trace = self.artifacts()
        self.assertEqual(result["status"], "crashed")
        self.assertEqual(trace[0]["tool_specs"], [])
        self.assertIsNone(result["calls_used"])

    def test_interrupt_is_recorded_then_re_raised(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_with(ScriptedModel([KeyboardInterrupt()]))
        result, _ = self.artifacts()
        self.assertEqual(result["status"], "interrupted")

    def run_recording_system_prompts(self, env_extra):
        self.config = AgentConfig.from_env({
            "MODEL_BASE_URL": "http://127.0.0.1:9/v1", "MODEL_ID": "fake-model",
            "RUN_TAG": "unit_run", "RESULTS_DIR": self.folder.name, **env_extra})
        seen = []

        class Recording(ScriptedModel):
            async def ainvoke(self, messages):
                seen.append(messages[0].content)
                return await super().ainvoke(messages)

        self.run_with(Recording([tool_call("query_logs", {"service": "go-server", "since": "5m"}), CONCLUDE]))
        return seen, self.artifacts()[1][0]

    def test_v1_prompt_is_the_unchanged_system_prompt_when_the_variable_is_absent(self):
        original = (Path(__file__).resolve().parent.parent / "agent" / "system_prompt.md").read_text(encoding="utf-8")
        self.assertEqual(SYSTEM_PROMPT, original)
        self.assertEqual(system_prompt_for("v1"), original)
        for env_extra in ({}, {"AGENT_PROMPT_VARIANT": ""}, {"AGENT_PROMPT_VARIANT": "v1"}):
            seen, header = self.run_recording_system_prompts(env_extra)
            self.assertEqual(len(seen), 3)                       # two investigate turns + conclusion
            self.assertTrue(all(prompt == original for prompt in seen))
            self.assertEqual((header["prompt_variant"], header["system_prompt"]), ("v1", original))
            self.assertEqual(header["system_prompt_sha256"],
                             hashlib.sha256(original.encode("utf-8")).hexdigest())

    def test_v2_appends_the_golden_signal_addendum_everywhere(self):
        seen, header = self.run_recording_system_prompts({"AGENT_PROMPT_VARIANT": "v2"})
        self.assertEqual(len(seen), 3)
        for prompt in seen:
            self.assertTrue(prompt.startswith(SYSTEM_PROMPT.rstrip("\n")))
            self.assertTrue(prompt.endswith(V2_ADDENDUM))
            self.assertIn("four golden signals", prompt)
        self.assertEqual(header["prompt_variant"], "v2")
        self.assertEqual(header["system_prompt_sha256"], prompt_sha256("v2"))
        self.assertNotEqual(prompt_sha256("v2"), prompt_sha256("v1"))
        for word in ("netem", "inject", "delay", "go-server", "go-enricher", "fault"):
            self.assertNotIn(word, V2_ADDENDUM.lower())        # generic guidance; names no fault or service
        with self.assertRaisesRegex(ValueError, "AGENT_PROMPT_VARIANT"):
            AgentConfig.from_env({"MODEL_BASE_URL": "http://x/v1", "MODEL_ID": "m", "AGENT_PROMPT_VARIANT": "v3"})


class SetupTests(unittest.TestCase):
    def test_mcp_server_receives_the_observation_limit_and_stack_settings(self):
        env = mcp_server_env({"COMPOSE_PROJECT_NAME": "fraud_test", "PROMETHEUS_URL": "http://localhost:19090"})
        self.assertEqual(env["OBSERVATION_MAX_CHARS"], str(MAX_OBSERVATION_CHARS))
        self.assertEqual(env["COMPOSE_PROJECT_NAME"], "fraud_test")
        self.assertEqual(env["PROMETHEUS_URL"], "http://localhost:19090")

    def test_model_settings_have_one_source(self):
        base = {"MODEL_BASE_URL": "http://localhost:8000/v1", "MODEL_ID": "m"}
        settings = AgentConfig.from_env(base).model_settings
        self.assertEqual(settings.as_dict(), {"temperature": 0.2, "max_tokens": 8192, "timeout_seconds": 120,
                                              "max_retries": 1, "reasoning_effort": None, "seed": None,
                                              "provider_order": (), "allow_fallbacks": True,
                                              "quantizations": (), "data_collection": None, "max_price": None,
                                              "require_zero_cost": False})
        self.assertIsNone(settings.provider_routing())
        self.assertEqual(AgentConfig.from_env({**base, "MODEL_REASONING_EFFORT": "none"}).model_settings.reasoning_effort,
                         "none")
        with self.assertRaisesRegex(ValueError, "MODEL_REASONING_EFFORT"):
            AgentConfig.from_env({**base, "MODEL_REASONING_EFFORT": "extreme"})

    def test_routing_and_seed_reach_the_client_only_when_set(self):
        base = {"MODEL_BASE_URL": "https://openrouter.ai/api/v1", "MODEL_ID": "z-ai/glm-5.3"}
        plain = build_llm(AgentConfig.from_env(base))
        self.assertIsNone(plain.extra_body)
        self.assertIsNone(plain.seed)
        pinned = build_llm(AgentConfig.from_env({
            **base, "MODEL_PROVIDER_ORDER": "z-ai", "MODEL_QUANTIZATIONS": "fp8",
            "MODEL_ALLOW_FALLBACKS": "false", "MODEL_DATA_COLLECTION": "deny", "MODEL_SEED": "7"}))
        self.assertEqual(pinned.extra_body, {"provider": {"allow_fallbacks": False, "order": ["z-ai"],
                                                          "quantizations": ["fp8"], "data_collection": "deny"}})
        self.assertEqual(pinned.seed, 7)

    def test_legacy_ollama_aliases_are_not_read(self):
        with self.assertRaisesRegex(ValueError, "MODEL_BASE_URL"):
            AgentConfig.from_env({"OLLAMA_BASE_URL": "http://localhost:11434/v1", "OLLAMA_MODEL": "qwen"})


if __name__ == "__main__":
    unittest.main()
