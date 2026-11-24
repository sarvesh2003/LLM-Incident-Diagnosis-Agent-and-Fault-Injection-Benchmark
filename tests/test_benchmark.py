"""Benchmark tooling without Docker or a model: generation audit, schedule and freeze, batch logic, rules, aggregation."""

import csv
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from experiments import aggregate, auto_label, baseline_rule, batch, free_tier, openrouter_audit, preregister, runner
from experiments.fake_model import serve_in_background

CRASHED = ["agent result status is 'crashed', not completed"]
LUNA_WORST = (98000 * 0.10 + 23600 * 0.50) / 1e6   # token-model worst case per run
GLM_WORST = (98000 * 1.40 + 23600 * 4.40) / 1e6

HEALTHY_STATE = "\n".join(f"{s}: state=running, status=\"Up 6 minutes\"" for s in runner.REQUIRED_SERVICES) + \
    "\nredis-cluster-setup: state=exited, status=\"Exited (0) 6 minutes ago\""


def state_with(service, state):
    return HEALTHY_STATE.replace(f"{service}: state=running", f"{service}: state={state}")


def write_run(runs, run_id, case, *, arm="luna@openai", row=None, status="completed", errors=(), result=None,
              observation=HEALTHY_STATE, audit=None, created="2026-11-23T00:00:00+00:00", extra=None):
    folder = Path(runs) / run_id
    folder.mkdir(parents=True)
    manifest = {"run_id": run_id, "case": case, "status": status,
                "validity_errors": list(errors), "arm": {"id": arm}, "model_id": "m", "created_at_utc": created,
                "agent_tool_schema_sha256": "a" * 64, **(extra or {})}
    if row:
        manifest["schedule_row"] = row
    if audit:
        manifest["generation_audit"] = audit
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    events = [{"event": "investigation_started"},
              {"event": "tool_call", "node": "baseline", "tool_name": "get_container_state", "result": observation}]
    (folder / f"{run_id}.trace.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    if result is not None:
        (folder / f"{run_id}.result.json").write_text(json.dumps(result), encoding="utf-8")
    return folder


def result(root="none", summary="", confidence="inconclusive", cost=0.01, status="completed", **extra):
    return {"status": status, "root_cause_id": root, "summary": summary, "confidence": confidence,
            "cost_usd_total": cost, "calls_used": 4, "wall_clock_s": 60.0, "llm_ms_total": 900.0,
            "llm_calls": 3, "confidence_was_downgraded": False, **extra}


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.runs = Path(self.folder.name)


class AuditTests(TempDirTest):
    def test_audit_against_the_offline_endpoint_records_provider_and_cost(self):
        server, url = serve_in_background()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        body = json.dumps({"model": "offline-fake", "messages": [{"role": "user", "content": "hello"}],
                           "provider": {"order": ["openai"], "allow_fallbacks": False}}).encode()
        request = urllib.request.Request(url + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            reply = json.load(response)
        self.assertGreater(reply["usage"]["cost"], 0)
        folder = self.runs / "inc-1"
        folder.mkdir()
        trace = [{"event": "investigation_started"},
                 {"event": "action_selected", "generation_ids": [reply["id"]]},
                 {"event": "conclusion", "generation_ids": ["gen-missing"]}]
        (folder / "inc-1.trace.jsonl").write_text("\n".join(json.dumps(e) for e in trace), encoding="utf-8")
        summary, errors = openrouter_audit.audit_run(folder, "inc-1", url, "secret-key", ["openai"],
                                                     sleep=lambda s: None, attempts=2)
        self.assertEqual(summary["providers_served"], ["openai"])
        self.assertAlmostEqual(summary["total_cost_usd"], reply["usage"]["cost"])
        self.assertEqual(summary["missing"], ["gen-missing"])
        self.assertEqual(errors, ["provider audit incomplete: 1 of 2 generations not retrievable"])
        written = (folder / "generations.json").read_text(encoding="utf-8")
        self.assertNotIn("secret-key", written)
        self.assertEqual(json.loads(written)["records"][0]["id"], reply["id"])

    def test_drift_rule_normalizes_display_names(self):
        records = [{"id": "g1", "provider_name": "Z.AI"}, {"id": "g2", "provider_name": "Novita"},
                   {"id": "g3", "provider_name": None}]
        errors = openrouter_audit.drift_errors(records, ["z-ai"])
        self.assertEqual(len(errors), 2)
        self.assertIn("g2 served by 'Novita', pinned ['z-ai']", errors[0])
        self.assertEqual(openrouter_audit.drift_errors(records[:1], ["z-ai/fp8"]), [])

    def test_key_failure_is_raised_and_not_retried(self):
        calls = []

        def opener(request, timeout):
            calls.append(request.get_header("Authorization"))
            raise urllib.error.HTTPError(request.full_url, 402, "Payment Required", {}, io.BytesIO(b"{}"))

        with self.assertRaisesRegex(openrouter_audit.KeyFailure, "HTTP 402") as raised:
            openrouter_audit.fetch_generation("https://x/api/v1", "secret-key", "g1", opener=opener,
                                              sleep=lambda s: None)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("secret-key", str(raised.exception))

    def test_key_limit_403_is_a_key_failure_but_429_is_not(self):
        def refuse(code):
            def opener(request, timeout):
                raise urllib.error.HTTPError(request.full_url, code, "x", {}, io.BytesIO(b"{}"))
            return opener

        with self.assertRaisesRegex(openrouter_audit.KeyFailure, "HTTP 403"):
            openrouter_audit.fetch_generation("https://x/api/v1", "k", "g1", opener=refuse(403), sleep=lambda s: None)
        # A 429 on the lookup is retried and ends as a missing generation, never a key failure.
        self.assertIsNone(openrouter_audit.fetch_generation("https://x/api/v1", "k", "g1", opener=refuse(429),
                                                            sleep=lambda s: None, attempts=2))


class PreregistrationTests(TempDirTest):
    def test_matrix_schedule_is_balanced_interleaved_and_seeded(self):
        config = preregister.load_config()
        rows = preregister.matrix_schedule(config)
        self.assertEqual(len(rows), 152)
        self.assertEqual(rows, preregister.matrix_schedule(config))
        self.assertEqual(rows, preregister.read_schedule())  # the committed schedule is this seed's output
        cells = {}
        for row in rows:
            cells[(row["arm"], row["case"])] = cells.get((row["arm"], row["case"]), 0) + 1
        hidden = ("latency_injected", "client_timeout_too_low", "cpu_throttled", "latency_enricher")
        visible = ("redis_node_stopped", "ingestion_paused", "broker_stopped", "enricher_paused")
        for arm in ("luna", "glm", "gemma", "nemotron"):
            self.assertEqual(sum(n for (a, _), n in cells.items() if a == arm), 38)
            self.assertEqual(cells[(arm, "healthy")], 6)
            for case in hidden:
                self.assertEqual(cells[(arm, case)], 6)
            for case in visible:
                self.assertEqual(cells[(arm, case)], 2)
        self.assertEqual({r["phase"] for r in rows if r["case"] in visible}, {"P3"})
        self.assertEqual({r["phase"] for r in rows if r["case"] not in visible}, {"P2"})
        paid_blocks = {r["block"] for r in rows if r["arm"] != "nemotron"}
        free_blocks = {r["block"] for r in rows if r["arm"] == "nemotron"}
        self.assertFalse(paid_blocks & free_blocks)          # the free arm has its own blocks
        self.assertLess(max(paid_blocks), min(free_blocks))
        for block in paid_blocks:
            in_block = [(r["arm"], r["case"]) for r in rows if r["block"] == block]
            phase = {r["phase"] for r in rows if r["block"] == block}
            self.assertEqual(len(phase), 1)
            self.assertEqual(len(in_block), len(set(in_block)))  # every paid arm x case of the phase once
            self.assertEqual(len(in_block), 3 * (5 if phase == {"P2"} else 4))
        self.assertNotEqual([r["arm"] for r in rows[:6]], ["luna"] * 6)

    def test_arm_environment_pins_routing_and_never_holds_a_key(self):
        config = preregister.load_config()
        arms = preregister.enabled_arms(config)
        self.assertEqual(sorted(arms), ["gemma", "glm", "luna", "nemotron"])
        free = preregister.arm_environment(config, arms["nemotron"])
        self.assertEqual((free["MODEL_ID"], free["MODEL_DATA_COLLECTION"], free["MODEL_MAX_PRICE"],
                          free["MODEL_REQUIRE_ZERO_COST"], free["MODEL_ALLOW_FALLBACKS"]),
                         ("nvidia/nemotron-3-ultra-550b-a55b:free", "allow", "0,0", "true", "false"))
        free_settings = runner.AgentConfig.from_env(free).model_settings
        self.assertEqual(free_settings.provider_routing()["max_price"], {"prompt": 0.0, "completion": 0.0})
        self.assertTrue(free_settings.require_zero_cost)
        self.assertEqual({arm["data_collection"] for key, arm in arms.items() if key != "nemotron"}, {"deny"})
        self.assertEqual({key: arm["cap_usd"] for key, arm in arms.items()},
                         {"luna": 0.60, "glm": 3.40, "gemma": 0.50, "nemotron": 0.0})
        env = preregister.arm_environment(config, arms["glm"])
        self.assertEqual((env["MODEL_ID"], env["MODEL_PROVIDER_ORDER"], env["MODEL_QUANTIZATIONS"]),
                         ("z-ai/glm-5.3", "z-ai", "fp8"))
        self.assertEqual((env["MODEL_ALLOW_FALLBACKS"], env["MODEL_DATA_COLLECTION"]), ("false", "deny"))
        self.assertEqual(env["ENV_LABEL"], "glm@z-ai-fp8")
        self.assertFalse(any("KEY" in name for name in env))
        settings = runner.AgentConfig.from_env(env).model_settings
        self.assertEqual(settings.provider_routing()["order"], ["z-ai"])

    def test_switching_a_pin_is_configuration_only_and_a_new_arm(self):
        config = preregister.load_config()
        glm = next(arm for arm in config["arms"] if arm["key"] == "glm")
        glm["pin"] = "wafer-fp4"
        arm = preregister.enabled_arms(config)["glm"]
        self.assertEqual((arm["id"], arm["provider_order"], arm["quantizations"]), ("glm@wafer-fp4", ["wafer"], ["fp4"]))
        self.assertIsNone(preregister.estimate_run_usd(arm, config["token_model"], "worst"))

    def test_cost_estimate_matches_the_plan(self):
        config = preregister.load_config()
        arms = preregister.enabled_arms(config)
        worst = preregister.estimate_run_usd(arms["luna"], config["token_model"], "worst")
        self.assertAlmostEqual(worst, (98000 * 0.10 + 23600 * 0.50) / 1e6)
        typical = preregister.estimate_run_usd(arms["gemma"], config["token_model"], "typical")
        self.assertAlmostEqual(typical, (29000 * 0.14 + 900 * 0.40) / 1e6)

    def test_freeze_detects_a_changed_protocol_file(self):
        prereg = preregister.freeze()
        self.assertEqual(preregister.protocol_mismatches(prereg), [])
        self.assertIn("fault-injection/network/09_latency_injection.sh", prereg["protocol_sha256"])
        prereg["protocol_sha256"]["agent/system_prompt.md"] = "0" * 64
        self.assertEqual(preregister.protocol_mismatches(prereg), ["agent/system_prompt.md"])

    def test_catalog_snapshot_reads_each_enabled_model(self):
        urls = []

        def opener(request, timeout):
            urls.append(request.full_url)
            return io.BytesIO(json.dumps({"data": {"endpoints": []}}).encode())

        snapshot = preregister.catalog_snapshot(preregister.load_config(), opener=opener)
        self.assertEqual(len(urls), 4)
        self.assertTrue(all(url.endswith("/endpoints") for url in urls))
        self.assertIn("google/gemma-4-31b-it", snapshot["models"])


class BatchTests(TempDirTest):
    def setUp(self):
        super().setUp()
        self.config = preregister.load_config()
        self.arms = preregister.enabled_arms(self.config)
        self.rows = [{"row": "r001", "block": 1, "arm": "luna", "case": "healthy", "rep": 1},
                     {"row": "r002", "block": 1, "arm": "glm", "case": "cpu_throttled", "rep": 1},
                     {"row": "r003", "block": 1, "arm": "luna", "case": "broker_stopped", "rep": 1}]

    def status(self):
        attempts = batch.load_attempts(self.runs)
        return {row["row"]: batch.row_status(attempts.get((self.arms[row["arm"]]["id"], row["row"]), []))
                for row in self.rows}

    def test_resume_requeue_and_model_invalid_rules(self):
        write_run(self.runs, "a", "healthy", row="r001", result=result())
        write_run(self.runs, "b", "cpu_throttled", arm="glm@z-ai-fp8", row="r002", status="invalid",
                  errors=["target did not recover"], result=result())
        write_run(self.runs, "c", "broker_stopped", row="r003", status="invalid",
                  errors=["agent conclusion could not be parsed"], result=result(root="PARSE_FAILURE"))
        self.assertEqual(self.status(), {"r001": "done", "r002": "requeue", "r003": "done"})
        write_run(self.runs, "d", "cpu_throttled", arm="glm@z-ai-fp8", row="r002", status="invalid",
                  errors=["interrupted"], created="2026-11-23T01:00:00+00:00")
        self.assertEqual(self.status()["r002"], "requeue")  # an interruption does not use up the re-run
        write_run(self.runs, "e", "cpu_throttled", arm="glm@z-ai-fp8", row="r002", status="invalid",
                  errors=["fault injection failed with exit 1"], created="2026-11-23T02:00:00+00:00")
        self.assertEqual(self.status()["r002"], "exhausted")

    def test_a_run_by_another_pin_does_not_count_for_this_arm(self):
        write_run(self.runs, "a", "healthy", arm="luna@other", row="r001", result=result())
        self.assertEqual(self.status()["r001"], "fresh")

    def test_fresh_rows_run_before_requeued_rows(self):
        write_run(self.runs, "a", "healthy", row="r001", status="invalid", errors=["stack"], result=result())
        order = [row["row"] for row in batch.pending_rows(self.rows, self.arms, batch.load_attempts(self.runs))]
        self.assertEqual(order, ["r002", "r003", "r001"])

    def fake_executor(self, outcomes):
        """Writes one run per call: (status, errors, result) from outcomes, in order."""
        calls = []

        def execute(row, arm, config, runs_dir, mode, prereg_sha):
            status, errors, run_result = outcomes[len(calls)]
            calls.append(row["row"])
            self.counter = getattr(self, "counter", 0) + 1  # unique across invocations
            created = f"2026-11-23T{self.counter:02d}:00:00+00:00"
            return write_run(runs_dir, f"inc-{self.counter}", row["case"], arm=arm["id"], row=row["row"],
                             status=status, errors=errors, result=run_result, created=created,
                             extra={"agent_started_at_utc": created})
        return execute, calls

    def run_batch(self, executor, rows=None, max_runs=None, other_spend_dirs=()):
        output = io.StringIO()
        ledger = free_tier.Ledger(self.runs / "ledger.json")
        with patch.object(batch, "STOP_FILE", self.runs / "absent.stop"), redirect_stdout(output), \
                patch("sys.stderr", output):
            code = batch.run_batch(self.config, rows or self.rows, self.runs, "matrix", max_runs, None, executor,
                                   ledger=ledger, other_spend_dirs=other_spend_dirs)
        return code, output.getvalue()

    def test_runs_every_pending_row_then_resumes_with_nothing_left(self):
        ok = ("completed", [], result())
        execute, calls = self.fake_executor([ok, ok, ok])
        self.assertEqual(self.run_batch(execute)[0], 0)
        self.assertEqual(calls, ["r001", "r002", "r003"])
        execute, calls = self.fake_executor([])
        code, output = self.run_batch(execute)
        self.assertEqual((code, calls), (0, []))
        self.assertIn("nothing left", output)

    def test_max_runs_and_stop_file(self):
        execute, calls = self.fake_executor([("completed", [], result())] * 3)
        self.assertEqual(self.run_batch(execute, max_runs=1)[0], 0)
        self.assertEqual(calls, ["r001"])
        stop = self.runs / "absent.stop"
        stop.write_text("", encoding="utf-8")
        code, output = self.run_batch(execute)
        self.assertEqual((code, calls), (0, ["r001"]))
        self.assertIn("stop file", output)

    def test_key_limit_stops_the_batch(self):
        crashed = result(status="crashed", error_type="APIStatusError",
                         traceback_tail="openai.APIStatusError: Error code: 402 - {'error': {'message': 'Key limit exceeded'}}")
        execute, calls = self.fake_executor([("invalid", ["agent result status is 'crashed', not completed"], crashed)])
        code, output = self.run_batch(execute)
        self.assertEqual((code, calls), (3, ["r001"]))
        self.assertIn("Error code: 402", output)
        self.assertEqual(self.status()["r001"], "fresh")  # the key-limit attempt is re-run after resuming

    def assert_key_stop(self, crashed, shown):
        for attempt_number in (1, 2):  # two key-limit stops in a row: the row is still never exhausted
            execute, calls = self.fake_executor([("invalid", CRASHED, crashed)])
            code, output = self.run_batch(execute)
            self.assertEqual((code, calls), (3, ["r001"]))
            self.assertIn(shown, output)
            self.assertEqual(self.status()["r001"], "fresh")
        attempt = batch.load_attempts(self.runs)[("luna@openai", "r001")][0]
        self.assertFalse(attempt.model_invalid or attempt.counts_for_requeue)
        self.assertEqual(aggregate.invalid_cause(attempt.manifest, attempt.result), "provider")
        self.assertNotIn("cost_estimated", attempt.manifest)  # refused before any provider: nothing spent
        self.assertEqual(batch.Spend(self.config, [self.runs]).arm("luna@openai"), 0.0)

    def test_key_limit_403_stops_the_batch_without_using_the_requeue(self):
        self.assert_key_stop(result(status="crashed", cost=None, llm_calls=0, error_type="OpenAIPermissionDeniedError",
                                    traceback_tail="langchain_openai.chat_models.base.OpenAIPermissionDeniedError: "
                                                   "Error code: 403 - {'error': {'message': 'Forbidden', 'code': 403}}"),
                             "refused the key (PermissionDeniedError)")

    def test_key_limit_message_stops_the_batch_case_insensitively(self):
        # An HTTP 200 whose body is an error (agent.graph.ProviderErrorResponse), naming the key limit.
        self.assert_key_stop(result(status="crashed", cost=None, llm_calls=0, error_type="ProviderErrorResponse",
                                    traceback_tail="agent.graph.ProviderErrorResponse: model endpoint returned an "
                                                   "error instead of a completion: {'message': 'KEY limit EXCEEDED'}"),
                             "KEY limit EXCEEDED")

    def test_rate_limit_is_not_a_key_failure(self):
        for text in ("openai.RateLimitError: Error code: 429 - {'error': {'message': "
                     "'Rate limit exceeded: free-models-per-day'}}",
                     "OpenAIRateLimitError: Error code: 429 - rate-limited upstream"):
            self.assertIsNone(batch.KEY_FAILURE.search(text))
        self.assertTrue(batch.KEY_FAILURE.search("HTTP 403"))
        self.assertTrue(batch.KEY_FAILURE.search("provider audit failed: generation lookup refused: HTTP 403"))

    def test_a_killed_run_is_marked_and_counted_at_the_reservation(self):
        glm_rows = [{"row": f"g{i}", "block": 1, "arm": "glm", "case": "healthy", "rep": i} for i in range(1, 4)]
        killed = ("invalid", ["Command '[...]' timed out after 1800 seconds"], None)  # no result, no trace
        execute, calls = self.fake_executor([killed, killed])
        code, output = self.run_batch(execute, rows=glm_rows, max_runs=2)
        self.assertEqual((code, calls), (0, ["g1", "g2"]))
        manifests = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.runs.glob("*/manifest.json"))]
        self.assertEqual([m.get("cost_estimated") for m in manifests], [True, True])
        self.assertAlmostEqual(manifests[0]["cost_estimated_usd"], GLM_WORST)
        self.assertIn("counted at the reservation", output)
        self.assertAlmostEqual(batch.Spend(self.config, [self.runs]).arm("glm@z-ai-fp8"), 2 * GLM_WORST)

    def test_pilot_spend_counts_against_the_cap(self):
        pilot = self.other_folder()
        write_run(pilot, "p1", "healthy", row="pilot-healthy-luna", result=result(cost=0.57))  # luna cap 0.60
        execute, calls = self.fake_executor([("completed", [], result())] * 3)
        code, output = self.run_batch(execute, other_spend_dirs=(pilot,))
        self.assertEqual((code, calls), (0, ["r002"]))  # luna: 0.57 + reservation 1.5 x 0.57 > 0.60
        self.assertIn("arm luna@openai stopped: spent $0.5700 (pre-pilot included)", output)

    def test_global_budget_refuses_a_run_that_could_pass_it(self):
        pilot = self.other_folder()
        # Spend under a switched GLM pin counts globally, not against glm@z-ai-fp8's cap.
        write_run(pilot, "p1", "healthy", arm="glm@novita-fp8", row="pilot-healthy-glm", result=result(cost=5.24))
        execute, calls = self.fake_executor([("completed", [], result())] * 3)
        code, output = self.run_batch(execute, other_spend_dirs=(pilot,))
        self.assertEqual((code, calls), (0, []))
        self.assertIn("recorded spend $5.2400 (all arms, pre-pilot included) + reservation $0.0216 > global budget "
                      "$5.25", output)
        self.config["spend_guard"]["global_budget_usd"] = 6.0  # configurable in arms.json
        code, output = self.run_batch(execute, other_spend_dirs=(pilot,))
        self.assertEqual(calls, ["r001", "r002", "r003"])

    def other_folder(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return Path(folder.name)

    def test_arm_cap_stops_only_that_arm(self):
        luna_rows = [{"row": f"r1{i}", "block": 1, "arm": "luna", "case": "healthy", "rep": i} for i in range(3)]
        expensive = ("completed", [], result(cost=0.59))  # cap 0.60: 0.59 + reservation (1.5 x 0.59) > 0.60
        execute, calls = self.fake_executor([expensive, expensive])
        code, output = self.run_batch(execute, rows=luna_rows + self.rows[1:2])
        self.assertEqual(calls, ["r10", "r002"])
        self.assertIn("arm luna@openai stopped", output)
        self.assertEqual(code, 0)

    def test_unpriced_pin_is_refused(self):
        glm = next(arm for arm in self.config["arms"] if arm["key"] == "glm")
        glm["pin"] = "wafer-fp4"
        code, output = self.run_batch(self.fake_executor([])[0], rows=self.rows[1:2])
        self.assertEqual(code, 2)
        self.assertIn("no price", output)

    def test_plan_needs_no_docker(self):
        with patch.object(runner, "command", side_effect=AssertionError("docker called")):
            text = batch.plan(self.config, preregister.read_schedule(), self.runs / "none",
                              ledger=free_tier.Ledger(self.runs / "ledger.json"), pilot_dir=self.runs / "pilot",
                              other_spend_dirs=())
        self.assertIn("Schedule: 152 rows (P2 120, P3 32), 16 blocks", text)
        for arm in ("luna@openai", "glm@z-ai-fp8", "gemma@novita-bf16", "nemotron@free"):
            self.assertIn(arm, text)
        self.assertIn("Pending runs: 152 (~30.4 h machine time at ~12 min/run on one stack; paid arms 114 runs", text)
        self.assertIn("50 requests/UTC day, worst case 16 requests/run -> 3 runs/day; 38 pending runs -> ~13 UTC days",
                      text)
        self.assertIn("Pre-pilot P1 (--pilot): 8 runs", text)
        self.assertIn("Cost switch glm", text)
        self.assertIn("Global budget $5.25 (arms.json spend_guard): recorded spend so far $0.0000", text)
        self.assertIn(f"luna@openai            reservation ${LUNA_WORST:.4f} (worst case ${LUNA_WORST:.4f}, "
                      "highest observed n/a)", text)
        # Pre-pilot spend shows in the budget, the arm's spend and its reservation.
        pilot = self.runs / "pilot"
        write_run(pilot, "p1", "healthy", row="pilot-healthy-luna", result=result(cost=0.1))
        text = batch.plan(self.config, preregister.read_schedule(), self.runs / "none",
                          ledger=free_tier.Ledger(self.runs / "ledger.json"), pilot_dir=pilot, other_spend_dirs=(pilot,))
        self.assertIn("recorded spend so far $0.1000 over all arms (pre-pilot $0.1000 included", text)
        self.assertIn("luna@openai            reservation $0.1500 (worst case $0.0216, highest observed $0.1000); "
                      "spent $0.1000 (pre-pilot $0.1000) of cap $0.60", text)

    def test_api_key_resolution_matches_the_agent(self):
        dotenv = self.runs / "test.env"
        dotenv.write_text("OPENROUTER_API_KEY=from-file\n", encoding="utf-8")
        with patch.object(runner, "DOTENV", dotenv), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(batch.resolved_api_key(), "from-file")
        with patch.object(runner, "DOTENV", dotenv), patch.dict("os.environ", {"MODEL_API_KEY": "m"}, clear=True):
            self.assertEqual(batch.resolved_api_key(), "m")
        with patch.object(runner, "DOTENV", self.runs / "absent.env"), patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(batch.resolved_api_key())


class SpendTests(TempDirTest):
    STARTED = {"agent_started_at_utc": "2026-11-23T00:01:00+00:00"}

    def setUp(self):
        super().setUp()
        self.config = preregister.load_config()

    def spend(self):
        return batch.Spend(self.config, [self.runs])

    def test_killed_or_crashed_runs_count_at_the_reservation(self):
        glm = "glm@z-ai-fp8"
        write_run(self.runs, "killed", "healthy", arm=glm, row="r1", status="invalid", extra=self.STARTED,
                  errors=["Command '[...]' timed out after 1800 seconds"])  # runner timeout: no result written
        write_run(self.runs, "crashed", "healthy", arm=glm, row="r2", status="invalid", errors=CRASHED, extra=self.STARTED,
                  result=result(status="crashed", cost=None, llm_calls=0, error_type="OpenAITimeoutError"))
        write_run(self.runs, "setup", "healthy", arm=glm, row="r3", status="invalid",
                  errors=["fault injection failed with exit 1"])  # the agent never started: nothing spent
        self.assertAlmostEqual(self.spend().arm(glm), 2 * GLM_WORST)
        self.assertEqual(self.spend().estimated_runs(), 2)

    def test_audited_cost_and_a_stored_estimate_use_the_larger_value(self):
        crashed = result(status="crashed", cost=None, llm_calls=0, error_type="OpenAITimeoutError")
        write_run(self.runs, "a", "healthy", arm="glm@z-ai-fp8", row="r1", status="invalid", errors=CRASHED,
                  result=crashed, extra=self.STARTED, audit={"retrieved": 3, "total_cost_usd": 0.9})
        attempt = batch.load_attempts(self.runs)[("glm@z-ai-fp8", "r1")][0]
        self.assertEqual(batch.attempt_spend(attempt, GLM_WORST), (0.9, True))
        attempt.manifest["generation_audit"]["total_cost_usd"] = 0.01
        self.assertEqual(batch.attempt_spend(attempt, GLM_WORST), (GLM_WORST, True))
        attempt.manifest["cost_estimated_usd"] = 0.3  # stored when the batch first saw the run
        self.assertEqual(batch.attempt_spend(attempt, GLM_WORST), (0.3, True))
        completed = batch.Attempt({"agent_started_at_utc": "x"}, result(cost=0.02), self.runs)
        self.assertEqual(batch.attempt_spend(completed, GLM_WORST), (0.02, False))

    def test_reservation_is_the_worst_case_until_a_completed_paid_run(self):
        self.assertAlmostEqual(self.spend().reservation("luna@openai"), LUNA_WORST)
        write_run(self.runs, "a", "healthy", row="r1", result=result(cost=0.01))  # 1.5 x 0.01 < worst case
        self.assertAlmostEqual(self.spend().reservation("luna@openai"), LUNA_WORST)
        write_run(self.runs, "b", "healthy", row="r2", result=result(cost=0.05))
        self.assertAlmostEqual(self.spend().reservation("luna@openai"), 0.075)
        write_run(self.runs, "c", "healthy", row="r3", status="invalid", errors=["target did not recover"],
                  result=result(cost=0.5))  # not a completed run: not an observation
        self.assertAlmostEqual(self.spend().reservation("luna@openai"), 0.075)
        write_run(self.runs, "d", "healthy", arm="nemotron@free", row="r4", result=result(cost=0.0004))
        self.assertEqual(self.spend().reservation("nemotron@free"), 0.0)  # a free arm is not a paid run
        self.assertAlmostEqual(self.spend().total(), 0.01 + 0.05 + 0.5 + 0.0004)


class RuleTests(unittest.TestCase):
    def test_r1_follows_the_dependency_order_and_ignores_the_setup_job(self):
        self.assertEqual(baseline_rule.rule_r1(HEALTHY_STATE), "healthy")
        both = state_with("kafka", "exited").replace("go-enricher: state=running", "go-enricher: state=restarting")
        self.assertEqual(baseline_rule.rule_r1(both), "kafka:exited")
        self.assertEqual(baseline_rule.rule_r1(HEALTHY_STATE.replace("redis3: state=running, status=\"Up 6 minutes\"\n", "")),
                         "redis3:absent")

    def test_r0_flags_every_exited_or_paused_container(self):
        self.assertEqual(baseline_rule.rule_r0(HEALTHY_STATE), "redis-cluster-setup:exited")
        self.assertFalse(baseline_rule.is_correct("healthy", baseline_rule.rule_r0(HEALTHY_STATE)))
        paused = baseline_rule.rule_r0(state_with("go-enricher", "paused"))
        self.assertTrue(baseline_rule.is_correct("enricher_paused", paused))

    def test_running_target_faults_cannot_be_found_by_the_rules(self):
        for case in ("client_timeout_too_low", "cpu_throttled", "latency_injected"):
            self.assertFalse(baseline_rule.is_correct(case, baseline_rule.rule_r1(HEALTHY_STATE)))
        self.assertTrue(baseline_rule.is_correct("redis_node_stopped", "redis2:exited"))
        self.assertFalse(baseline_rule.is_correct("redis_node_stopped", "redis2:paused"))


class AutoLabelTests(unittest.TestCase):
    def label(self, case, root, summary="", confidence="best_guess"):
        return auto_label.auto_label(case, result(root=root, summary=summary, confidence=confidence))

    def test_fault_labels(self):
        self.assertEqual(self.label("redis_node_stopped", "redis2-stopped"), "correct")
        self.assertEqual(self.label("redis_node_stopped", "redis-node-down", "a redis node is down"), "wrong")
        self.assertEqual(self.label("broker_stopped", "go-enricher-crash", "kafka is down"), "wrong")
        self.assertEqual(self.label("enricher_paused", "go-enricher-paused", "go-enricher is frozen"), "correct")
        self.assertEqual(self.label("client_timeout_too_low", "go-server-redis-client-timeout"), "correct")
        self.assertEqual(self.label("client_timeout_too_low", "go-server-slow"), "wrong")
        self.assertEqual(self.label("cpu_throttled", "go-server-cpu-throttled"), "correct")
        self.assertEqual(self.label("latency_injected", "network-latency", "go server responses delayed"), "correct")
        self.assertEqual(self.label("latency_injected", "none", "", "inconclusive"), "abstain")
        self.assertEqual(self.label("cpu_throttled", "unclear-degradation", "", "inconclusive"), "abstain")

    def test_healthy_labels_and_unrated(self):
        self.assertEqual(self.label("healthy", "none", "redis-cluster-setup exited normally"), "correct")
        self.assertEqual(self.label("healthy", "pipeline-healthy"), "correct")
        self.assertEqual(self.label("healthy", "redis-cluster-setup-exited"), "wrong")
        self.assertEqual(self.label("healthy", "kafka-consumer-lag"), "wrong")
        self.assertEqual(auto_label.auto_label("healthy", result(root="PARSE_FAILURE")), "unrated")
        self.assertEqual(auto_label.auto_label("healthy", None), "unrated")

    def test_kappa(self):
        self.assertEqual(auto_label.cohen_kappa([("correct", "correct"), ("wrong", "wrong")]), 1.0)
        self.assertEqual(auto_label.cohen_kappa([("correct", "wrong"), ("wrong", "correct")]), -1.0)
        self.assertIsNone(auto_label.cohen_kappa([]))
        pairs = [("correct", "correct")] * 6 + [("wrong", "wrong")] * 2 + [("correct", "wrong")] * 2
        self.assertEqual(auto_label.cohen_kappa(pairs), 0.5455)


class AggregateTests(TempDirTest):
    @staticmethod
    def review(diagnosis, evidence="supported", validity="valid", reason="", detected=None, localized=None):
        detected = detected or ("yes" if diagnosis == "correct" or diagnosis == "wrong" else "no")
        localized = localized or ("yes" if diagnosis == "correct" else "no")
        return {"reviewer": "h", "run_validity": validity, "validity_reason": reason,
                "diagnosis_label": diagnosis, "evidence_label": evidence, "detected": detected,
                "localized": localized}

    def test_per_protocol_itt_rules_calibration_and_cost(self):
        write_run(self.runs, "f1", "redis_node_stopped", observation=state_with("redis2", "exited"),
                  result=result("redis2-stopped", confidence="confirmed", cost=0.02,
                                model_self_reported_confidence="confirmed"),
                  audit={"retrieved": 3, "total_cost_usd": 0.03})
        write_run(self.runs, "f2", "cpu_throttled", result=result("go-server-ok", confidence="confirmed", cost=0.02,
                                                                  model_self_reported_confidence="confirmed"))
        write_run(self.runs, "f3", "cpu_throttled", status="invalid",
                  errors=["agent conclusion could not be parsed"], result=result("PARSE_FAILURE", cost=0.01))
        write_run(self.runs, "f4", "broker_stopped", status="invalid", errors=["target did not recover"],
                  result=result(cost=0.01))
        write_run(self.runs, "h1", "healthy", result=result(cost=0.01))
        write_run(self.runs, "h2", "healthy", result=result("kafka-lag", confidence="best_guess", cost=0.01))
        labels = {"f1": self.review("correct"), "f2": self.review("wrong", "unsupported", localized="yes"),
                  "h1": self.review("correct", detected="no", localized="n/a"),
                  "h2": self.review("wrong", "unsupported", localized="n/a")}
        report = aggregate.aggregate(self.runs, labels)
        arm = report["arms"]["luna@openai"]
        self.assertEqual((arm["attempted"], arm["human_valid"]), (6, 4))
        self.assertEqual(arm["invalid_by_cause"], {"infrastructure": 1, "model": 1})
        hidden, visible = arm["hidden"], arm["visible"]
        self.assertEqual((hidden["strict_accuracy"]["k"], hidden["strict_accuracy"]["n"]), (0, 1))
        self.assertEqual((hidden["strict_accuracy_itt"]["k"], hidden["strict_accuracy_itt"]["n"]), (0, 2))
        self.assertEqual((hidden["localized"]["k"], visible["strict_accuracy"]["k"]), (1, 1))
        self.assertEqual((arm["healthy"]["healthy_false_alarm"]["k"], arm["healthy"]["healthy_false_alarm"]["n"]), (1, 2))
        r1 = visible["rules"]["r1"]
        self.assertEqual((r1["strict_accuracy"]["k"], r1["strict_accuracy"]["n"]), (1, 1))
        self.assertEqual(arm["healthy"]["rules"]["r1"]["healthy_false_alarm"]["k"], 0)
        self.assertEqual(arm["healthy"]["rules"]["r0"]["healthy_false_alarm"]["k"], 2)
        confidence = arm["all"]["confidence"]
        self.assertEqual((confidence["p_correct_given_confirmed_enforced"]["k"],
                          confidence["p_correct_given_confirmed_enforced"]["n"]), (1, 2))
        self.assertEqual(confidence["confirmed_but_wrong_after_gate"], 1)
        # Audited cost (0.03) is larger than the reported one, so it is used; valid spend 0.07 / 2 correct.
        self.assertAlmostEqual(arm["cost_usd_all_attempts"], 0.09)
        self.assertAlmostEqual(arm["all"]["cost_usd"]["per_correct_diagnosis"], 0.035)
        self.assertEqual(set(arm["per_fault"]), {"redis_node_stopped", "cpu_throttled", "healthy"})
        self.assertEqual((report["pooled_healthy_false_alarm"]["k"], report["pooled_healthy_false_alarm"]["n"]), (1, 2))

    def test_unlabelled_completed_runs_fail_unless_allowed(self):
        write_run(self.runs, "h1", "healthy", result=result())
        with self.assertRaisesRegex(ValueError, "lack human labels"):
            aggregate.aggregate(self.runs, {})
        report = aggregate.aggregate(self.runs, {}, allow_unlabelled=True)
        self.assertEqual(report["unlabelled_machine_valid"], ["h1"])
        self.assertEqual(report["arms"]["luna@openai"]["invalid_by_cause"], {"awaiting label": 1})

    def test_mcnemar_and_provider_cause(self):
        self.assertEqual(aggregate.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(aggregate.mcnemar_exact(0, 6), 0.0312)
        self.assertEqual(aggregate.invalid_cause({"validity_errors": ["provider drift: x"]}, None), "provider")
        self.assertEqual(aggregate.invalid_cause({"validity_errors": ["agent exited with code 1"]},
                                                 {"status": "crashed", "error_type": "RateLimitError"}), "provider")

    def test_provider_api_errors_are_not_infrastructure(self):
        manifest = {"validity_errors": CRASHED, "arm": {"id": "glm@z-ai-fp8"}}
        crashed = lambda error_type, tail="": {"status": "crashed", "error_type": error_type, "traceback_tail": tail}
        for error_type in ("OpenAIPermissionDeniedError", "OpenAITimeoutError", "OpenAIAPIError", "OpenAIRateLimitError",
                           "ProviderErrorResponse"):  # langchain-openai's wrapped classes; a 200 with an error body
            self.assertEqual(aggregate.invalid_cause(manifest, crashed(error_type)), "provider", error_type)
        self.assertEqual(aggregate.invalid_cause(manifest, crashed("SomethingNew", "x: Error code: 503 - busy")),
                         "provider")
        self.assertEqual(aggregate.invalid_cause(manifest, crashed("RuntimeError", "MCP server did not provide")),
                         "infrastructure")


if __name__ == "__main__":
    unittest.main()
