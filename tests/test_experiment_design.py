"""Experiment design with every boundary faked: free arm, per-case settings, gates, rules, labels, metrics."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from experiments import (aggregate, baseline_rule, batch, free_tier, label_sheet, observability_probe, preregister,
                         runbook_rule, runner, score, symptoms, validate_faults)
from tests.test_benchmark import HEALTHY_STATE, result
from tests.test_experiment_runner import MODEL_ENV, FakeStack, done

FREE_ID = "nemotron@free"
FREE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def write(runs, run_id, case, *, arm, row=None, status="completed", errors=(), run_result=None, extra=None,
          observation=HEALTHY_STATE, created="2026-11-23T00:00:00+00:00", stderr=None):
    folder = Path(runs) / run_id
    folder.mkdir(parents=True)
    manifest = {"run_id": run_id, "case": case, "status": status,
                "validity_errors": list(errors), "arm": arm, "model_id": arm.get("model_id", "m"),
                "created_at_utc": created, "agent_tool_schema_sha256": "a" * 64, **(extra or {})}
    if row:
        manifest["schedule_row"] = row
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    events = [{"event": "investigation_started", "model": arm.get("model_id"), "env_label": arm.get("id")},
              {"event": "tool_call", "node": "baseline", "tool_name": "get_container_state", "result": observation}]
    (folder / f"{run_id}.trace.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    if run_result is not None:
        (folder / f"{run_id}.result.json").write_text(json.dumps(run_result), encoding="utf-8")
    if stderr:
        (folder / "agent.stderr.log").write_text(stderr, encoding="utf-8")
    return folder


def free_result(cost=0.0, attempts=5, served=(FREE_MODEL,), **extra):
    return result(cost=cost, llm_attempts=attempts, served_models=list(served), **extra)


# A 429 on the first call: no reply, so no cost to verify (llm_calls 0).
RATE_LIMITED = result(status="crashed", cost=None, error_type="RateLimitError", llm_calls=0, llm_attempts=0,
                      traceback_tail="openai.RateLimitError: Error code: 429 - {'error': {'message': "
                                     "'Rate limit exceeded: free-models-per-day'}}")
CRASH_ERRORS = ["agent result status is 'crashed', not completed"]


class TempDir(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.runs = Path(self.folder.name)


# --- free arm ---------------------------------------------------------------

class FreeArmBatchTests(TempDir):
    def setUp(self):
        super().setUp()
        self.config = preregister.load_config()
        self.arms = preregister.enabled_arms(self.config)
        self.ledger = free_tier.Ledger(self.runs / "state" / "free_tier.json")
        self.day = "2026-11-23"
        self.free_rows = [{"row": f"f{i}", "phase": "P2", "block": 9, "arm": "nemotron", "case": "healthy", "rep": i}
                          for i in range(1, 5)]
        self.paid_rows = [{"row": f"p{i}", "phase": "P2", "block": 1, "arm": "luna", "case": "healthy", "rep": i}
                          for i in range(1, 4)]

    def executor(self, outcomes):
        """outcomes: list of (status, errors, result) used in call order (free and paid alike)."""
        calls = []

        def execute(row, arm, config, runs_dir, mode, prereg_sha):
            status, errors, run_result, *extra = outcomes[len(calls)]  # optional 4th: extra manifest fields
            calls.append(row["row"])
            self.counter = getattr(self, "counter", 0) + 1  # unique across invocations
            return write(runs_dir, f"inc-{self.counter}", row["case"], row=row["row"], status=status,
                         errors=errors, run_result=run_result, created=f"2026-11-23T{self.counter:02d}:00:00+00:00",
                         extra=extra[0] if extra else None,
                         arm={key: arm.get(key) for key in batch.MANIFEST_ARM_KEYS})
        return execute, calls

    def run_free(self, outcomes, rows, today=None, config=None):
        execute, calls = self.executor(outcomes)
        output = io.StringIO()
        with patch.object(batch, "STOP_FILE", self.runs / "absent.stop"), redirect_stdout(output), \
                patch("sys.stderr", output):
            code = batch.run_batch(config or self.config, rows, self.runs / "runs", "matrix", None, None, execute,
                                   ledger=self.ledger, today=today or (lambda: self.day), other_spend_dirs=())
        return code, calls, output.getvalue()

    def status(self, row, arm_id=FREE_ID):
        return batch.row_status(batch.load_attempts(self.runs / "runs").get((arm_id, row), []))

    def test_free_arm_is_configured_to_cost_nothing(self):
        self.assertEqual(free_tier.config_problems(self.arms["nemotron"]), [])
        paid = dict(self.arms["nemotron"], model_id="nvidia/nemotron-3-ultra-550b-a55b")
        self.assertIn("is not a ':free' variant", " ".join(free_tier.config_problems(paid)))
        self.assertTrue(free_tier.config_problems(dict(self.arms["nemotron"], cap_usd=0.01)))
        self.assertTrue(free_tier.config_problems(dict(self.arms["nemotron"], allow_fallbacks=True)))
        self.assertTrue(free_tier.config_problems(dict(self.arms["nemotron"], max_price=None)))

    def test_misconfigured_free_arm_is_refused_and_paid_arms_still_run(self):
        config = json.loads(json.dumps(self.config))
        next(arm for arm in config["arms"] if arm["key"] == "nemotron")["model_id"] = "nvidia/paid-model"
        ok = ("completed", [], result())
        code, calls, output = self.run_free([ok, ok, ok], self.free_rows[:1] + self.paid_rows, config=config)
        self.assertEqual((code, calls), (0, ["p1", "p2", "p3"]))
        self.assertIn("refused, it could cost money", output)

    def test_free_and_paid_rows_alternate_while_budget_lasts(self):
        ok_free, ok_paid = ("completed", [], free_result()), ("completed", [], result())
        code, calls, _ = self.run_free([ok_free, ok_paid, ok_free, ok_paid, ok_free, ok_paid],
                                  self.paid_rows + self.free_rows[:3])
        self.assertEqual(calls, ["f1", "p1", "f2", "p2", "f3", "p3"])
        self.assertEqual(self.ledger.used(FREE_ID, self.day), 15)  # measured llm_attempts replace the reservation

    def test_cost_above_zero_halts_the_free_arm_for_good_and_paid_arms_continue(self):
        charged = ("completed", [], free_result(cost=0.0004))
        ok = ("completed", [], result())
        code, calls, output = self.run_free([charged, ok, ok, ok], self.free_rows + self.paid_rows)
        self.assertEqual((code, calls), (0, ["f1", "p1", "p2", "p3"]))
        self.assertIn("HALTED permanently", output)
        halt = free_tier.read_halt(self.runs / "runs")
        self.assertEqual(halt["arm"], FREE_ID)
        self.assertIn("Holm family becomes those arms", halt["consequence"])
        self.assertIsNotNone(self.ledger.halted(FREE_ID))
        # A later invocation does not run it either.
        code, calls, output = self.run_free([], self.free_rows[1:])
        self.assertEqual(calls, [])
        self.assertIn("halted", output)

    def test_cost_guard_served_model_and_unverifiable_cost_halt(self):
        accepted = [FREE_MODEL]
        guard = result(status="crashed", cost=None, error_type="ZeroCostViolation")
        self.assertIn("zero-cost guard", free_tier.cost_violation({}, guard, accepted))
        self.assertIn("not the accepted free variant",
                      free_tier.cost_violation({}, free_result(served=("nvidia/nemotron-paid",)), accepted))
        self.assertIn("audit reports cost", free_tier.cost_violation(
            {"generation_audit": {"retrieved": 2, "total_cost_usd": 0.01, "models_served": [FREE_MODEL]}},
            free_result(), accepted))
        self.assertIn("cannot be verified", free_tier.cost_violation({}, free_result(cost=None), accepted))
        self.assertIsNone(free_tier.cost_violation({"generation_audit": {"retrieved": 1, "total_cost_usd": 0.0,
                                                                         "models_served": [FREE_MODEL]}},
                                                   free_result(), accepted))

    def test_base_and_dated_served_models_are_accepted_only_with_a_verified_zero(self):
        accepted = self.arms["nemotron"]["free_tier"]["accepted_served_models"]
        self.assertEqual(accepted, [FREE_MODEL])
        base = "nvidia/nemotron-3-ultra-550b-a55b"
        guarded = {"require_zero_cost": True, "max_retries": 1}

        def run(served, cost=0.0, reported=3, settings=guarded, audit_cost=0.0, retrieved=3):
            audit = {"generations": 3, "retrieved": retrieved, "missing": ["g"] * (3 - retrieved),
                     "total_cost_usd": audit_cost, "models_served": [served]}
            return free_tier.cost_violation({"generation_audit": audit},
                                            free_result(cost=cost, served=(served,), cost_reported_calls=reported,
                                                        model_settings=settings), accepted)

        for served in (FREE_MODEL, base, f"{base}-20260604", f"{base}-20260604:free"):
            self.assertIsNone(run(served), served)
        self.assertIn("not every call reported a cost of exactly 0", run(base, cost=None))
        self.assertIn("not every call reported a cost of exactly 0", run(f"{base}-20260604", reported=2))
        self.assertIn("zero-cost guard was not on", run(base, settings={"require_zero_cost": False}))
        self.assertIn("audit did not confirm $0", run(f"{base}-20260604", retrieved=2))
        self.assertIn("replies reported cost", run(base, cost=0.0001))
        self.assertIn("audit reports cost", run(base, audit_cost=0.0001))
        for served in (f"{base}-2026064", f"{base}-20260604-paid", "nvidia/nemotron-3-super:free", f"{base}:extended"):
            self.assertIn("not the accepted free variant", run(served), served)
        # Through the batch: a dated slug with a verified $0 does not halt the arm.
        audit = {"generation_audit": {"generations": 3, "retrieved": 3, "missing": [], "total_cost_usd": 0.0,
                                      "models_served": [f"{base}-20260604"]}}
        ok = ("completed", [], free_result(served=(f"{base}-20260604",), cost_reported_calls=3, model_settings=guarded),
              audit)
        code, calls, output = self.run_free([ok], self.free_rows[:1])
        self.assertEqual((code, calls), (0, ["f1"]))
        self.assertNotIn("HALTED", output)
        self.assertIsNone(self.ledger.halted(FREE_ID))
        # The same slug without the audit's confirmation halts it for good.
        unconfirmed = ("completed", [], free_result(served=(f"{base}-20260604",), cost_reported_calls=3,
                                                     model_settings=guarded))
        code, calls, output = self.run_free([unconfirmed], self.free_rows[1:2])
        self.assertIn("HALTED permanently", output)
        self.assertIn("accepted only with a verified $0", self.ledger.halted(FREE_ID)["reason"])

    def test_daily_budget_starts_a_run_only_if_a_worst_case_run_fits(self):
        self.ledger.record(FREE_ID, self.day, "earlier", 40)  # 10 left of 50, a run may need 16
        ok = ("completed", [], result())
        code, calls, output = self.run_free([ok], self.free_rows[:1] + self.paid_rows[:1])
        self.assertEqual(calls, ["p1"])
        self.assertIn("waiting: 10 of 50 requests left on UTC day 2026-11-23", output)
        self.assertEqual(self.status("f1"), "fresh")
        code, calls, _ = self.run_free([("completed", [], free_result(attempts=16))] * 3, self.free_rows,
                                  today=lambda: "2026-11-24")
        self.assertEqual(calls, ["f1", "f2", "f3"])  # 3 x 16 = 48 of 50; a fourth run does not fit
        self.assertEqual(self.ledger.used(FREE_ID, "2026-11-24"), 48)

    def test_crashed_or_missing_results_count_conservatively(self):
        self.assertEqual(free_tier.requests_used(None, 16), 16)
        self.assertEqual(free_tier.requests_used(RATE_LIMITED, 16), 0 + 1 + 2)
        self.assertEqual(free_tier.requests_used(free_result(attempts=7), 16), 7)
        self.assertEqual(free_tier.estimate_days(38, 50, 16), (3, 13))
        self.assertEqual(free_tier.estimate_days(38, 1000, 16), (62, 1))

    def test_rate_limit_pauses_as_availability_failure_and_never_exhausts_the_row(self):
        limited = ("invalid", CRASH_ERRORS, RATE_LIMITED)
        ok = ("completed", [], result())
        code, calls, output = self.run_free([limited, ok, ok, ok], self.free_rows[:1] + self.paid_rows)
        self.assertEqual(calls, ["f1", "p1", "p2", "p3"])  # paid arms are not stopped
        self.assertIn("availability failure, not a model failure", output)
        self.assertEqual(self.ledger.remaining(FREE_ID, self.day, 50), 0)  # per-day limit: the day is used up
        attempts = batch.load_attempts(self.runs / "runs")[(FREE_ID, "f1")]
        self.assertEqual(attempts[0].availability_failure, "daily_limit")
        self.assertIsNone(attempts[0].key_failure)  # a 429 pauses the free arm; it never stops the batch
        self.assertFalse(attempts[0].counts_for_requeue or attempts[0].model_invalid)
        self.assertEqual(self.status("f1"), "fresh")
        self.assertEqual(aggregate.invalid_cause(attempts[0].manifest, attempts[0].result), "availability")
        # Two more days of 429s: still never exhausted; the third distinct day halts the arm as unavailable.
        for day in ("2026-11-24", "2026-11-25"):
            self.run_free([limited], self.free_rows[:1], today=lambda day=day: day)
            self.assertEqual(self.status("f1"), "fresh")
        self.assertIn("availability failures on 3 distinct UTC days", self.ledger.halted(FREE_ID)["reason"])

    def test_completed_run_resets_the_availability_day_count(self):
        transient = result(status="crashed", cost=None, error_type="InternalServerError", llm_calls=0, llm_attempts=1,
                           traceback_tail="openai.InternalServerError: Error code: 503 - overloaded")
        self.run_free([("invalid", CRASH_ERRORS, transient)], self.free_rows[:1])
        self.assertEqual(self.ledger.arm(FREE_ID)["availability_days"], [self.day])
        self.assertEqual(self.ledger.remaining(FREE_ID, self.day, 50), 50 - 4)  # 1 attempt + the failed call x 3; not exhausted
        self.run_free([("completed", [], free_result())], self.free_rows[:1], today=lambda: "2026-11-24")
        self.assertEqual(self.ledger.arm(FREE_ID)["availability_days"], [])

    def test_model_gone_halts(self):
        gone = result(status="crashed", cost=None, error_type="NotFoundError", llm_calls=0, llm_attempts=0,
                      traceback_tail="openai.NotFoundError: Error code: 404 - No endpoints found for model")
        code, calls, output = self.run_free([("invalid", CRASH_ERRORS, gone)], self.free_rows[:2])
        self.assertEqual(calls, ["f1"])
        self.assertIn("unavailable (permanent)", self.ledger.halted(FREE_ID)["reason"])

    def test_paid_rate_limit_is_not_an_availability_failure(self):
        manifest = {"status": "invalid", "validity_errors": CRASH_ERRORS, "arm": {"id": "luna@openai"}}
        self.assertIsNone(free_tier.availability_failure(manifest, RATE_LIMITED))
        self.assertEqual(aggregate.invalid_cause(manifest, RATE_LIMITED), "provider")


class CostSwitchTests(TempDir):
    def pilot(self, *costs, arm="glm@z-ai-fp8"):
        start = len(list(self.runs.glob("*/manifest.json")))
        for index, cost in enumerate(costs, start=start):
            write(self.runs, f"g{index}", "healthy", arm={"id": arm}, row=f"pilot-healthy-glm-{index}",
                  run_result=result(cost=cost))

    def status(self, config=None):
        return batch.cost_switch_status(config or preregister.load_config(), self.runs)[0]

    def test_glm_switch_rule(self):
        self.assertEqual(self.status()["status"], "pending: no pre-pilot cost yet")
        self.pilot(0.05, 0.08)
        self.assertEqual(self.status()["status"], "not triggered")
        self.pilot(0.2, 0.3, 0.25)  # median of 5 runs: 0.2 > 0.109
        self.assertIn("UNDECIDED", self.status()["status"])
        config = preregister.load_config()
        glm = next(arm for arm in config["arms"] if arm["key"] == "glm")
        glm["cost_switch"]["decision"] = "accept"
        self.assertIn("accepted: glm may stop early at its $3.40 cap", self.status(config)["status"])
        glm["pin"] = "novita-fp8"
        self.assertIn("switched to glm@novita-fp8", self.status(config)["status"])
        arm = preregister.enabled_arms(config)["glm"]
        self.assertEqual((arm["id"], arm["price_usd_per_million"]), ("glm@novita-fp8", {"input": 0.63, "output": 1.98}))

    def test_main_phase_refuses_while_undecided(self):
        self.pilot(0.2, 0.3)
        statuses = batch.cost_switch_status(preregister.load_config(), self.runs)
        with patch.object(batch, "cost_switch_status", lambda config: statuses), \
                patch.dict("os.environ", {"MODEL_API_KEY": "k"}), \
                patch.object(runner, "command", lambda *a, **k: done()):
            problems = batch.preflight([], "matrix", False, {"protocol_sha256": preregister.protocol_hashes()},
                                       preregister.load_config())
        self.assertTrue(any("cost switch for glm" in problem for problem in problems))


# --- runner: per-case settings, latency_enricher, target down ---------------

class CaseSettingsTests(unittest.TestCase):
    def test_settings_file_and_overrides(self):
        self.assertEqual(runner.load_fault_settings()["cpu_throttled"], {"LOW_CPUS": "0.1"})
        default = runner.configured_case("cpu_throttled")
        self.assertEqual((default.script_env(), runner.expected_nano_cpus(default)), ({"LOW_CPUS": "0.1"}, 100_000_000))
        low = runner.configured_case("cpu_throttled", {"LOW_CPUS": "0.01"})
        self.assertEqual(runner.expected_nano_cpus(low), 10_000_000)
        with self.assertRaisesRegex(ValueError, "not one of the registered values"):
            runner.configured_case("cpu_throttled", {"LOW_CPUS": "0.5"})
        with self.assertRaisesRegex(ValueError, "not configurable"):
            runner.configured_case("latency_enricher", {"TARGET": "go-server"})
        self.assertEqual(runner.configured_case("latency_enricher").script_env(), {"TARGET": "go-enricher"})
        self.assertEqual(runner.configured_case("latency_injected").env, ())

    def test_script_action_passes_the_case_env_to_the_script(self):
        seen = {}

        def command(args, *, timeout=60, env=None):
            seen[args[-1]] = env
            return done()

        with tempfile.TemporaryDirectory() as folder, patch.object(runner, "command", command):
            runner.script_action(runner.configured_case("latency_enricher"), "inject", Path(folder))
            runner.script_action(runner.configured_case("cpu_throttled", {"LOW_CPUS": "0.01"}), "restore", Path(folder))
            runner.script_action(runner.configured_case("latency_injected"), "inject", Path(folder))
        self.assertEqual(seen["inject"], None)  # latency_injected: script defaults (TARGET=go-server)
        self.assertEqual(seen["restore"]["LOW_CPUS"], "0.01")

    def probe(self, case, *, nano_cpus="0", qdisc="", running=True):
        services = []

        def compose(*args, **kwargs):
            services.append(args[-1])
            return done(stdout="cid1\n")

        def command(args, *, timeout=60, env=None):
            if args[:2] == ["docker", "inspect"]:
                return done(stdout=nano_cpus)
            if args[:2] == ["docker", "run"]:
                return done(stdout=qdisc)
            raise AssertionError(args)

        state = {"running": running, "paused": False, "healthy": True, "count": 1}
        with patch.object(runner, "command", command), patch.object(runner, "compose", compose), \
                patch.object(runner, "container_state", lambda service: services.append(service) or state):
            return runner.fault_is_live(case), runner.fault_is_restored(case), services

    def test_cpu_live_probe_follows_the_configured_quota(self):
        low = runner.configured_case("cpu_throttled", {"LOW_CPUS": "0.01"})
        (live, detail), _, _ = self.probe(low, nano_cpus="10000000")
        self.assertEqual((live, detail["mechanism"]["nano_cpus"]), (True, 10_000_000))
        (live, _), _, _ = self.probe(runner.configured_case("cpu_throttled"), nano_cpus="10000000")
        self.assertFalse(live)  # the default 0.1 case does not accept a 0.01 quota

    def test_latency_enricher_live_and_restore_probes_target_the_enricher(self):
        case = runner.configured_case("latency_enricher")
        self.assertEqual((case.service, case.active_state, case.script), ("go-enricher", "netem_delay",
                                                                           "network/09_latency_injection.sh"))
        active = "qdisc netem 8001: root refcnt 2 limit 1000 delay 200ms\n"
        (live, _), (restored, _), services = self.probe(case, qdisc=active)
        self.assertEqual((live, restored), (True, False))
        self.assertEqual(set(services), {"go-enricher"})
        (live, _), (restored, _), _ = self.probe(case, qdisc="qdisc noqueue 0: root refcnt 2\n")
        self.assertEqual((live, restored), (False, True))
        self.assertEqual(runner.case_role("latency_enricher"), "hidden")
        self.assertIn("latency_enricher", validate_faults.HIDDEN)

    def test_target_that_exits_after_injection_is_invalid_with_a_clear_cause(self):
        class Crashed(FakeStack):
            def fault_is_live(self, case):
                return False, {"running": False, "paused": False, "healthy": True, "count": 1}

        with tempfile.TemporaryDirectory() as folder:
            fake = Crashed(folder)
            with ExitStack() as stack:
                for item in fake.patches():
                    stack.enter_context(item)
                stack.enter_context(patch.dict("os.environ", MODEL_ENV))
                stack.enter_context(patch.object(runner.time, "sleep", lambda s: None))
                run_dir = runner.run_case(runner.configured_case("client_timeout_too_low"), Path(folder))
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "invalid")
        self.assertTrue(manifest["validity_errors"][0].startswith(runner.TARGET_DOWN))
        self.assertIn("no restart policy", manifest["validity_errors"][0])
        self.assertEqual(aggregate.invalid_cause(manifest, None), "target_down")
        self.assertIn("restore", fake.calls)  # still restored
        self.assertEqual(manifest["case_env"], {})
        self.assertEqual(manifest["case_role"], "hidden")


# --- symptoms, gates, observability ------------------------------------------

def red(server_p50=0.01, server_rps=5.0, server_err=0.0, enricher_p50=0.02, enricher_rps=5.0, enricher_err=0.0,
        server_procs=8.0, enricher_procs=8.0, server_cpu=0.01, enricher_cpu=0.01):
    return {"server_p50_s": server_p50, "server_requests_per_s": server_rps, "server_errors_per_s": server_err,
            "enricher_p50_s": enricher_p50, "enricher_processed_per_s": enricher_rps,
            "enricher_errors_per_s": enricher_err, "server_gomaxprocs": server_procs,
            "enricher_gomaxprocs": enricher_procs, "server_cpu_per_s": server_cpu, "enricher_cpu_per_s": enricher_cpu}


class SymptomTests(unittest.TestCase):
    def test_criteria_and_manipulation_check(self):
        self.assertEqual(symptoms.symptoms_present(red(), red()), {})
        found = symptoms.symptoms_present(red(), red(server_p50=0.45, server_rps=2.0))
        self.assertEqual(list(found), ["go-server"])
        self.assertEqual(len(found["go-server"]), 2)
        self.assertIn("go-enricher", symptoms.symptoms_present(red(), red(enricher_rps=None)))  # no samples = 0
        self.assertIn("go-server", symptoms.symptoms_present(red(), red(server_err=0.1)))
        self.assertTrue(symptoms.manipulation_check("latency_injected", True, red(), red(server_p50=1.0))["passed"])
        self.assertFalse(symptoms.manipulation_check("cpu_throttled", True, red(), red())["passed"])
        self.assertTrue(symptoms.manipulation_check("healthy", False, red(), red())["passed"])
        failed = dict(red(), server_p50_s="query failed: refused")
        self.assertIsNone(symptoms.manipulation_check("healthy", False, red(), failed)["passed"])

    def test_snapshot_queries_prometheus_and_records_failures(self):
        def opener(url, timeout):
            if "gomaxprocs" in url:
                raise OSError("refused")
            value = "NaN" if "histogram_quantile" in url else "2.5"
            return io.BytesIO(json.dumps({"status": "success", "data": {"result": [{"value": [0, value]}]}}).encode())

        taken = symptoms.snapshot("http://localhost:1", opener)
        self.assertEqual((taken["server_requests_per_s"], taken["server_p50_s"]), (2.5, None))
        self.assertTrue(taken["server_gomaxprocs"].startswith("query failed"))


class ObservabilityProbeTests(unittest.TestCase):
    def test_probe_lists_and_checks(self):
        self.assertEqual(observability_probe.probes("healthy"), [])
        self.assertEqual(observability_probe.probes("latency_enricher")[0]["args"]["promql"],
                         observability_probe.ENRICHER_P50)
        slow = '(no labels) -> latest 1700000000:0.41, 1700000005:0.43; range=0.004..0.43'
        fast = '(no labels) -> latest 1700000000:0.004, 1700000005:0.005; range=0.004..0.006'
        self.assertTrue(observability_probe.evaluate("latency_injected", [slow])["matched"])
        self.assertFalse(observability_probe.evaluate("latency_injected", [fast])["matched"])
        long_reply = slow + "x" * 2000
        verdict = observability_probe.evaluate("latency_injected", [long_reply])
        self.assertFalse(verdict["matched"])
        self.assertIn("more than the agent's 2000", verdict["calls"][0]["detail"])
        config = "REDIS_DIAL_TIMEOUT_MS=1\nREDIS_READ_TIMEOUT_MS=1\nKAFKA_BROKER=kafka:29092"
        healthy_config = "REDIS_DIAL_TIMEOUT_MS=(unset, using library default)"
        self.assertTrue(observability_probe.evaluate("client_timeout_too_low", [config])["matched"])
        self.assertFalse(observability_probe.evaluate("client_timeout_too_low", [healthy_config])["matched"])
        cpu = '{instance="go-server:9100", job="go-server"} -> latest 1700000000:0.0095'
        self.assertTrue(observability_probe.evaluate("cpu_throttled", [cpu, fast], {"LOW_CPUS": "0.01"})["matched"])
        self.assertFalse(observability_probe.evaluate("cpu_throttled", [cpu, fast], {"LOW_CPUS": "0.1"})["matched"])
        state = 'kafka: state=running, status="Up"\ngo-enricher: state=paused, status="Up 6 minutes (Paused)"'
        self.assertTrue(observability_probe.evaluate("enricher_paused", [state])["matched"])
        self.assertFalse(observability_probe.evaluate("ingestion_paused", [state])["matched"])


class GateTests(unittest.TestCase):
    def gate(self, case="latency_injected", *, live=lambda n: True, during=True, before=False, sustained=None,
             restore=()):
        snapshots = iter([red(), red(), sustained or red(server_p50=0.5)])
        probes = iter([{"matched": before, "calls": []}, {"matched": during, "calls": [{"detail": "latest 0.5"}]}])
        clock, checks = [0.0], []

        def fault_is_live(case_):
            checks.append(1)
            return live(len(checks)), {"running": True}

        def sleep(seconds):
            clock[0] += seconds

        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch.object(validate_faults, "snapshot", lambda: next(snapshots)))
            stack.enter_context(patch.object(validate_faults, "script_action", lambda c, a, d: done()))
            stack.enter_context(patch.object(validate_faults, "fault_is_live", fault_is_live))
            stack.enter_context(patch.object(validate_faults, "restore_and_wait",
                                             lambda c, d: (list(restore), {"recovered": not restore})))
            stack.enter_context(patch.object(validate_faults, "wait_for_growth", lambda t, s: {"grew": True}))
            return validate_faults.gate(case, Path(folder), 360, probe=lambda name, o: next(probes), sleep=sleep,
                                        clock=lambda: clock[0])

    def test_all_gates_pass(self):
        entry = self.gate()
        self.assertEqual(entry["verdict"], "pass")
        self.assertEqual(len(entry["live_checks"]), 1 + 360 // 30)  # start plus every 30 s
        self.assertEqual(list(entry["gates"]["G2"]["detail"]), ["go-server"])

    def test_each_gate_can_fail(self):
        self.assertFalse(self.gate(live=lambda n: n < 5)["gates"]["G1"]["pass"])
        self.assertFalse(self.gate(sustained=red())["gates"]["G2"]["pass"])
        self.assertFalse(self.gate(during=False)["gates"]["G3"]["pass"])
        self.assertFalse(self.gate(during=None)["gates"]["G3"]["pass"])  # probe could not run
        self.assertFalse(self.gate(before=True)["gates"]["G4"]["pass"])
        failed = self.gate(restore=["target did not recover"])
        self.assertEqual((failed["gates"]["G5"]["pass"], failed["verdict"]), (False, "fail"))

    def test_phase0_decision_and_two_stack_summary(self):
        decide = validate_faults.phase0_decision
        self.assertEqual(decide({"latency_injected": True, "latency_enricher": True})["decision"], "GO")
        self.assertEqual(decide({"latency_injected": True, "cpu_throttled": True})["decision"], "NO-GO")
        self.assertEqual(decide({"latency_injected": True, "cpu_throttled": True,
                                 "client_timeout_too_low": True})["decision"], "GO")
        with tempfile.TemporaryDirectory() as folder:
            files = []
            for index, enricher in enumerate(("pass", "fail")):
                path = Path(folder) / f"g{index}.json"
                path.write_text(json.dumps({"cases": [{"case": "latency_injected", "verdict": "pass"},
                                                      {"case": "latency_enricher", "verdict": enricher}]}))
                files.append(path)
            summary = validate_faults.summarize(files)
        self.assertEqual(summary["case_passed_on_every_stack"], {"latency_enricher": False, "latency_injected": True})
        self.assertEqual(summary["decision"], "NO-GO")

    def test_bad_setting_is_rejected(self):
        self.assertEqual(validate_faults.parse_sets(["cpu_throttled.LOW_CPUS=0.01"]),
                         {"cpu_throttled": {"LOW_CPUS": "0.01"}})
        with self.assertRaises(SystemExit):
            validate_faults.parse_sets(["LOW_CPUS=0.01"])


# --- rules R2 and R3 -----------------------------------------------------------

def snap(red_values, config=None):
    return {"states": {s: "running" for s in runner.REQUIRED_SERVICES}, "red": red_values,
            "config": config or {"go-server": {"REDIS_READ_TIMEOUT_MS": ""}, "go-enricher": {"KAFKA_BROKER": "k"}}}


class RuleR2R3Tests(unittest.TestCase):
    def test_r2_recency(self):
        recent = HEALTHY_STATE.replace('go-server: state=running, status="Up 6 minutes"',
                                       'go-server: state=running, status="Up 25 seconds"')
        self.assertEqual(baseline_rule.rule_r2(recent), "go-server:recently_restarted")
        self.assertEqual(baseline_rule.rule_r2(HEALTHY_STATE), "healthy")
        about = HEALTHY_STATE.replace('status="Up 6 minutes"', 'status="Up About a minute"')
        self.assertEqual(baseline_rule.rule_r2(about), "healthy")  # every service is young: no single outlier
        stopped = HEALTHY_STATE.replace("kafka: state=running", "kafka: state=exited")
        self.assertEqual(baseline_rule.rule_r2(stopped), "kafka:exited")  # R1 first
        self.assertTrue(baseline_rule.localizes("client_timeout_too_low", "go-server:recently_restarted"))
        self.assertFalse(baseline_rule.is_correct("client_timeout_too_low", "go-server:recently_restarted"))

    def test_r3_decision_list(self):
        thresholds = dict(runbook_rule.PROVISIONAL)
        decide = lambda before, after: runbook_rule.decide(snap(before), after, thresholds)
        low = snap(red(), {"go-server": {"REDIS_READ_TIMEOUT_MS": "1"}, "go-enricher": {"KAFKA_BROKER": "k"}})
        self.assertEqual(runbook_rule.decide(snap(red()), low, thresholds), "go-server:config:REDIS_READ_TIMEOUT_MS")
        self.assertEqual(decide(red(), snap(red(server_p50=0.5))), "go-server:network_delay")
        self.assertEqual(decide(red(), snap(red(server_p50=0.5, server_procs=2.0))), "go-server:cpu")
        self.assertEqual(decide(red(), snap(red(server_procs=2.0))), "abstain")  # CPU signal alone names nothing
        self.assertEqual(decide(red(), snap(red(enricher_p50=0.9))), "go-enricher:network_delay")
        self.assertEqual(decide(red(), snap(red(server_err=1.0))), "go-server:errors")
        self.assertEqual(decide(red(), snap(red())), "abstain")
        self.assertIsNone(runbook_rule.decide(None, snap(red()), thresholds))
        cpu_limited = dict(thresholds, cpu_rate={"go-server": 0.02, "go-enricher": None})
        self.assertEqual(runbook_rule.decide(snap(red()), snap(red(server_p50=0.5, server_cpu=0.05)), cpu_limited),
                         "go-server:cpu")

    def test_r3_never_sees_the_case(self):
        import inspect
        self.assertEqual(list(inspect.signature(runbook_rule.decide).parameters), ["before", "after", "thresholds"])

    def test_r3_strict_and_localization_scoring(self):
        strict, local = baseline_rule.r3_correct, baseline_rule.localizes
        self.assertTrue(strict("latency_injected", "go-server:network_delay"))
        self.assertFalse(strict("latency_injected", "go-server:cpu"))
        self.assertTrue(local("latency_injected", "go-server:cpu"))
        self.assertTrue(strict("cpu_throttled", "go-server:cpu"))
        self.assertTrue(strict("latency_enricher", "go-enricher:network_delay"))
        self.assertFalse(local("latency_enricher", "go-server:network_delay"))
        self.assertTrue(strict("client_timeout_too_low", "go-server:config:REDIS_DIAL_TIMEOUT_MS+REDIS_READ_TIMEOUT_MS"))
        self.assertFalse(strict("client_timeout_too_low", "go-server:config:KAFKA_BROKER"))
        self.assertTrue(strict("healthy", "abstain") and local("healthy", "abstain"))
        self.assertFalse(strict("healthy", "go-server:errors"))
        self.assertFalse(strict("latency_injected", "abstain") or local("latency_injected", "abstain"))
        self.assertFalse(strict("redis_node_stopped", "redis2:exited"))  # R3 has no container-state branch

    def test_r3_applied_from_the_manifest_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            run = write(folder, "r", "latency_injected", arm={"id": "a"},
                        extra={"runbook_snapshot_before": snap(red()), "runbook_snapshot": snap(red(server_p50=0.4))})
            row = baseline_rule.apply_rules(run, json.loads((run / "manifest.json").read_text()),
                                            dict(runbook_rule.PROVISIONAL))
        self.assertEqual((row["r3_prediction"], row["r3_correct"], row["r3_localized"]),
                         ("go-server:network_delay", True, True))
        self.assertEqual((row["r1_correct"], row["r1_localized"], row["r2_prediction"]), (False, False, "healthy"))

    def test_thresholds_are_derived_from_healthy_runs_only(self):
        with tempfile.TemporaryDirectory() as folder:
            write(folder, "h1", "healthy", arm={"id": "a"},
                  extra={"runbook_snapshot_before": snap(red()), "runbook_snapshot": snap(red(server_p50=0.012))})
            write(folder, "h2", "healthy", arm={"id": "a"},
                  extra={"runbook_snapshot_before": snap(red()),
                         "runbook_snapshot": snap(red(server_err=0.05, server_cpu=0.02))})
            write(folder, "f1", "latency_injected", arm={"id": "a"},
                  extra={"runbook_snapshot_before": snap(red()), "runbook_snapshot": snap(red(server_p50=9.0))})
            pairs = runbook_rule.healthy_pairs([Path(folder)], [])
        self.assertEqual(len(pairs), 2)  # the fault run is never used
        derived = runbook_rule.derive(pairs)
        self.assertEqual(derived["status"], "derived")
        self.assertAlmostEqual(derived["latency_ratio"], 1.5 * 1.2)  # largest healthy ratio: server 0.012/0.01
        self.assertAlmostEqual(derived["error_ratio"], round(1.5 * 0.01, 6))
        self.assertAlmostEqual(derived["cpu_rate"]["go-server"], 0.03)

    def test_committed_thresholds_are_provisional_until_derived(self):
        self.assertEqual(runbook_rule.load_thresholds()["status"], "provisional")


# --- labels -----------------------------------------------------------------

class LabelTests(TempDir):
    def test_level_labels_are_validated(self):
        base = {"reviewer": "h", "run_validity": "valid", "diagnosis_label": "correct", "evidence_label": "supported"}
        check = score.check_levels
        check("a", "latency_injected", {**base, "detected": "yes", "localized": "yes"}, True)
        check("b", "healthy", {**base, "detected": "no", "localized": "n/a"}, True)
        check("c", "healthy", base, False)  # optional unless required
        for case, review, message in (
                ("latency_injected", {**base, "detected": "", "localized": "yes"}, "needs detected"),
                ("latency_injected", {**base, "detected": "yes", "localized": "maybe"}, "needs localized"),
                ("healthy", {**base, "detected": "no", "localized": "yes"}, "n/a or blank"),
                ("latency_injected", {**base, "detected": "no", "localized": "yes"}, "needs detected=yes"),
                ("latency_injected", {**base, "detected": "yes", "localized": "no"}, "must be localized")):
            with self.assertRaisesRegex(ValueError, message):
                check("x", case, review, True)

    def test_blind_sheet_hides_the_model_and_merges_back(self):
        arm = {"id": "luna@openai", "key": "luna", "model_id": "openai/gpt-6-luna"}
        run_result = result("go-server-latency", "gpt-6-luna thinks go-server is slow", confidence="best_guess",
                            served_models=["openai/gpt-6-luna"], input_tokens_total=999)
        write(self.runs, "inc-a", "latency_injected", arm=arm, run_result=run_result)
        write(self.runs, "inc-b", "healthy", arm={"id": "gemma@novita-bf16", "model_id": "google/gemma-4-31b-it"},
              run_result=result())
        write(self.runs, "inc-c", "healthy", arm=arm, status="invalid", errors=["x"], run_result=result())
        out = self.runs / "labelling"
        self.assertEqual(label_sheet.export(self.runs, out, preregister.RUBRIC, seed=1), 2)
        text = "".join(path.read_text(encoding="utf-8") for path in (out / "packets").glob("*.json"))
        text += (out / "label_sheet.csv").read_text(encoding="utf-8")
        for hidden in ("luna", "gpt-6", "gemma", "served_model", "cost_usd", "env_label", "input_tokens"):
            self.assertNotIn(hidden, text.lower())
        self.assertIn("[model] thinks go-server is slow", text)
        self.assertIn("Names go-server AND network delay", text)  # blind to the model, not to the case
        sheet = list(csv.DictReader((out / "label_sheet.csv").open(encoding="utf-8")))
        self.assertEqual({row["case"] for row in sheet}, {"latency_injected", "healthy"})
        key = {row["blind_id"]: row["run_id"] for row in csv.DictReader((out / "label_key.csv").open(encoding="utf-8"))}
        for row in sheet:
            row.update({"reviewer": "h", "run_validity": "valid", "diagnosis_label": "correct",
                        "evidence_label": "supported", "detected": "yes" if row["case"] != "healthy" else "no",
                        "localized": "yes" if row["case"] != "healthy" else "n/a"})
        filled = self.runs / "filled.csv"
        label_sheet.write_csv(filled, label_sheet.SHEET_FIELDS, sheet)
        labels_path = self.runs / "human_labels.csv"
        self.assertEqual(label_sheet.merge(filled, out / "label_key.csv", labels_path), 2)
        labels = score.load_labels(labels_path)
        self.assertEqual(set(labels), set(key.values()))
        self.assertEqual(labels["inc-a"]["localized"], "yes")
        with self.assertRaisesRegex(ValueError, "already labelled"):
            label_sheet.merge(filled, out / "label_key.csv", labels_path)


# --- aggregate metrics -------------------------------------------------------

class MetricTests(TempDir):
    def test_statistics(self):
        self.assertEqual(aggregate.fisher_exact(3, 0, 0, 3), 0.1)
        self.assertEqual(aggregate.fisher_exact(0, 0, 0, 0), 1.0)
        self.assertEqual(aggregate.holm({"a": 0.01, "b": 0.04, "c": 0.03}), {"a": 0.03, "c": 0.06, "b": 0.06})
        self.assertAlmostEqual(aggregate.pass_hat_k(4, 6, 2), 0.4)
        self.assertEqual(aggregate.pass_hat_k(6, 6, 4), 1.0)
        self.assertIsNone(aggregate.pass_hat_k(1, 2, 4))
        self.assertEqual(aggregate.mcnemar_exact(8, 0), 0.0078)  # 8 pairs: below .05/4 = .0125
        self.assertEqual(aggregate.mcnemar_exact(7, 0), 0.0156)  # 7 pairs: below .05/3 only

    @staticmethod
    def review(diagnosis, case):
        healthy = case == "healthy"
        return {"reviewer": "h", "run_validity": "valid", "validity_reason": "", "diagnosis_label": diagnosis,
                "evidence_label": "supported",
                "detected": "no" if (healthy and diagnosis == "correct") or diagnosis == "abstain" else "yes",
                "localized": "n/a" if healthy else ("yes" if diagnosis == "correct" else "no")}

    def test_report_blocks_comparisons_sensitivity_family_and_availability(self):
        labels, arms = {}, {"luna@openai": {"id": "luna@openai"}, FREE_ID: {"id": FREE_ID, "free": True}}
        plan = {"latency_injected": ["correct"] * 4 + ["wrong"] * 2, "latency_enricher": ["wrong"] * 3 + ["abstain"],
                "cpu_throttled": ["correct", "wrong"], "client_timeout_too_low": ["correct"] * 2,
                "healthy": ["correct", "wrong"], "broker_stopped": ["correct"]}
        for arm_id, arm in arms.items():
            for case, outcomes in plan.items():
                for index, outcome in enumerate(outcomes):
                    run_id = f"{arm_id[:4]}-{case}-{index}"
                    self_conf = "confirmed" if index < 2 else "best_guess"
                    enforced = "best_guess" if case == "latency_injected" and index == 1 else self_conf
                    write(self.runs, run_id, case, arm=arm,
                          run_result=result("x", confidence=enforced, model_self_reported_confidence=self_conf,
                                            cost=0.01),
                          extra={"manipulation_check": {"passed": not (case == "cpu_throttled" and index == 1)}})
                    labels[run_id] = self.review(outcome, case)
        write(self.runs, "nemo-limited", "healthy", arm=arms[FREE_ID], status="invalid", errors=CRASH_ERRORS,
              run_result=RATE_LIMITED)
        report = aggregate.aggregate(self.runs, labels, thresholds=dict(runbook_rule.PROVISIONAL))
        luna = report["arms"]["luna@openai"]
        hidden = luna["hidden"]
        self.assertEqual((hidden["strict_accuracy"]["k"], hidden["strict_accuracy"]["n"]), (7, 14))
        self.assertEqual((hidden["selective_accuracy"]["k"], hidden["selective_accuracy"]["n"]), (7, 13))
        self.assertEqual(hidden["coverage"]["k"], 13)
        self.assertEqual(luna["per_fault"]["latency_injected"]["strict_accuracy"]["k"], 4)
        self.assertIn("rules", luna["per_fault"]["latency_enricher"])  # every metric per fault
        self.assertAlmostEqual(hidden["pass_k"]["pass^2"],
                               round((6 / 15 + 0 + 0 + 1) / 4, 4))  # cells: 4/6, 0/4, 1/2, 2/2
        confidence = luna["per_fault"]["latency_injected"]["confidence"]
        self.assertEqual((confidence["downgraded"], confidence["correct_confirmations_downgraded"],
                          confidence["wrong_confirmations_caught"]), (1, 1, 0))
        localization = luna["registered_comparisons"]["localization"]
        self.assertEqual(localization["strict_accuracy"]["latency_injected"], "4/6")
        self.assertEqual(localization["strict_accuracy"]["latency_enricher"], "0/4")
        self.assertIn("fisher_exact_p", luna["registered_comparisons"]["mechanism"]["strict_accuracy"])
        sensitivity = luna["sensitivity"]
        self.assertEqual(sensitivity["hidden_without_client_timeout_too_low"]["strict_accuracy"]["n"], 12)
        self.assertEqual(sensitivity["hidden_without_failed_manipulation_check"]["strict_accuracy"]["n"], 13)
        self.assertIn("holm_p", hidden["rules"]["r1"]["paired_strict"])
        free = report["arms"][FREE_ID]
        self.assertEqual((free["availability_failures"], free["invalid_by_cause"]), (1, {"availability": 1}))
        self.assertEqual(report["headline_family"]["arms"], ["luna@openai", FREE_ID])
        self.assertEqual(report["pooled_healthy_false_alarm"]["n"], 4)
        self.assertIsNotNone(report["auto_label_agreement"]["compared"])
        self.assertEqual(report["r3_thresholds_status"], "provisional")
        # The free arm halts: the family is the paid arms only, and the free arm stays in the report.
        free_tier.write_halt(self.runs, {"arm": FREE_ID, "reason": "cost"})
        report = aggregate.aggregate(self.runs, labels, thresholds=dict(runbook_rule.PROVISIONAL))
        self.assertEqual(report["headline_family"]["arms"], ["luna@openai"])
        self.assertEqual(report["pooled_healthy_false_alarm"]["n"], 2)
        self.assertIsNone(report["arms"][FREE_ID]["hidden"]["rules"]["r1"]["paired_strict"]["holm_p"])
        self.assertIn(FREE_ID, report["arms"])


class AutoLabelNewCaseTests(unittest.TestCase):
    def test_latency_enricher_needs_the_enricher_and_delay(self):
        from experiments.auto_label import auto_label
        label = lambda root, summary="": auto_label("latency_enricher", result(root, summary, "best_guess"))
        self.assertEqual(label("go-enricher-network-latency"), "correct")
        self.assertEqual(label("go-server-network-latency"), "wrong")
        self.assertEqual(label("go-enricher-slow", "go-enricher is slow"), "wrong")


class PreregistrationDesignTests(unittest.TestCase):
    def test_freeze_covers_new_protocol_and_analysis_files(self):
        prereg = preregister.freeze()
        for path in ("experiments/symptoms.py", "experiments/fault_settings.json"):
            self.assertIn(path, prereg["protocol_sha256"])
        for path in ("experiments/prereg/analysis_plan.json", "experiments/prereg/r3_thresholds.json",
                     "experiments/runbook_rule.py", "experiments/free_tier.py"):
            self.assertIn(path, prereg["analysis_sha256"])
        self.assertEqual(set(prereg["rules"]["rubric"]), set(runner.CASES))
        self.assertIn("4 enabled arms", prereg["rules"]["headline_family"])
        plan = json.loads((runner.ROOT / "experiments" / "prereg" / "analysis_plan.json").read_text(encoding="utf-8"))
        self.assertIn("client_timeout_too_low", plan["sensitivity"][0])
        self.assertEqual(set(plan["registered_comparisons"]), {"localization", "mechanism"})

    def test_pilot_is_every_arm_on_healthy_and_latency(self):
        config = preregister.load_config()
        rows = batch.pilot_schedule(config, preregister.enabled_arms(config), None)
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["case"] for row in rows}, {"healthy", "latency_injected"})
        self.assertEqual({row["phase"] for row in rows}, {"P1"})


if __name__ == "__main__":
    unittest.main()
