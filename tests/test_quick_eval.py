"""Quick-eval group mode with fakes: plan, reservations, shared injection, concurrent agents, stop rules, free arm."""

import io
import json
import subprocess
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent.prompts import prompt_sha256
from experiments import batch, free_tier, openrouter_audit, preregister, quick_eval, runner

GLM_WORST = (98000 * 1.40 + 23600 * 4.40) / 1e6
LUNA_WORST = (98000 * 0.10 + 23600 * 0.50) / 1e6
GEMMA_WORST = (98000 * 0.14 + 2100 * 0.40) / 1e6
GROUP_WORST = GLM_WORST + LUNA_WORST + GEMMA_WORST


def done(code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def result(status="completed", cost=0.01, **extra):
    return {"status": status, "root_cause_id": "none", "summary": "ok", "confidence": "inconclusive",
            "cost_usd_total": cost, "calls_used": 4, "wall_clock_s": 60.0, "llm_calls": 3,
            "tool_schema_sha256": "f" * 64, **extra}


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.runs = self.root / "quick"
        self.pilot = self.root / "pilot"
        self.matrix = self.root / "v2"
        self.folders = [self.runs, self.pilot, self.matrix]
        self.config = preregister.load_config()
        self.arms = quick_eval.quick_arms(self.config)
        patcher = patch.object(batch, "STOP_FILE", self.root / "batch.stop")
        patcher.start()
        self.addCleanup(patcher.stop)


def write_attempt(folder, arm, *, row, status="completed", errors=(), res=None, audit=None, extra=None):
    run_id = "inc-" + row.replace("@", "-")
    run_dir = Path(folder) / run_id
    run_dir.mkdir(parents=True)
    manifest = {"run_id": run_id, "schedule_row": row, "status": status, "validity_errors": list(errors),
                "arm": {key: arm.get(key) for key in batch.MANIFEST_ARM_KEYS}, "case": "healthy",
                "created_at_utc": "2026-11-24T00:00:00+00:00", "agent_started_at_utc": "x", **(extra or {})}
    if audit is not None:
        manifest["generation_audit"] = audit
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if res is not None:
        (run_dir / f"{run_id}.result.json").write_text(json.dumps(res), encoding="utf-8")
    return run_dir


class PlanTests(unittest.TestCase):
    def test_plan_is_seeded_smoke_first_and_marked_exploratory(self):
        plan = quick_eval.build_plan()
        self.assertEqual(plan, quick_eval.build_plan())
        self.assertEqual(plan["note"], "exploratory; not the pre-registered matrix")
        self.assertEqual((len(plan["groups"]), plan["runs"]), (9, 27))
        self.assertEqual(plan["groups"][0], {"group_id": "q01", "case": "healthy", "rep": 1, "smoke": True})
        cases = [g["case"] for g in plan["groups"]]
        self.assertEqual({c: cases.count(c) for c in set(cases)},
                         {"redis_node_stopped": 2, "broker_stopped": 2, "latency_injected": 2,
                          "latency_enricher": 2, "healthy": 1})
        self.assertNotEqual([g["case"] for g in quick_eval.build_plan(1)["groups"][1:]], cases[1:])

    def test_the_committed_plan_matches_the_code(self):
        self.assertEqual(quick_eval.load_plan(), quick_eval.build_plan())

    def test_arm_selection(self):
        config = preregister.load_config()
        self.assertEqual(list(quick_eval.quick_arms(config)), ["luna", "glm", "gemma"])
        self.assertIn("nemotron", quick_eval.quick_arms(config, ["luna", "nemotron"]))
        with self.assertRaises(SystemExit):
            quick_eval.quick_arms(config, ["nemotron"])      # a free arm alone is not a group
        with self.assertRaises(SystemExit):
            quick_eval.quick_arms(config, ["luna", "frontier"])   # not enabled


class ReservationTests(TempDirTest):
    def test_group_reserves_the_sum_of_the_arm_reservations(self):
        reserve, problems, money = quick_eval.group_reservation(self.config, self.arms, self.folders, self.runs)
        self.assertEqual(problems, [])
        self.assertAlmostEqual(money["group_reservation_usd"], GROUP_WORST)
        self.assertAlmostEqual(reserve["glm@z-ai-fp8"], GLM_WORST)

    def test_global_budget_includes_pilot_spend(self):
        luna = self.arms["luna"]
        glm = self.arms["glm"]
        write_attempt(self.pilot, glm, row="pilot-a", res=result(cost=0.2))   # highest observed 0.2 -> reserve 0.3
        # Fill the rest of the budget with matrix spend so that only the group reservation breaks it.
        filler = 5.25 - 0.2 - 0.1
        write_attempt(self.matrix, luna, row="m1", res=result(cost=filler))
        _, problems, money = quick_eval.group_reservation(self.config, self.arms, self.folders, self.runs)
        self.assertAlmostEqual(money["reservations_usd"]["glm@z-ai-fp8"], 0.3)
        self.assertTrue(any(p.startswith("global budget") for p in problems), problems)
        self.assertTrue(any(p.startswith("arm luna@openai") for p in problems), problems)   # luna cap 0.60

    def test_quick_total_and_per_group_caps(self):
        glm = self.arms["glm"]
        write_attempt(self.runs, glm, row="quick-old-glm", res=result(cost=2.9))
        _, problems, _ = quick_eval.group_reservation(self.config, self.arms, self.folders, self.runs)
        self.assertTrue(any(p.startswith("quick-eval cap") for p in problems), problems)
        # 1.5 x 2.9 reservation for GLM alone exceeds the $0.75 per-group cap.
        self.assertTrue(any(p.startswith("per-group cap") for p in problems), problems)

    def test_caps_are_the_approved_values(self):
        self.assertEqual((quick_eval.TOTAL_CAP_USD, quick_eval.GROUP_CAP_USD), (3.00, 0.75))


class FakeStack:
    """Docker/agent boundary for run_group; agents block on a barrier so they must run concurrently."""

    def __init__(self, folder, arms_count=3, *, live_at_end=True, agent=None):
        self.folder = Path(folder)
        self.calls, self.envs, self.writer_threads = [], [], set()
        self.live_at_end = live_at_end
        self.live_checks = 0
        self.barrier = threading.Barrier(arms_count, timeout=10)
        self.agent = agent
        self.lock = threading.Lock()

    def patches(self):
        real_write = runner.write_manifest

        def write_manifest(path, manifest):
            self.writer_threads.add(threading.current_thread().name)
            real_write(path, manifest)

        fakes = {
            "stack_health": lambda: (True, {}), "verify_stack_marker": lambda: (True, ""),
            "consume_marker": lambda run_id: self.calls.append(("consume_marker", run_id)) or {"container_ids": {}},
            "container_images": lambda: {}, "prometheus_health": lambda: {"prometheus": 1.0},
            "traffic_probe": lambda: {"grew": True},
            "topic_end_offsets": lambda: {"raw_transactions": 10, "enriched_transactions": 9},
            "redis_key_counts": lambda: {"redis1": 3},
            "script_action": lambda case, action, log_dir: self.calls.append(action) or done(),
            "fault_is_live": self.fault_is_live,
            "restore_and_wait": lambda case, log_dir: (self.calls.append("restore") or [], {"recovered": True}),
            "wait_for_growth": lambda topic, seconds: self.calls.append("processing") or {"grew": True},
            "command": self.command,
            "runbook_snapshot": lambda: {"states": {}, "red": {}, "config": {}},
            "write_manifest": write_manifest,
            "DOTENV": self.folder / "absent.env",
        }
        audit = lambda run_dir, run_id, base_url, key, order: (
            {"retrieved": 2, "generations": 2, "missing": [], "providers_served": list(order), "total_cost_usd": 0.01},
            [])
        return [patch.object(runner, name, fake) for name, fake in fakes.items()] + \
            [patch.object(openrouter_audit, "audit_run", audit), patch.object(quick_eval.time, "sleep", lambda s: None),
             patch.dict("os.environ", {"FAULT_ID": "9", "FAULT_NAME": "leak", "MODEL_API_KEY": "test-key"})]

    def fault_is_live(self, case):
        with self.lock:
            self.live_checks += 1
            first = self.live_checks == 1
        self.calls.append("live")
        return first or self.live_at_end, {}

    def command(self, args, *, timeout=60, env=None):
        if args[0] == "git":
            return done(stdout="abc\n" if args[1] == "rev-parse" else "")
        assert args[1:] == ["-m", "agent.run"], args
        self.envs.append(env)
        self.barrier.wait()   # all agents of the group are running at the same time
        return (self.agent or self.write_artifacts)(env)

    @staticmethod
    def write_artifacts(env):
        run_dir, run_id = Path(env["RESULTS_DIR"]), env["RUN_TAG"]
        header = {"event": "investigation_started", "tool_specs": [{"name": n} for n in sorted(runner.REQUIRED_TOOLS)],
                  "prompt_variant": env.get("AGENT_PROMPT_VARIANT") or "v1"}
        events = [header, {"event": "tool_call", "ts": "2026-11-24T00:00:00+00:00", "result": "x [TRUNCATED: 5 of 9]"},
                  {"event": "action_selected", "agent_visible_prompt": "p"}]
        (run_dir / f"{run_id}.trace.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
        (run_dir / f"{run_id}.result.json").write_text(json.dumps(result(cost=0.01)), encoding="utf-8")
        return done(stdout="ok")


class GroupTests(TempDirTest):
    def run_group(self, case="broker_stopped", prompt_variant="v1", parent_env=None, **options):
        fake = FakeStack(self.root, **options)
        group = {"group_id": "q07", "case": case, "rep": 1, "smoke": False}
        reservations = {arm["id"]: 0.05 for arm in self.arms.values()}
        with ExitStack() as stack:
            for item in fake.patches():
                stack.enter_context(item)
            stack.enter_context(patch.dict("os.environ", parent_env or {}))
            outcome = quick_eval.run_group(group, self.arms, self.config, self.runs, reservations, "sha",
                                           prompt_variant=prompt_variant)
        return outcome, fake

    def manifests(self, outcome):
        return [json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8")) for run in outcome.runs]

    def test_v1_group_never_passes_a_prompt_variant_to_the_agents(self):
        outcome, fake = self.run_group(parent_env={"AGENT_PROMPT_VARIANT": "v2"})   # a stray parent setting
        self.assertFalse([env for env in fake.envs if "AGENT_PROMPT_VARIANT" in env])
        for manifest in self.manifests(outcome):
            self.assertEqual((manifest["status"], manifest["prompt_variant"]), ("completed", "v1"))
            self.assertEqual(manifest["system_prompt_sha256"], prompt_sha256("v1"))

    def test_v2_group_sets_the_variant_for_every_agent_and_records_it(self):
        outcome, fake = self.run_group(case="latency_injected", prompt_variant="v2")
        self.assertEqual({env["AGENT_PROMPT_VARIANT"] for env in fake.envs}, {"v2"})
        for manifest in self.manifests(outcome):
            self.assertEqual(manifest["status"], "completed", manifest["validity_errors"])
            self.assertEqual((manifest["prompt_variant"], manifest["system_prompt_sha256"]), ("v2", prompt_sha256("v2")))

    def test_an_agent_that_ran_another_prompt_variant_is_invalid(self):
        def agent(env):
            return FakeStack.write_artifacts({**env, "AGENT_PROMPT_VARIANT": "v1"})

        outcome, _ = self.run_group(prompt_variant="v2", agent=agent)
        for manifest in self.manifests(outcome):
            self.assertEqual(manifest["status"], "invalid")
            self.assertIn("agent used prompt variant 'v1', expected 'v2'", manifest["validity_errors"])

    def test_one_injection_three_concurrent_agents_separate_folders(self):
        outcome, fake = self.run_group()
        self.assertEqual(fake.calls.count("inject"), 1)
        self.assertEqual(fake.calls.count("restore"), 1)
        self.assertEqual(fake.calls.count("processing"), 1)
        self.assertEqual([c for c in fake.calls if isinstance(c, tuple)], [("consume_marker", "q07")])
        dirs = {env["RESULTS_DIR"] for env in fake.envs}
        self.assertEqual(len(dirs), 3)
        self.assertEqual({env["RUN_TAG"] for env in fake.envs}, {run.run_id for run in outcome.runs})
        self.assertEqual({env["MODEL_ID"] for env in fake.envs},
                         {arm["model_id"] for arm in self.arms.values()})
        for env in fake.envs:
            self.assertFalse([name for name in env if name.startswith("FAULT_")])
        manifests = [json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8")) for run in outcome.runs]
        self.assertEqual({m["group_id"] for m in manifests}, {"q07"})
        self.assertEqual({m["group_size"] for m in manifests}, {3})
        self.assertEqual(len({m["run_id"] for m in manifests}), 3)
        self.assertEqual(len({m["inject_started_at_utc"] for m in manifests}), 1)   # one shared injection
        for manifest in manifests:
            self.assertEqual(manifest["status"], "completed", manifest["validity_errors"])
            self.assertTrue(manifest["fault_live_at_end"])
            self.assertIn("agent_started_at_utc", manifest)
            self.assertIn("agent_ended_at_utc", manifest)
            self.assertIn("generation_audit", manifest)
            self.assertEqual(manifest["schedule_row"], f"quick-q07-{manifest['arm']['key']}")
        # Spend.load_attempts sees them as ledger entries per arm.
        self.assertEqual(len(batch.Spend(self.config, [self.runs]).attempts), 3)

    def test_free_arm_runs_as_a_fourth_concurrent_agent_and_is_audited(self):
        self.arms = quick_eval.quick_arms(self.config, ("luna", "glm", "gemma", "nemotron"))
        outcome, fake = self.run_group(case="latency_injected", arms_count=4)
        self.assertEqual(len({env["RESULTS_DIR"] for env in fake.envs}), 4)
        free_env = next(env for env in fake.envs if env["MODEL_ID"].endswith(":free"))
        self.assertEqual(free_env["MODEL_REQUIRE_ZERO_COST"], "true")
        manifests = [json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8")) for run in outcome.runs]
        self.assertEqual({m["group_size"] for m in manifests}, {4})
        self.assertTrue(all("generation_audit" in m for m in manifests))
        self.assertEqual(fake.calls.count("inject"), 1)

    def test_only_the_parent_thread_writes_manifests(self):
        _, fake = self.run_group()
        self.assertEqual(fake.writer_threads, {threading.main_thread().name})

    def test_fault_gone_at_the_end_invalidates_runs_and_is_recorded(self):
        outcome, _ = self.run_group(live_at_end=False)
        for run in outcome.runs:
            manifest = json.loads((run.run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["fault_live_at_end"])
            self.assertEqual(manifest["status"], "invalid")
            self.assertIn("fault ended before the agent completed", manifest["validity_errors"])

    def test_a_killed_agent_is_invalid_and_others_still_complete(self):
        def agent(env):
            if env["MODEL_ID"] == "z-ai/glm-5.3":
                raise subprocess.TimeoutExpired(["agent"], 1800)
            return FakeStack.write_artifacts(env)

        fake = FakeStack(self.root, agent=agent)
        group = {"group_id": "q02", "case": "healthy", "rep": 1, "smoke": False}
        reservations = {arm["id"]: (GLM_WORST if arm["key"] == "glm" else 0.02) for arm in self.arms.values()}
        with ExitStack() as stack:
            for item in fake.patches():
                stack.enter_context(item)
            outcome = quick_eval.run_group(group, self.arms, self.config, self.runs, reservations, "sha")
        attempts, counted = quick_eval.settle(outcome, self.runs)
        status = {a.manifest["arm"]["key"]: a.manifest for a in attempts}
        self.assertEqual(status["glm"]["status"], "invalid")
        self.assertTrue(status["glm"]["cost_estimated"])
        self.assertAlmostEqual(status["glm"]["cost_estimated_usd"], GLM_WORST)
        self.assertEqual((status["luna"]["status"], status["gemma"]["status"]), ("completed", "completed"))
        self.assertAlmostEqual(counted, GLM_WORST + 0.02)   # two audited runs at $0.01 + GLM at its reservation


def fake_group_runner(specs, calls):
    """A group runner writing one attempt per active arm: specs[group_id][arm_key] -> (status, errors, result, audit)."""

    def runner_(group, arms, config, runs_dir, reservations, plan_sha):
        calls.append((group["group_id"], sorted(arms)))
        runs = []
        for key, arm in arms.items():
            status, errors, res, audit = specs.get(group["group_id"], {}).get(key) or (
                "completed", [], result(cost=0.01),
                {"retrieved": 1, "missing": [], "providers_served": arm["provider_order"], "total_cost_usd": 0.01})
            run_dir = write_attempt(runs_dir, arm, row=f"quick-{group['group_id']}-{key}", status=status,
                                    errors=errors, res=res, audit=audit,
                                    extra={"group_id": group["group_id"], "group_errors": []})
            runs.append(quick_eval.Run(key, arm, run_dir.name, run_dir, {}))
        return quick_eval.GroupOutcome(group, runs, [], reservations)

    return runner_


class LoopTests(TempDirTest):
    plan = {"groups": [{"group_id": "q01", "case": "healthy", "rep": 1, "smoke": True},
                       {"group_id": "q02", "case": "broker_stopped", "rep": 1, "smoke": False},
                       {"group_id": "q03", "case": "latency_injected", "rep": 1, "smoke": False}]}

    def run_quick(self, specs, arm_keys=quick_eval.ARM_KEYS, ledger=None, **options):
        calls, teardowns = [], []
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = quick_eval.run_quick(
                self.config, self.plan, self.runs, arm_keys=arm_keys, folders=self.folders,
                group_runner=fake_group_runner(specs, calls), prepare=lambda d, g: None,
                teardown=lambda d, g: teardowns.append(g),
                ledger=ledger or free_tier.Ledger(self.root / "free.json"), today=lambda: "2026-11-24", **options)
        return code, calls, teardowns, out.getvalue() + err.getvalue()

    def test_runs_every_group_then_resumes_with_nothing_left(self):
        code, calls, teardowns, _ = self.run_quick({})
        self.assertEqual((code, [c[0] for c in calls], teardowns), (0, ["q01", "q02", "q03"], ["q01", "q02", "q03"]))
        code, calls, _, output = self.run_quick({})
        self.assertEqual((code, calls), (0, []))

    def test_key_failure_stops_after_the_group(self):
        crashed = result(status="crashed", cost=None, llm_calls=0, error_type="OpenAIPermissionDeniedError",
                         traceback_tail="Error code: 403 - {'error': {'message': 'Key limit exceeded'}}")
        specs = {"q02": {"gemma": ("invalid", ["agent result status is 'crashed', not completed"], crashed, None)}}
        code, calls, teardowns, output = self.run_quick(specs)
        self.assertEqual((code, [c[0] for c in calls]), (3, ["q01", "q02"]))
        self.assertEqual(teardowns, ["q01", "q02"])   # the stack still comes down
        self.assertIn("key failure in gemma@novita-bf16", output)

    def test_provider_mismatch_stops(self):
        audit = {"retrieved": 1, "missing": [], "providers_served": ["DeepInfra"], "total_cost_usd": 0.01}
        specs = {"q02": {"glm": ("invalid", ["provider drift: generation g served by 'DeepInfra', pinned ['z-ai']"],
                                 result(), audit)}}
        code, calls, _, output = self.run_quick(specs)
        self.assertEqual((code, len(calls)), (4, 2))
        self.assertIn("provider mismatch in glm@z-ai-fp8", output)

    def test_smoke_group_cost_out_of_range_stops(self):
        specs = {"q01": {"luna": ("completed", [], result(cost=0.2),
                                  {"retrieved": 1, "missing": [], "providers_served": ["OpenAI"], "total_cost_usd": 0.2})}}
        code, calls, _, output = self.run_quick(specs)
        self.assertEqual((code, len(calls)), (4, 1))
        self.assertIn("smoke check: luna@openai: recorded cost 0.2", output)

    def test_two_invalid_groups_in_a_row_stop(self):
        bad = {key: ("invalid", ["agent exited with code 1"], result(status="crashed"), None) for key in quick_eval.ARM_KEYS}
        code, calls, _, output = self.run_quick({"q02": bad, "q03": bad})
        self.assertEqual((code, len(calls)), (4, 3))
        self.assertIn("two consecutive invalid groups", output)

    def test_group_over_the_cap_stops(self):
        audit = {"retrieved": 1, "missing": [], "providers_served": ["z-ai"], "total_cost_usd": 0.8}
        code, calls, _, output = self.run_quick({"q02": {"glm": ("completed", [], result(cost=0.8), audit)}})
        self.assertEqual((code, len(calls)), (4, 2))
        self.assertIn("group cost", output)

    def test_refused_group_runs_nothing(self):
        write_attempt(self.pilot, self.arms["luna"], row="pilot-x", res=result(cost=0.59))   # luna cap 0.60
        code, calls, _, output = self.run_quick({})
        self.assertEqual((code, calls), (2, []))
        self.assertIn("arm luna@openai", output)

    def test_free_arm_joins_while_its_budget_lasts_and_never_stops_the_batch(self):
        ledger = free_tier.Ledger(self.root / "free.json")
        keys = ("luna", "glm", "gemma", "nemotron")
        free_ok = ("completed", [], result(cost=0.0, llm_calls=4, served_models=["nvidia/nemotron-3-ultra-550b-a55b:free"]),
                   {"retrieved": 4, "missing": [], "providers_served": ["Nvidia"], "total_cost_usd": 0.0,
                    "models_served": ["nvidia/nemotron-3-ultra-550b-a55b:free"]})
        # 40 of 50 requests used today: one worst-case run (16) no longer fits -> nemotron sits out, paid arms run.
        ledger.record("nemotron@free", "2026-11-24", "earlier", 40)
        code, calls, _, output = self.run_quick({"q01": {"nemotron": free_ok}}, arm_keys=keys, ledger=ledger)
        self.assertEqual(code, 0)
        self.assertEqual([c[1] for c in calls], [["gemma", "glm", "luna"]] * 3)
        self.assertIn("sits out", output)

    def test_free_arm_rate_limit_pauses_it_and_the_paid_arms_continue(self):
        ledger = free_tier.Ledger(self.root / "free.json")
        keys = ("luna", "glm", "gemma", "nemotron")
        limited = ("invalid", ["agent result status is 'crashed', not completed"],
                   result(status="crashed", cost=None, llm_calls=0, error_type="OpenAIRateLimitError",
                          traceback_tail="Error code: 429 - rate-limited upstream"), None)
        code, calls, _, output = self.run_quick({"q01": {"nemotron": limited}}, arm_keys=keys, ledger=ledger)
        self.assertEqual(code, 0, output)
        self.assertEqual([c[1] for c in calls], [["gemma", "glm", "luna", "nemotron"], ["gemma", "glm", "luna"],
                                                 ["gemma", "glm", "luna"]])
        self.assertIsNone(ledger.halted("nemotron@free"))

    def test_free_arm_paid_reply_halts_it_and_the_batch_continues(self):
        ledger = free_tier.Ledger(self.root / "free.json")
        keys = ("luna", "glm", "gemma", "nemotron")
        paid_reply = ("completed", [], result(cost=0.003, served_models=["nvidia/nemotron-3-ultra-550b-a55b"]),
                      {"retrieved": 1, "missing": [], "providers_served": ["Nvidia"], "total_cost_usd": 0.003,
                       "models_served": ["nvidia/nemotron-3-ultra-550b-a55b"]})
        code, calls, _, output = self.run_quick({"q01": {"nemotron": paid_reply}}, arm_keys=keys, ledger=ledger)
        self.assertEqual(code, 0, output)
        self.assertIsNotNone(ledger.halted("nemotron@free"))
        self.assertEqual(len(calls), 3)

    def test_cases_filter_runs_only_those_groups(self):
        code, calls, _, _ = self.run_quick({}, cases={"latency_injected"})
        self.assertEqual((code, [c[0] for c in calls]), (0, ["q03"]))

    def test_lower_caps_refuse_and_stop(self):
        code, calls, _, output = self.run_quick({}, group_cap=0.20)   # the three-arm worst case is above $0.20
        self.assertEqual((code, calls), (2, []))
        self.assertIn("per-group cap", output)
        audit = {"retrieved": 1, "missing": [], "providers_served": ["OpenAI"], "total_cost_usd": 0.06}
        code, calls, _, output = self.run_quick({"q02": {"luna": ("completed", [], result(cost=0.06), audit)}},
                                                arm_keys=("luna",), total_cap=0.05)
        self.assertEqual((code, len(calls)), (4, 2))
        self.assertIn("quick-eval total", output)

    def test_pin_override_is_in_memory_and_makes_a_new_arm_id(self):
        pins = quick_eval.parse_pins("gemma=crusoe-bf16")
        config = quick_eval.apply_pins(self.config, pins)
        gemma = quick_eval.quick_arms(config, ("luna", "gemma"))["gemma"]
        self.assertEqual((gemma["id"], gemma["provider_order"], gemma["quantizations"]),
                         ("gemma@crusoe-bf16", ["crusoe"], ["bf16"]))
        self.assertEqual(quick_eval.quick_arms(self.config)["gemma"]["id"], "gemma@novita-bf16")   # untouched
        self.assertEqual(preregister.load_config(), self.config)
        for text in ("gemma=nowhere", "nosuch=x", "glm=wafer-fp4"):   # unknown pin, unknown arm, unpriced pin
            with self.assertRaises(SystemExit):
                quick_eval.apply_pins(self.config, quick_eval.parse_pins(text))
        with self.assertRaises(SystemExit):
            quick_eval.parse_pins("gemma")

    def test_spend_folders_always_count_the_v1_quick_runs_once(self):
        other = self.root / "quick-v2"
        self.assertEqual(quick_eval.spend_folders(other, self.pilot, self.matrix, self.runs),
                         [other, self.runs, self.pilot, self.matrix])
        self.assertEqual(quick_eval.spend_folders(self.runs, self.pilot, self.matrix, self.runs),
                         [self.runs, self.pilot, self.matrix])

    def test_a_runs_dir_holds_one_prompt_variant(self):
        self.assertEqual(quick_eval.variant_conflicts(self.runs, "v2"), [])
        write_attempt(self.runs, self.arms["luna"], row="old")            # no prompt_variant: a v1 run
        self.assertEqual(quick_eval.variant_conflicts(self.runs, "v2"), ["v1"])
        self.assertEqual(quick_eval.variant_conflicts(self.runs, "v1"), [])


class SummaryTests(TempDirTest):
    def test_summary_counts_truncation_markers_and_orders_rows(self):
        plan = {"groups": [{"group_id": "q01"}, {"group_id": "q02"}]}
        for group in ("q02", "q01"):
            for key in ("gemma", "luna"):
                run_dir = write_attempt(self.runs, self.arms[key], row=f"quick-{group}-{key}",
                                        res=result(summary="line one\nline two"),
                                        extra={"group_id": group, "fault_live_at_end": True},
                                        audit={"retrieved": 1, "providers_served": ["OpenAI"], "total_cost_usd": 0.02})
                events = [{"event": "tool_call", "result": "a [TRUNCATED: 1 of 3]"},
                          {"event": "tool_call", "result": "[3 of 9 series omitted; add label matchers]"},
                          {"event": "tool_call", "result": "plain"},
                          {"event": "tool_call", "tool_name": "get_metrics", "args": {
                              "promql": "histogram_quantile(0.99, rate(fraud_server_transaction_duration_seconds_bucket[1m]))",
                              "window": "30m"}, "result": "x"},
                          {"event": "tool_call", "tool_name": "get_metrics", "args": {"promql": "up", "window": "1h"},
                           "result": "x"}]
                (run_dir / f"{run_dir.name}.trace.jsonl").write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
        rows = quick_eval.summary_rows(self.runs, plan)
        self.assertEqual([(r["group_id"], r["arm"]) for r in rows],
                         [("q01", "luna"), ("q01", "gemma"), ("q02", "luna"), ("q02", "gemma")])
        self.assertEqual(rows[0]["observations_truncated_or_omitted"], 2)
        self.assertEqual(rows[0]["root_cause_summary"], "line one line two")
        self.assertEqual(rows[0]["cost_usd"], 0.02)
        self.assertEqual((rows[0]["latency_metric_queries"], rows[0]["prompt_variant"]), (1, "v1"))
        self.assertTrue(quick_eval.write_summary(self.runs, plan).exists())


if __name__ == "__main__":
    unittest.main()
