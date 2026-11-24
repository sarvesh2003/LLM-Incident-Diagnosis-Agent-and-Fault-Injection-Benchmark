"""Exploratory quick evaluation (not the pre-registered matrix): one fault injection shared by concurrent agents.

Each group is one case on one fresh stack, investigated by one agent per arm; runs go to experiments/runs/quick.
Spend is reserved per group against arm, global, quick-eval and per-group caps; stop rules apply between groups.

Usage:
  python -m experiments.quick_eval --plan            # write/verify experiments/quick_eval_plan.json, no Docker
  python -m experiments.quick_eval --run [--max-groups 1] [--groups q01,q02] [--arms luna,glm,gemma,nemotron]
  python -m experiments.quick_eval --summary         # experiments/runs/quick/summary.csv
  python -m experiments.quick_eval --run --offline --groups q01   # fake model (experiments.fake_model), own runs dir
  python -m experiments.quick_eval --run --prompt-variant v2 --runs-dir experiments/runs/quick-v2 \
      --cases latency_injected,latency_enricher --arms luna,gemma,nemotron --pin gemma=crusoe-bf16 \
      --total-cap 0.50 --group-cap 0.20
"""

from __future__ import annotations

import argparse
import copy
import csv
import functools
import json
import os
import random
import re
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import AgentConfig
from agent.prompts import PROMPT_VARIANTS, prompt_sha256
from experiments import batch, free_tier, openrouter_audit, preregister, runner, symptoms
from experiments.aggregate import run_cost

ROOT = runner.ROOT
PLAN_FILE = ROOT / "experiments" / "quick_eval_plan.json"
QUICK_RUNS = ROOT / "experiments" / "runs" / "quick"
OFFLINE_RUNS = ROOT / "experiments" / "runs" / "offline-verification" / "quick"
SEED = 20261124
ARM_KEYS = ("luna", "glm", "gemma")
GROUP_CASES = (("redis_node_stopped", 2), ("broker_stopped", 2), ("latency_injected", 2),
               ("latency_enricher", 2), ("healthy", 1))
SMOKE_CASE = "healthy"
TOTAL_CAP_USD = 3.00
GROUP_CAP_USD = 0.75
SMOKE_MAX_RUN_USD = 0.15
AGENT_TIMEOUT_SECONDS = 1800    # same per-agent timeout as runner.run_case
NOTE = "exploratory; not the pre-registered matrix"
TRUNCATION_MARKERS = ("[TRUNCATED", "omitted")
LATENCY_QUERY = re.compile(r"duration|latency|bucket|quantile", re.IGNORECASE)


# --- plan -------------------------------------------------------------------

def build_plan(seed: int = SEED) -> dict:
    """Groups in execution order: the healthy smoke group first, the rest shuffled with `seed`."""
    rest = [(case, rep) for case, reps in GROUP_CASES for rep in range(1, reps + 1)]
    smoke = rest.pop(rest.index((SMOKE_CASE, 1)))
    random.Random(seed).shuffle(rest)
    groups = [{"group_id": f"q{n:02d}", "case": case, "rep": rep, "smoke": n == 1}
              for n, (case, rep) in enumerate([smoke, *rest], start=1)]
    return {"note": NOTE, "seed": seed, "arms": list(ARM_KEYS), "group_size": len(ARM_KEYS),
            "groups": groups, "runs": len(groups) * len(ARM_KEYS),
            "caps_usd": {"quick_total": TOTAL_CAP_USD, "per_group": GROUP_CAP_USD,
                         "smoke_max_per_run": SMOKE_MAX_RUN_USD},
            "agent_timeout_seconds": AGENT_TIMEOUT_SECONDS,
            "order_rule": "first group is the healthy smoke test; the other groups are shuffled with random.Random(seed)"}


def load_plan(path: Path = PLAN_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_plan(path: Path = PLAN_FILE) -> dict:
    """Write the plan, or verify an existing file matches it (never silently re-ordered)."""
    plan = build_plan()
    if path.exists():
        existing = load_plan(path)
        if existing != plan:
            raise SystemExit(f"{path} differs from the plan this code builds; refusing to overwrite it")
        return existing
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def quick_arms(config: dict, keys: tuple[str, ...] | list[str] = ARM_KEYS) -> dict[str, dict]:
    """The selected enabled arms; at least one paid arm, free arms optional."""
    arms = preregister.enabled_arms(config)
    missing = [key for key in keys if key not in arms]
    if missing:
        raise SystemExit(f"arms {missing} are not enabled in arms.json (enabled: {sorted(arms)})")
    chosen = {key: arms[key] for key in keys}
    if not any(not arm["free"] for arm in chosen.values()):
        raise SystemExit("quick eval needs at least one paid arm")
    return chosen


def apply_pins(config: dict, pins: dict[str, str]) -> dict:
    """A copy of the arms config with arm key -> pin overrides; each pin must exist in arms.json with a price."""
    config = copy.deepcopy(config)
    by_key = {arm["key"]: arm for arm in config["arms"]}
    for key, pin in pins.items():
        if key not in by_key:
            raise SystemExit(f"--pin: no arm {key!r} in arms.json")
        if pin not in by_key[key]["pins"]:
            raise SystemExit(f"--pin: arm {key!r} has no pin {pin!r} (defined: {sorted(by_key[key]['pins'])})")
        if not by_key[key]["pins"][pin].get("price_usd_per_million"):
            raise SystemExit(f"--pin: pin {key}={pin} has no price, so its spend cannot be reserved")
        by_key[key]["pin"] = pin
    return config


def parse_pins(text: str | None) -> dict[str, str]:
    pins = {}
    for item in (text or "").split(","):
        if item.strip():
            key, separator, pin = item.partition("=")
            if not separator or not key.strip() or not pin.strip():
                raise SystemExit(f"--pin expects arm=pin[,arm=pin], got {item!r}")
            pins[key.strip()] = pin.strip()
    return pins


def paid(arms: dict[str, dict]) -> dict[str, dict]:
    return {key: arm for key, arm in arms.items() if not arm["free"]}


# --- money ------------------------------------------------------------------

def spend_folders(runs_dir: Path, pilot_dir: Path = batch.PILOT_RUNS, matrix_dir: Path = runner.RUNS,
                  quick_dir: Path = QUICK_RUNS) -> list[Path]:
    """Folders whose spend counts: this runs dir, the v1 quick runs, the pre-pilot and the matrix (each once)."""
    unique: dict[Path, Path] = {}
    for folder in (runs_dir, quick_dir, pilot_dir, matrix_dir):
        unique.setdefault(Path(folder).resolve(), folder)
    return list(unique.values())


def variant_conflicts(runs_dir: Path, variant: str) -> list[str]:
    """Prompt variants other than `variant` recorded in runs_dir (a manifest without one is v1)."""
    found = set()
    for path in runs_dir.glob("*/manifest.json") if runs_dir.exists() else []:
        found.add(json.loads(path.read_text(encoding="utf-8")).get("prompt_variant") or "v1")
    return sorted(found - {variant})


def group_reservation(config: dict, arms: dict[str, dict], folders: list[Path], runs_dir: Path,
                      total_cap: float = TOTAL_CAP_USD, group_cap: float = GROUP_CAP_USD) -> tuple[dict[str, float], list[str], dict]:
    """(reservation per arm id, reasons the group may not start, ledger snapshot)."""
    # All arms' reservations are taken together before the concurrent runs start, since none can be stopped mid-run.
    spend = batch.Spend(config, folders)
    quick = batch.Spend(config, [runs_dir])
    reserve = {arm["id"]: spend.reservation(arm["id"]) for arm in arms.values()}
    needed = sum(reserve.values())
    total, quick_total = spend.total(), quick.total()
    problems = []
    for arm in arms.values():
        spent = spend.arm(arm["id"])
        if not arm["free"] and spent + reserve[arm["id"]] > arm["cap_usd"]:
            problems.append(f"arm {arm['id']}: spent ${spent:.4f} + reservation ${reserve[arm['id']]:.4f} "
                            f"> cap ${arm['cap_usd']:.2f}")
    if total + needed > spend.budget:
        problems.append(f"global budget: recorded ${total:.4f} + group reservation ${needed:.4f} > ${spend.budget:.2f}")
    if quick_total + needed > total_cap:
        problems.append(f"quick-eval cap: spent ${quick_total:.4f} + group reservation ${needed:.4f} > ${total_cap:.2f}")
    if needed > group_cap:
        problems.append(f"per-group cap: group reservation ${needed:.4f} > ${group_cap:.2f}")
    snapshot = {"reservations_usd": reserve, "group_reservation_usd": needed, "global_spent_usd": total,
                "global_budget_usd": spend.budget, "quick_spent_usd": quick_total,
                "arm_spent_usd": {arm["id"]: spend.arm(arm["id"]) for arm in arms.values()}}
    return reserve, problems, snapshot


# --- one group --------------------------------------------------------------

@dataclass
class Run:
    key: str
    arm: dict
    run_id: str
    run_dir: Path
    manifest: dict
    env: dict | None = None
    agent_config: AgentConfig | None = None
    errors: list[str] = field(default_factory=list)
    agent_ok: bool = False

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def save(self) -> None:
        runner.write_manifest(self.manifest_path, self.manifest)


@dataclass
class GroupOutcome:
    group: dict
    runs: list[Run]
    group_errors: list[str]
    reservations: dict[str, float]

    def attempts(self) -> list[batch.Attempt]:
        found = []
        for run in self.runs:
            path = run.run_dir / f"{run.run_id}.result.json"
            result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            found.append(batch.Attempt(json.loads(run.manifest_path.read_text(encoding="utf-8")), result, run.run_dir))
        return found


def launch_agent(env: dict) -> subprocess.CompletedProcess:
    """One agent process (with its own MCP server); runs in a worker thread and writes no records itself."""
    return runner.command([runner.agent_python(), "-m", "agent.run"], timeout=AGENT_TIMEOUT_SECONDS, env=env)


def new_runs(group: dict, arms: dict[str, dict], config: dict, runs_dir: Path, case: runner.Case,
             extra: dict, prompt_variant: str = "v1") -> list[Run]:
    runs = []
    for key, arm in arms.items():
        run_id = "inc-" + uuid.uuid4().hex
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        with batch.arm_environment(config, arm):
            manifest = {**runner.new_manifest(case, run_id), **extra,
                        "schedule_row": f"quick-{group['group_id']}-{key}",
                        "arm": {name: arm.get(name) for name in batch.MANIFEST_ARM_KEYS},
                        "prompt_variant": prompt_variant, "system_prompt_sha256": prompt_sha256(prompt_variant)}
            run = Run(key, arm, run_id, run_dir, manifest)
            try:
                run.env = runner.agent_environment(run_id, run_dir)
                run.env.pop("AGENT_PROMPT_VARIANT", None)   # the variant comes only from --prompt-variant
                if prompt_variant != "v1":
                    run.env["AGENT_PROMPT_VARIANT"] = prompt_variant
                run.agent_config = AgentConfig.from_env(run.env)
                manifest["model_id"] = run.agent_config.model_id
                manifest["model_settings"] = run.agent_config.model_settings.as_dict()
            except Exception as error:
                run.errors.append(str(error))
        run.save()
        runs.append(run)
    return runs


def finish_agent(run: Run, case: runner.Case, outcome) -> None:
    """Record one agent's end and check its artifacts; runs in the parent thread."""
    manifest = run.manifest
    manifest["agent_ended_at_utc"] = runner.utc_now()
    try:
        if isinstance(outcome, BaseException):
            raise outcome
        (run.run_dir / "agent.stdout.log").write_text(outcome.stdout, encoding="utf-8")
        (run.run_dir / "agent.stderr.log").write_text(outcome.stderr, encoding="utf-8")
        manifest["agent_exit_code"] = outcome.returncode
        if outcome.returncode != 0:
            raise RuntimeError(f"agent exited with code {outcome.returncode}")
        agent_result, trace = runner.check_agent_artifacts(run.run_dir, run.run_id)
        observations = [event["ts"] for event in trace if event.get("event") == "tool_call"]
        manifest["first_observation_at_utc"] = min(observations) if observations else None
        manifest["last_observation_at_utc"] = max(observations) if observations else None
        manifest["agent_tool_schema_sha256"] = agent_result.get("tool_schema_sha256")
        used = trace[0].get("prompt_variant") or "v1"
        if used != manifest["prompt_variant"]:
            raise RuntimeError(f"agent used prompt variant {used!r}, expected {manifest['prompt_variant']!r}")
        run.agent_ok = True
    except Exception as error:
        run.errors.append(str(error))
    # Checked as each agent ends: a fault that ended early means that agent may have seen a recovered stack.
    if case.script:
        try:
            live, state = runner.fault_is_live(case)
        except Exception as error:
            live, state = False, {"error": str(error)}
        manifest["live_probe_after"] = {"at_utc": runner.utc_now(), "live": live, "state": state}
        manifest["fault_live_at_end"] = live
        if not live and run.agent_ok:
            down = runner.live_failure(case, state).startswith(runner.TARGET_DOWN)
            run.errors.append(f"{runner.TARGET_DOWN}: {case.service} exited during the agent run" if down
                              else "fault ended before the agent completed")
    else:
        manifest["fault_live_at_end"] = None   # healthy control: nothing injected
    if run.agent_ok and not run.errors:
        manifest["status"] = "agent_completed"
    run.save()


def run_group(group: dict, arms: dict[str, dict], config: dict, runs_dir: Path, reservations: dict[str, float],
              plan_sha: str | None, prompt_variant: str = "v1") -> GroupOutcome:
    """One injection shared by one agent per arm; restore, recovery and processing checks run once for the group."""
    case = runner.configured_case(group["case"])
    extra = {"batch": "quick", "quick_eval_note": NOTE, "quick_plan_sha256": plan_sha,
             "schedule_phase": "quick", "schedule_block": group["group_id"], "schedule_rep": group["rep"],
             "group_id": group["group_id"], "group_size": len(arms), "group_case": group["case"],
             "group_smoke": bool(group.get("smoke")), "group_reservation_usd": dict(reservations)}
    runs = new_runs(group, arms, config, runs_dir, case, extra, prompt_variant)
    shared: dict = {}
    group_errors: list[str] = []
    injected = interrupted = agents_ran = False
    try:
        if any(run.errors for run in runs):
            raise RuntimeError("model settings invalid for " + ", ".join(r.key for r in runs if r.errors))
        runner.preflight(shared)
        shared["stack_baseline"] = runner.consume_marker(group["group_id"])
        runner.record_snapshot(shared, "runbook_snapshot_before")
        if case.script:
            injected = True  # restore even if injection fails part way
            shared["inject_started_at_utc"] = runner.utc_now()
            log_dir = runs_dir / f"{group['group_id']}-injection-{runs[0].run_id[-8:]}"
            log_dir.mkdir(parents=True, exist_ok=True)
            injected_result = runner.script_action(case, "inject", log_dir)
            shared["inject_ended_at_utc"] = runner.utc_now()
            shared["injection_log_dir"] = str(log_dir.relative_to(ROOT)) if log_dir.is_relative_to(ROOT) else str(log_dir)
            if injected_result.returncode != 0:
                raise RuntimeError(f"fault injection failed with exit {injected_result.returncode}")
            time.sleep(case.settle_seconds)
            live, state = runner.fault_is_live(case)
            shared["live_probe_before"] = {"at_utc": runner.utc_now(), "live": live, "state": state}
            if not live:
                raise RuntimeError(runner.live_failure(case, state))
        for run in runs:
            run.manifest.update(shared)
            run.manifest["status"] = "preflight_passed"
        # Concurrent investigation: workers only run the agent process; every record is written here.
        agents_ran = True
        with ThreadPoolExecutor(max_workers=len(runs)) as pool:
            futures = {}
            for run in runs:
                run.manifest["agent_started_at_utc"] = runner.utc_now()
                run.save()
                futures[pool.submit(launch_agent, run.env)] = run
            for future in as_completed(futures):
                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = error
                finish_agent(futures[future], case, outcome)
        if case.script:
            live, state = runner.fault_is_live(case)
            shared["group_live_probe_end"] = {"at_utc": runner.utc_now(), "live": live, "state": state}
        runner.record_snapshot(shared, "runbook_snapshot")
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        group_errors.append("interrupted" if interrupted else str(error))
    finally:
        if injected:
            shared["restore_started_at_utc"] = runner.utc_now()
            try:
                log_dir = Path(ROOT / shared["injection_log_dir"]) if "injection_log_dir" in shared else runs_dir
                errors, shared["recovery_probe"] = runner.restore_and_wait(case, log_dir)
                group_errors.extend(errors)
            except Exception as error:
                group_errors.append(f"restore exception: {error}")
        if "preflight_states" in shared:
            try:
                shared["processing_probe"] = runner.wait_for_growth("enriched_transactions", runner.RECOVERY_SECONDS)
                if not shared["processing_probe"]["grew"]:
                    group_errors.append("enriched output did not resume after the run")
            except Exception as error:
                group_errors.append(f"processing probe failed: {error}")
        try:
            shared["topic_end_offsets_after"] = runner.topic_end_offsets()
            shared["redis_key_counts_after"] = runner.redis_key_counts()
        except Exception as error:
            group_errors.append(f"post-run state capture failed: {error}")
        for run in runs:
            finalize_run(run, case, shared, group_errors, agents_ran and not interrupted)
    if interrupted:
        raise KeyboardInterrupt
    return GroupOutcome(group, runs, group_errors, reservations)


def finalize_run(run: Run, case: runner.Case, shared: dict, group_errors: list[str], audit: bool) -> None:
    manifest = run.manifest
    manifest.update(shared)
    before, after = shared.get("runbook_snapshot_before"), shared.get("runbook_snapshot")
    if manifest.get("status") == "agent_completed":
        manifest["manipulation_check"] = symptoms.manipulation_check(
            case.name, case.script is not None, (before or {}).get("red"), (after or {}).get("red"))
    config = run.agent_config
    pinned = config is not None and bool(config.model_settings.provider_order or config.model_settings.require_zero_cost)
    if audit and pinned and "agent_started_at_utc" in manifest:
        try:
            manifest["generation_audit"], errors = openrouter_audit.audit_run(
                run.run_dir, run.run_id, config.base_url, config.api_key, config.model_settings.provider_order)
            run.errors.extend(errors)
        except Exception as error:
            run.errors.append(f"provider audit failed: {error}")
    manifest["group_errors"] = list(group_errors)
    manifest["validity_errors"] = [*run.errors, *group_errors]
    ok = not manifest["validity_errors"] and manifest.get("status") == "agent_completed"
    manifest["status"] = "completed" if ok else "invalid"
    manifest["finished_at_utc"] = runner.utc_now()
    run.save()


# --- after a group: settle money, stop rules --------------------------------

def settle(outcome: GroupOutcome, runs_dir: Path) -> tuple[list[batch.Attempt], float]:
    """Mark runs whose cost may be missing (counted at the reservation); returns attempts and the group's cost."""
    attempts, counted_total = outcome.attempts(), 0.0
    for attempt in attempts:
        reserve = outcome.reservations[attempt.manifest["arm"]["id"]]
        if batch.needs_estimate(attempt):
            attempt.manifest.update({"cost_estimated": True, "cost_estimated_usd": reserve})
            runner.write_manifest(attempt.run_dir / "manifest.json", attempt.manifest)
        counted, estimated = batch.attempt_spend(attempt, reserve)
        counted_total += counted
        batch.log(runs_dir, {"event": "run_finished", "group_id": outcome.group["group_id"],
                             "case": outcome.group["case"], "arm": attempt.manifest["arm"]["id"],
                             "run_id": attempt.manifest["run_id"], "status": attempt.manifest["status"],
                             "validity_errors": attempt.manifest.get("validity_errors"),
                             "cost_usd": run_cost(attempt.manifest, attempt.result), "counted_usd": counted,
                             "cost_estimated": estimated})
    return attempts, counted_total


def provider_mismatch(attempt: batch.Attempt) -> str | None:
    errors = [e for e in attempt.manifest.get("validity_errors") or [] if e.startswith("provider drift")]
    audit = attempt.manifest.get("generation_audit") or {}
    pinned = {openrouter_audit.normalize(p) for p in (attempt.manifest.get("arm") or {}).get("provider_order") or []}
    served = {openrouter_audit.normalize(p) for p in audit.get("providers_served") or []}
    if pinned and served - pinned:
        errors.append(f"served by {sorted(served - pinned)}, pinned {sorted(pinned)}")
    return "; ".join(errors) or None


def is_free(attempt: batch.Attempt) -> bool:
    return bool((attempt.manifest.get("arm") or {}).get("free"))


def group_invalid(outcome: GroupOutcome, attempts: list[batch.Attempt]) -> bool:
    """Group-level failure (stack, injection, restore, processing) or no paid run completed."""
    return bool(outcome.group_errors) or not any(a.completed for a in attempts if not is_free(a))


def error_signature(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())[:200]


def smoke_problems(attempts: list[batch.Attempt], arms: dict[str, dict], outcome: GroupOutcome) -> list[str]:
    """Stricter checks for the smoke group: completed runs, costs in range, pinned providers, clean restore."""
    problems = []
    attempts = [attempt for attempt in attempts if not is_free(attempt)]   # the free arm is never a smoke gate
    if len(attempts) != len(paid(arms)):
        problems.append(f"{len(attempts)} paid runs, expected {len(paid(arms))}")
    problems += [f"group: {error}" for error in outcome.group_errors]
    for attempt in attempts:
        arm_id = attempt.manifest["arm"]["id"]
        if not attempt.completed:
            problems.append(f"{arm_id}: {attempt.manifest['status']} {attempt.manifest.get('validity_errors')}")
        if attempt.result is None:
            problems.append(f"{arm_id}: no result file")
        cost = (attempt.result or {}).get("cost_usd_total")
        if cost is None or not 0 < float(cost) < SMOKE_MAX_RUN_USD:
            problems.append(f"{arm_id}: recorded cost {cost} not in (0, {SMOKE_MAX_RUN_USD})")
        audit = attempt.manifest.get("generation_audit") or {}
        if not audit.get("retrieved") or audit.get("missing"):
            problems.append(f"{arm_id}: generation audit incomplete")
        if provider_mismatch(attempt):
            problems.append(f"{arm_id}: provider mismatch {provider_mismatch(attempt)}")
        if attempt.key_failure:
            problems.append(f"{arm_id}: key failure {attempt.key_failure}")
    return problems


def stop_reasons(outcome: GroupOutcome, attempts: list[batch.Attempt], group_cost: float, quick_total: float,
                 arms: dict[str, dict], history: dict, total_cap: float = TOTAL_CAP_USD,
                 group_cap: float = GROUP_CAP_USD) -> list[str]:
    """Why the batch must stop after this group (empty: carry on); `history` carries streaks across groups."""
    reasons = []
    for attempt in attempts:
        arm_id = attempt.manifest["arm"]["id"]
        if is_free(attempt):
            continue   # the free arm never stops the batch; run_quick applies the free-tier rules to it
        if attempt.key_failure:
            reasons.append(f"key failure in {arm_id}: {attempt.key_failure}")
        mismatch = provider_mismatch(attempt)
        if mismatch:
            reasons.append(f"provider mismatch in {arm_id}: {mismatch}")
    if group_cost > group_cap:
        reasons.append(f"group cost ${group_cost:.4f} > ${group_cap:.2f}")
    if quick_total > total_cap:
        reasons.append(f"quick-eval total ${quick_total:.4f} > ${total_cap:.2f}")
    invalid = group_invalid(outcome, attempts)
    history["invalid_streak"] = history.get("invalid_streak", 0) + 1 if invalid else 0
    if history["invalid_streak"] >= 2:
        reasons.append("two consecutive invalid groups")
    seen = history.setdefault("infra_errors", set())
    for error in outcome.group_errors:
        signature = error_signature(error)
        if signature in seen:
            reasons.append(f"same infrastructure error twice: {error}")
        seen.add(signature)
    if outcome.group.get("smoke"):
        reasons += [f"smoke check: {problem}" for problem in smoke_problems(attempts, arms, outcome)]
    return reasons


# --- loop -------------------------------------------------------------------

def attempted_groups(runs_dir: Path) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for path in sorted(runs_dir.glob("*/manifest.json")) if runs_dir.exists() else []:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("group_id"):
            found.setdefault(manifest["group_id"], []).append(manifest)
    return found


def prepare_stack(runs_dir: Path, group_id: str) -> None:
    """batch.prepare_stack with one retry; a second failure stops the batch."""
    for attempt in (1, 2):
        try:
            batch.prepare_stack()
            return
        except Exception as error:
            batch.log(runs_dir, {"event": "stack_setup_failed", "group_id": group_id, "attempt": attempt,
                                 "error": str(error)[-500:]})
            if attempt == 2:
                raise batch.BatchStop(f"stack setup failed twice before {group_id}: {error}") from None


def stack_down(runs_dir: Path, group_id: str) -> None:
    try:
        batch.stack_cli("down")
    except Exception as error:
        batch.log(runs_dir, {"event": "stack_down_failed", "group_id": group_id, "error": str(error)[-500:]})


def free_arm_block(arm: dict, spend: batch.Spend, ledger: free_tier.Ledger, day: str,
                   requests_per_day: int | None) -> str | None:
    """Why the free arm cannot join the next group (None: it can); same gates as the batch."""
    problems = free_tier.config_problems(arm)
    if problems:
        return f"refused, it could cost money: {'; '.join(problems)}"
    if spend.arm(arm["id"]) > 0:
        return f"recorded spend ${spend.arm(arm['id']):.4f} on a $0 arm"
    return batch.free_gate(arm, ledger, day, requests_per_day)


def run_quick(config: dict, plan: dict, runs_dir: Path, *, arm_keys: tuple[str, ...] | list[str] = ARM_KEYS,
              folders: list[Path] | None = None, max_groups: int | None = None, only: set[str] | None = None,
              plan_sha: str | None = None, group_runner=run_group, prepare=prepare_stack, teardown=stack_down,
              ledger: free_tier.Ledger | None = None, today=free_tier.utc_day,
              requests_per_day: int | None = None, cases: set[str] | None = None,
              total_cap: float = TOTAL_CAP_USD, group_cap: float = GROUP_CAP_USD) -> int:
    """Run the plan's pending groups in order. Exit codes: 0 done, 2 refused, 3 key failure, 4 stop rule, 130 interrupt."""
    arms = quick_arms(config, arm_keys)
    folders = folders or spend_folders(runs_dir)
    ledger = ledger or free_tier.Ledger()
    runs_dir.mkdir(parents=True, exist_ok=True)
    done = attempted_groups(runs_dir)
    history: dict = {"invalid_streak": 0, "infra_errors": set()}
    for group in plan["groups"]:   # streaks carry over from earlier invocations
        if group["group_id"] in done:
            manifests = [m for m in done[group["group_id"]] if not (m.get("arm") or {}).get("free")]
            errors = (manifests or done[group["group_id"]])[0].get("group_errors") or []
            invalid = bool(errors) or not any(m.get("status") == "completed" for m in manifests)
            history["invalid_streak"] = history["invalid_streak"] + 1 if invalid else 0
            history["infra_errors"].update(error_signature(e) for e in errors)
    pending = [g for g in plan["groups"] if g["group_id"] not in done and (not only or g["group_id"] in only)
               and (not cases or g["case"] in cases)]
    batch.log(runs_dir, {"event": "quick_started", "arms": [a["id"] for a in arms.values()],
                         "groups_pending": [g["group_id"] for g in pending]})
    blocked: dict[str, str] = {}   # free arms sitting out for the rest of this invocation
    started = 0
    for group in pending:
        if max_groups is not None and started >= max_groups:
            print(f"stopped after --max-groups {max_groups}")
            return 0
        if batch.STOP_FILE.exists():
            print(f"stop file {batch.STOP_FILE} found; stopping")
            return 0
        day = today()
        spend = batch.Spend(config, folders)
        active = dict(paid(arms))
        for key, arm in arms.items():
            if not arm["free"]:
                continue
            reason = blocked.get(key) or free_arm_block(arm, spend, ledger, day, requests_per_day)
            if reason:
                blocked[key] = reason
                batch.log(runs_dir, {"event": "free_arm_skipped", "group_id": group["group_id"], "arm": arm["id"],
                                     "reason": reason})
                print(f"free arm {arm['id']} sits out {group['group_id']}: {reason}")
            else:
                active[key] = arm
        reservations, problems, money = group_reservation(config, active, folders, runs_dir, total_cap, group_cap)
        if problems:
            batch.log(runs_dir, {"event": "group_refused", "group_id": group["group_id"], "problems": problems, **money})
            print(f"group {group['group_id']} refused:\n- " + "\n- ".join(problems), file=sys.stderr)
            return 2
        request_keys = {}
        for key, arm in active.items():
            if arm["free"]:   # reserve a worst-case run of the day's request budget before it starts
                request_keys[key] = f"reserved quick-{group['group_id']}"
                ledger.record(arm["id"], day, request_keys[key], batch.free_limits(arm, requests_per_day)[1])
        batch.log(runs_dir, {"event": "group_reserved", "group_id": group["group_id"], "case": group["case"],
                             "arms": [a["id"] for a in active.values()], **money})
        print(f"[{runner.utc_now()}] {group['group_id']} {group['case']} x {', '.join(a['id'] for a in active.values())}"
              f" (reserved ${money['group_reservation_usd']:.4f}; quick ${money['quick_spent_usd']:.4f} of "
              f"${total_cap:.2f}; all ${money['global_spent_usd']:.4f} of ${money['global_budget_usd']:.2f})",
              flush=True)
        started += 1
        try:
            try:
                prepare(runs_dir, group["group_id"])
                outcome = group_runner(group, active, config, runs_dir, reservations, plan_sha)
            finally:
                teardown(runs_dir, group["group_id"])
        except KeyboardInterrupt:
            batch.log(runs_dir, {"event": "interrupted", "group_id": group["group_id"]})
            print("interrupted: the group was restored and recorded")
            return 130
        except batch.BatchStop as error:
            batch.log(runs_dir, {"event": "quick_stopped", "group_id": group["group_id"], "reason": str(error)})
            print(f"stopped: {error}", file=sys.stderr)
            return 4
        attempts, group_cost = settle(outcome, runs_dir)
        for attempt in attempts:
            key = attempt.manifest["arm"]["key"]
            if key in request_keys:   # settle requests; $0 / served-model / availability rules (may halt the arm)
                reason = batch.after_free_run(attempt, active[key], ledger, runs_dir, day, request_keys[key])
                if not reason and attempt.key_failure:
                    reason = f"paused: key failure {attempt.key_failure}"
                if reason:
                    blocked[key] = reason
                    batch.log(runs_dir, {"event": "free_arm_blocked", "arm": active[key]["id"], "reason": reason})
        quick_total = batch.Spend(config, [runs_dir]).total()
        for attempt in attempts:
            print(f"    {attempt.manifest['arm']['id']:20s} {attempt.manifest['status']:9s} "
                  f"cost ${run_cost(attempt.manifest, attempt.result) or 0:.4f} "
                  f"{attempt.manifest.get('validity_errors') or ''}")
        print(f"    group cost ${group_cost:.4f}; quick total ${quick_total:.4f}"
              + (f"; group errors {outcome.group_errors}" if outcome.group_errors else ""), flush=True)
        # Stop rules apply only between groups, so a running group is always restored and recorded first.
        reasons = stop_reasons(outcome, attempts, group_cost, quick_total, arms, history, total_cap, group_cap)
        batch.log(runs_dir, {"event": "group_finished", "group_id": group["group_id"], "group_cost_usd": group_cost,
                             "quick_total_usd": quick_total, "group_errors": outcome.group_errors,
                             "stop_reasons": reasons})
        if reasons:
            print("stopping:\n- " + "\n- ".join(reasons), file=sys.stderr)
            return 3 if any(r.startswith("key failure") for r in reasons) else 4
    print("nothing left to run in the quick plan" if not pending else "quick plan finished")
    return 0


# --- summary ----------------------------------------------------------------

SUMMARY_FIELDS = ("group_id", "case", "arm", "run_id", "status", "validity_errors", "root_cause_id",
                  "root_cause_summary", "confidence", "tool_calls_used", "observations_truncated_or_omitted",
                  "duration_s", "cost_usd", "cost_estimated", "providers_served", "fault_live_at_end",
                  "prompt_variant", "latency_metric_queries")


def one_line(text: str | None, limit: int = 200) -> str:
    line = " ".join((text or "").split())
    return line if len(line) <= limit else line[:limit - 3] + "..."


def summary_rows(runs_dir: Path, plan: dict) -> list[dict]:
    order = {g["group_id"]: n for n, g in enumerate(plan["groups"])}
    rows = []
    for path in runs_dir.glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not manifest.get("group_id"):
            continue
        run_id = manifest["run_id"]
        result_path = path.parent / f"{run_id}.result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        trace_path = path.parent / f"{run_id}.trace.jsonl"
        trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()] \
            if trace_path.exists() else []
        marked = sum(1 for event in trace if event.get("event") == "tool_call"
                     and any(marker in str(event.get("result", "")) for marker in TRUNCATION_MARKERS))
        latency = sum(1 for event in trace if event.get("event") == "tool_call" and event.get("tool_name") == "get_metrics"
                      and LATENCY_QUERY.search(str((event.get("args") or {}).get("promql", ""))))
        cost = run_cost(manifest, result or None)
        rows.append({
            "group_id": manifest["group_id"], "case": manifest["case"], "arm": manifest["arm"]["key"],
            "run_id": run_id, "status": manifest["status"],
            "validity_errors": " | ".join(manifest.get("validity_errors") or []),
            "root_cause_id": result.get("root_cause_id"), "root_cause_summary": one_line(result.get("summary")),
            "confidence": result.get("confidence"), "tool_calls_used": result.get("calls_used"),
            "observations_truncated_or_omitted": marked,
            "duration_s": round(result["wall_clock_s"], 1) if result.get("wall_clock_s") is not None else None,
            "cost_usd": round(cost, 6) if cost is not None else None,
            "cost_estimated": bool(manifest.get("cost_estimated")),
            "providers_served": ",".join((manifest.get("generation_audit") or {}).get("providers_served") or []),
            "fault_live_at_end": manifest.get("fault_live_at_end"),
            "prompt_variant": manifest.get("prompt_variant") or "v1", "latency_metric_queries": latency,
        })
    arm_order = {key: n for n, key in enumerate(ARM_KEYS)}
    rows.sort(key=lambda r: (order.get(r["group_id"], 99), arm_order.get(r["arm"], 99)))
    return rows


def write_summary(runs_dir: Path, plan: dict) -> Path:
    path = runs_dir / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows(runs_dir, plan))
    return path


def describe(config: dict, plan: dict, runs_dir: Path, arm_keys: tuple[str, ...] = ARM_KEYS,
             total_cap: float = TOTAL_CAP_USD, group_cap: float = GROUP_CAP_USD, cases: set[str] | None = None) -> str:
    arms = quick_arms(config, arm_keys)
    plan = {**plan, "groups": [g for g in plan["groups"] if not cases or g["case"] in cases]}
    tokens = config["token_model"]
    reserve, problems, ledger = group_reservation(config, arms, spend_folders(runs_dir), runs_dir, total_cap, group_cap)
    lines = [f"Quick eval ({NOTE}): {len(plan['groups'])} groups x {len(arms)} arms = "
             f"{len(plan['groups']) * len(arms)} runs, "
             f"seed {plan['seed']}, runs dir {runs_dir}"]
    done = attempted_groups(runs_dir)
    for group in plan["groups"]:
        state = "attempted" if group["group_id"] in done else "pending"
        lines.append(f"  {group['group_id']} {group['case']:20s} rep {group['rep']}"
                     f"{' (smoke)' if group.get('smoke') else ''}  {state}")
    for arm in arms.values():
        typical = preregister.estimate_run_usd(arm, tokens, "typical")
        lines.append(f"  {arm['id']:20s} {arm['model_id']:28s} typical ${typical:.4f}/run, reservation "
                     f"${reserve[arm['id']]:.4f}, spent ${ledger['arm_spent_usd'][arm['id']]:.4f} of cap ${arm['cap_usd']:.2f}")
    typical_group = sum(preregister.estimate_run_usd(a, tokens, "typical") for a in arms.values())
    lines.append(f"Per group: reservation ${ledger['group_reservation_usd']:.4f} (cap ${group_cap:.2f}), typical "
                 f"${typical_group:.4f}; plan typical ${typical_group * len(plan['groups']):.4f} "
                 f"(quick cap ${total_cap:.2f}, spent ${ledger['quick_spent_usd']:.4f}).")
    lines.append(f"Global: recorded ${ledger['global_spent_usd']:.4f} of ${ledger['global_budget_usd']:.2f} "
                 "(this runs dir + v1 quick + pre-pilot + matrix folders).")
    if problems:
        lines.append("Next group would be refused:\n- " + "\n- ".join(problems))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="write/verify the plan file and print it (no Docker)")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summary", action="store_true", help="write <runs dir>/summary.csv")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--groups", help="comma-separated group ids to run (default: all pending)")
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--arms", default=",".join(ARM_KEYS),
                        help="comma-separated arm keys run concurrently per group (default luna,glm,gemma; "
                             "a free arm such as nemotron is optional and never stops a group)")
    parser.add_argument("--free-requests-per-day", type=int,
                        help="free arm daily request budget (default arms.json free_tier, 50)")
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="v1",
                        help="agent prompt variant (AGENT_PROMPT_VARIANT; v2 needs its own --runs-dir)")
    parser.add_argument("--cases", help="comma-separated cases: run only the plan's groups of these cases")
    parser.add_argument("--pin", help="arm=pin[,arm=pin]: another pin from arms.json for this invocation only")
    parser.add_argument("--total-cap", type=float, default=TOTAL_CAP_USD,
                        help=f"quick-eval cap for this runs dir (default ${TOTAL_CAP_USD:.2f}; may only be lowered)")
    parser.add_argument("--group-cap", type=float, default=GROUP_CAP_USD,
                        help=f"per-group cap (default ${GROUP_CAP_USD:.2f}; may only be lowered)")
    parser.add_argument("--offline", action="store_true",
                        help="use the local fake endpoint (experiments.fake_model), never a paid API; own runs dir")
    args = parser.parse_args()

    os.environ.setdefault("EXPERIMENT_STACK_FILE", str(batch.STACK_FILE))
    if not (0 < args.total_cap <= TOTAL_CAP_USD and 0 < args.group_cap <= GROUP_CAP_USD):
        parser.error(f"--total-cap / --group-cap must be in (0, {TOTAL_CAP_USD}] / (0, {GROUP_CAP_USD}]")
    pins = parse_pins(args.pin)
    config = apply_pins(preregister.load_config(), pins)
    plan = write_plan() if args.plan else load_plan()
    runs_dir = args.runs_dir.resolve() if args.runs_dir else (OFFLINE_RUNS if args.offline else QUICK_RUNS)
    arm_keys = tuple(key.strip() for key in args.arms.split(",") if key.strip())
    cases = {case.strip() for case in args.cases.split(",") if case.strip()} if args.cases else None
    unknown = sorted((cases or set()) - {group["case"] for group in plan["groups"]})
    if unknown:
        parser.error(f"--cases: no plan groups for {unknown}")
    if args.prompt_variant != "v1" and runs_dir.resolve() == QUICK_RUNS.resolve():
        parser.error(f"--prompt-variant {args.prompt_variant} needs its own --runs-dir (not the v1 folder {QUICK_RUNS})")
    conflicts = variant_conflicts(runs_dir, args.prompt_variant)
    if conflicts:
        parser.error(f"{runs_dir} already holds runs with prompt variant(s) {conflicts}; use another --runs-dir")
    if args.plan:
        print(describe(config, plan, runs_dir, arm_keys, args.total_cap, args.group_cap, cases))
        return 0
    if args.summary:
        print(f"wrote {write_summary(runs_dir, plan)}")
        return 0

    folders = None
    if args.offline:
        from experiments.fake_model import serve_in_background
        _, url = serve_in_background()
        config = {**config, "base_url": url}
        os.environ["MODEL_API_KEY"] = "offline-fake-key"   # never send the real key to the fake endpoint
        folders = [runs_dir]                               # fake spend never enters the real ledger
        print(f"offline: fake endpoint {url} (not a model); runs -> {runs_dir}")
    rows = [{"case": group["case"]} for group in plan["groups"]]
    problems = batch.preflight(rows, "quick", True, None)
    if problems:
        print("refusing to start:\n- " + "\n- ".join(problems), file=sys.stderr)
        return 2
    dirty = runner.command(["git", "status", "--porcelain"]).stdout.strip()
    if dirty and not args.offline:
        print("refusing to start: commit the source first (manifests record source_dirty)", file=sys.stderr)
        return 2
    print(f"stack profile {os.environ['EXPERIMENT_STACK_FILE']} (project {runner.stack.project()}), runs -> {runs_dir}; "
          f"prompt {args.prompt_variant}, caps ${args.total_cap:.2f} total / ${args.group_cap:.2f} per group"
          + (f", pins {pins}" if pins else ""))
    batch.log(runs_dir, {"event": "quick_options", "prompt_variant": args.prompt_variant,
                         "system_prompt_sha256": prompt_sha256(args.prompt_variant), "cases": sorted(cases or []),
                         "pins": pins, "total_cap_usd": args.total_cap, "group_cap_usd": args.group_cap,
                         "spend_folders": [str(f) for f in folders or spend_folders(runs_dir)]})
    only = set(args.groups.split(",")) if args.groups else None
    try:
        return run_quick(config, plan, runs_dir, arm_keys=arm_keys, folders=folders, max_groups=args.max_groups,
                         only=only, plan_sha=preregister.sha256_text(PLAN_FILE),
                         requests_per_day=args.free_requests_per_day,
                         ledger=free_tier.Ledger(runs_dir / "free_ledger.json") if args.offline else None,
                         group_runner=functools.partial(run_group, prompt_variant=args.prompt_variant),
                         cases=cases, total_cap=args.total_cap, group_cap=args.group_cap)
    except KeyboardInterrupt:
        print("interrupted between groups")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
