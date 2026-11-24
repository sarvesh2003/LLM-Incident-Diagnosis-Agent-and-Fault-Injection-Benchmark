"""Resumable batch driver for the pre-registered benchmark: one fresh stack per schedule row.

The run folders are the ledger. Per-arm caps and a global budget gate each run; a key failure stops the batch.
Stop with Ctrl+C or experiments/state/batch.stop; re-running the same command resumes.

Usage:
  python -m experiments.batch --plan [--phase P2]
  python -m experiments.batch --pilot [--arm luna] [--case healthy]
  python -m experiments.batch --run [--phase P2] [--arm glm] [--case cpu_throttled] [--max-runs 6]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.config import api_key_from, dotenv_api_keys
from experiments import free_tier, preregister, runner
from experiments.aggregate import invalid_cause, run_cost

STACK_FILE = runner.ROOT / "experiments" / "bench_stack.json"
STOP_FILE = runner.STATE_DIR / "batch.stop"
PILOT_RUNS = runner.ROOT / "experiments" / "runs" / "pilot"
MINUTES_PER_RUN = 12            # stack lifecycle + agent, measured in the trial run
READY_SECONDS = 300
TRAFFIC_SECONDS = 180
STACK_IMAGES = ("go-server", "go-enricher", "python-producer")
ARM_VARIABLES = ("MODEL_BASE_URL", "MODEL_ID", "MODEL_PROVIDER", "ENV_LABEL", "MODEL_PROVIDER_ORDER",
                 "MODEL_QUANTIZATIONS", "MODEL_ALLOW_FALLBACKS", "MODEL_DATA_COLLECTION", "MODEL_SEED",
                 "MODEL_REASONING_EFFORT", "MODEL_MAX_PRICE", "MODEL_REQUIRE_ZERO_COST")
MANIFEST_ARM_KEYS = ("id", "key", "pin", "model_id", "provider_order", "quantizations", "price_usd_per_million",
                     "reasoning", "effective_temperature", "allow_fallbacks", "data_collection", "max_price",
                     "free", "require_zero_cost")
# Bad key (401), credits (402) or the key's own limit (403): stop the batch, never re-queue. A 429 is not one.
KEY_FAILURE = re.compile(r"Error code: (401|402|403)\b|HTTP (401|402|403)\b|PermissionDeniedError|Key limit exceeded",
                         re.IGNORECASE)
GLOBAL_BUDGET_USD = 5.25        # default for arms.json spend_guard.global_budget_usd
OBSERVED_RUN_FACTOR = 1.5       # reservation once an arm has a completed paid run: max(worst, factor x highest)


class BatchStop(RuntimeError):
    """Stop the whole batch (key failure, stack that cannot be prepared)."""


@dataclass
class Attempt:
    manifest: dict
    result: dict | None
    run_dir: Path

    @property
    def stderr(self) -> str:
        path = self.run_dir / "agent.stderr.log"
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    @property
    def completed(self) -> bool:
        return self.manifest.get("status") == "completed" and not self.manifest.get("validity_errors")

    @property
    def model_invalid(self) -> bool:
        return not self.completed and invalid_cause(self.manifest, self.result) == "model"

    @property
    def key_failure(self) -> str | None:
        texts = [*(self.manifest.get("validity_errors") or []), (self.result or {}).get("traceback_tail") or "",
                 self.stderr]
        match = next((KEY_FAILURE.search(text) for text in texts if KEY_FAILURE.search(text)), None)
        return match.group(0) if match else None

    @property
    def availability_failure(self) -> str | None:
        """Free arms only: a rate limit or unavailable provider, never a model failure."""
        return free_tier.availability_failure(self.manifest, self.result, self.stderr)

    @property
    def counts_for_requeue(self) -> bool:
        return not self.completed and "interrupted" not in (self.manifest.get("validity_errors") or []) \
            and not self.key_failure and not self.availability_failure


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_attempts(runs_dir: Path) -> dict[tuple[str, str], list[Attempt]]:
    """Attempts per (arm id, schedule row), oldest first."""
    attempts = defaultdict(list)
    for path in sorted(runs_dir.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not manifest.get("schedule_row"):
            continue
        result_path = path.parent / f"{manifest['run_id']}.result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        attempts[((manifest.get("arm") or {}).get("id"), manifest["schedule_row"])].append(
            Attempt(manifest, result, path.parent))
    for items in attempts.values():
        items.sort(key=lambda a: a.manifest.get("created_at_utc", ""))
    return attempts


def row_status(attempts: list[Attempt]) -> str:
    if any(a.completed or a.model_invalid for a in attempts):
        return "done"
    counted = sum(a.counts_for_requeue for a in attempts)
    return {0: "fresh", 1: "requeue"}.get(counted, "exhausted")


def pending_rows(schedule: list[dict], arms: dict[str, dict], attempts: dict) -> list[dict]:
    """Fresh rows in schedule order, then rows to re-run once (a row is done once completed or model-invalid)."""
    status = {row["row"]: row_status(attempts.get((arms[row["arm"]]["id"], row["row"]), [])) for row in schedule}
    return ([row for row in schedule if status[row["row"]] == "fresh"] +
            [row for row in schedule if status[row["row"]] == "requeue"])


# --- money: spend per arm, reservations, global budget --------------------

def needs_estimate(attempt: Attempt) -> bool:
    """True when the run's spend may be missing from its record (killed, crashed or unreported cost)."""
    # A timed-out call is still billed, retries included; a refused key is rejected before any billing.
    result = attempt.result
    if result is None:
        return "agent_started_at_utc" in attempt.manifest
    if result.get("cost_usd_total") is not None:
        return False
    if result.get("status") == "completed":
        return bool(result.get("llm_calls"))
    return not attempt.key_failure


def attempt_spend(attempt: Attempt, reservation: float) -> tuple[float, bool]:
    """(USD counted against the caps, estimated?); an estimated run counts at max(cost, stored reservation)."""
    cost = run_cost(attempt.manifest, attempt.result) or 0.0
    if not needs_estimate(attempt):
        return cost, False
    stored = attempt.manifest.get("cost_estimated_usd")
    return max(cost, reservation if stored is None else float(stored)), True


def arm_variants(config: dict) -> dict[str, dict]:
    """Every arm x pin of arms.json, resolved, by arm id (runs of a switched pin still count globally)."""
    variants = {}
    for arm in config["arms"]:
        for pin in arm["pins"]:
            resolved = preregister.resolve_arm(config, {**arm, "pin": pin})
            variants[resolved["id"]] = resolved
    return variants


class Spend:
    """The money ledger: every attempt in the given run folders (matrix and pre-pilot), per arm id."""

    def __init__(self, config: dict, folders: list[Path] | tuple[Path, ...]):
        guard = config.get("spend_guard") or {}
        self.budget = float(guard.get("global_budget_usd", GLOBAL_BUDGET_USD))
        self.factor = float(guard.get("observed_run_factor", OBSERVED_RUN_FACTOR))
        self.tokens = config["token_model"]
        self.variants = arm_variants(config)
        self.attempts: dict[str, list[Attempt]] = defaultdict(list)
        seen = set()
        for folder in folders:
            folder = Path(folder)
            if not folder.exists() or folder.resolve() in seen:
                continue
            seen.add(folder.resolve())
            for (arm_id, _), items in load_attempts(folder).items():
                self.attempts[arm_id].extend(items)

    def worst(self, arm_id: str) -> float:
        arm = self.variants.get(arm_id)
        return (preregister.estimate_run_usd(arm, self.tokens, "worst") if arm else None) or 0.0

    def highest_observed(self, arm_id: str) -> float | None:
        """The highest cost of a completed run of a paid arm, or None before its first one."""
        if (self.variants.get(arm_id) or {}).get("free"):
            return None
        costs = [cost for attempt in self.attempts.get(arm_id, []) if attempt.completed
                 and (cost := run_cost(attempt.manifest, attempt.result)) is not None]
        return max(costs, default=None)

    def reservation(self, arm_id: str) -> float:
        """Worst case until the arm has a completed paid run, then max(worst, factor x highest observed)."""
        highest = self.highest_observed(arm_id)
        worst = self.worst(arm_id)
        return worst if highest is None else max(worst, self.factor * highest)

    def arm(self, arm_id: str) -> float:
        reservation = self.reservation(arm_id)
        return sum(attempt_spend(attempt, reservation)[0] for attempt in self.attempts.get(arm_id, []))

    def total(self) -> float:
        return sum(self.arm(arm_id) for arm_id in list(self.attempts))

    def estimated_runs(self) -> int:
        return sum(needs_estimate(attempt) for items in self.attempts.values() for attempt in items)


def spend_folders(runs_dir: Path, others: tuple[Path, ...] | None, pilot_dir: Path = PILOT_RUNS) -> list[Path]:
    return [runs_dir, *((pilot_dir, runner.RUNS) if others is None else others)]


def select(schedule: list[dict], arm: str | None, case: str | None, phase: str | None = None) -> list[dict]:
    return [row for row in schedule if (not arm or row["arm"] == arm) and (not case or row["case"] == case)
            and (not phase or row.get("phase") == phase)]


def pilot_schedule(config: dict, arms: dict, case: str | None) -> list[dict]:
    cases = [case] if case else config["matrix"].get("pilot_cases", ["healthy"])
    return [{"row": f"pilot-{name}-{key}", "phase": "P1", "block": 0, "arm": key, "case": name, "rep": 1}
            for name in cases for key in arms]


def free_limits(arm: dict, requests_per_day: int | None = None) -> tuple[int, int]:
    """(requests per UTC day, worst-case requests per run)."""
    tier = arm.get("free_tier") or {}
    return int(requests_per_day or tier.get("requests_per_day") or 50), int(tier.get("worst_case_requests_per_run") or 16)


# --- GLM cost switch: the main phase refuses to start until a triggered switch is resolved ---

def cost_switch_status(config: dict, pilot_dir: Path = PILOT_RUNS) -> list[dict]:
    """One record per arm with a cost_switch rule: the pre-pilot median and which branch applies."""
    statuses = []
    attempts = load_attempts(pilot_dir) if pilot_dir.exists() else {}
    for arm in config["arms"]:
        rule = arm.get("cost_switch")
        if not arm.get("enabled") or not rule:
            continue
        default_id = f"{arm['key']}@{rule['default_pin']}"
        costs = [cost for (arm_id, _), items in attempts.items() if arm_id == default_id for attempt in items
                 if (cost := run_cost(attempt.manifest, attempt.result)) is not None]
        median = statistics.median(costs) if costs else None
        threshold = float(rule["threshold_median_usd_per_run"])
        record = {"arm": arm["key"], "default_pin": rule["default_pin"], "alternative_pin": rule["alternative_pin"],
                  "pin": arm["pin"], "pilot_runs_with_cost": len(costs), "pilot_median_usd": median,
                  "threshold_usd": threshold, "decision": rule.get("decision")}
        if median is None:
            record["status"] = "pending: no pre-pilot cost yet"
        elif median <= threshold:
            record["status"] = "not triggered"
        elif arm["pin"] == rule["alternative_pin"]:
            record["status"] = f"triggered -> switched to {arm['key']}@{rule['alternative_pin']}"
        elif rule.get("decision") == "accept":
            record["status"] = f"triggered -> accepted: {arm['key']} may stop early at its ${arm['cap_usd']:.2f} cap"
        else:
            record["status"] = "triggered -> UNDECIDED: switch the pin or set cost_switch.decision 'accept'"
        statuses.append(record)
    return statuses


# --- plan (no Docker, no network) ------------------------------------------

def plan(config: dict, schedule: list[dict], runs_dir: Path, ledger: free_tier.Ledger | None = None,
         requests_per_day: int | None = None, pilot_dir: Path = PILOT_RUNS,
         other_spend_dirs: tuple[Path, ...] | None = None) -> str:
    arms = preregister.enabled_arms(config)
    tokens = config["token_model"]
    ledger = ledger or free_tier.Ledger()
    attempts = load_attempts(runs_dir) if runs_dir.exists() else {}
    spend = Spend(config, spend_folders(runs_dir, other_spend_dirs, pilot_dir))
    pilot_spend = Spend(config, [pilot_dir])
    pending = pending_rows(schedule, arms, attempts)
    phases = Counter(row.get("phase") for row in schedule)
    lines = [f"Schedule: {len(schedule)} rows ({', '.join(f'{p} {n}' for p, n in sorted(phases.items()))}), "
             f"{len({r['block'] for r in schedule})} blocks, seed {config['matrix']['seed']}, runs dir {runs_dir}"]
    per_cell = Counter((r["arm"], r["case"]) for r in schedule)
    cases = [c for c in config["matrix"]["cases"] if any(r["case"] == c for r in schedule)]
    lines.append("rows per arm x case (" + ", ".join(f"{n + 1}={c}" for n, c in enumerate(cases)) + ")")
    lines.append(f"  {'':9s}" + " ".join(f"{n + 1:>2}" for n in range(len(cases))))
    for key in arms:
        lines.append(f"  {key:9s}" + " ".join(f"{per_cell[(key, c)]:>2}" for c in cases)
                     + f"  = {sum(per_cell[(key, c)] for c in cases)}")
    lines.append("")
    lines.append("arm id                 model                                  typical$/run worst$/run  rows "
                 "pending  est.typical$ est.worst$  spent$   cap$ reserve$")
    totals = Counter()
    money = lambda v, width: f"{v:>{width}.4f}" if v is not None else f"{'n/a':>{width}}"
    for key, arm in arms.items():
        typical, worst = (preregister.estimate_run_usd(arm, tokens, s) for s in ("typical", "worst"))
        rows = sum(r["arm"] == key for r in schedule)
        todo = sum(r["arm"] == key for r in pending)
        spent = spend.arm(arm["id"])
        reserve = spend.reservation(arm["id"])
        est_typical = typical * todo if typical is not None else None
        est_worst = worst * todo if worst is not None else None
        totals.update({"typical": est_typical or 0, "worst": est_worst or 0, "rows": rows, "todo": todo,
                       "cap": arm["cap_usd"], "spent": spent, "paid_todo": 0 if arm["free"] else todo})
        lines.append(f"{arm['id']:22s} {arm['model_id']:38s} {money(typical, 12)} {money(worst, 10)} "
                     f"{rows:>5} {todo:>7} {money(est_typical, 13)} {money(est_worst, 10)} {spent:>7.4f} "
                     f"{arm['cap_usd']:>6.2f} {reserve:>8.4f}")
        if worst is not None and spent + reserve * todo > arm["cap_usd"] and not arm["free"]:
            lines.append(f"  note: spent ${spent:.2f} + {todo} runs at the reservation (${reserve * todo:.2f}) exceed "
                         "the cap; the cap stops the arm")
        if typical is None:
            lines.append(f"  WARNING: pin {arm['pin']} has no price; the batch refuses to run this arm")
    lines.append(f"{'total':22s} {'':38s} {'':12s} {'':10s} {totals['rows']:>5} {totals['todo']:>7} "
                 f"{totals['typical']:>13.4f} {totals['worst']:>10.4f} {totals['spent']:>7.4f} {totals['cap']:>6.2f}")
    total = spend.total()
    lines.append(f"\nGlobal budget ${spend.budget:.2f} (arms.json spend_guard): recorded spend so far ${total:.4f} over "
                 f"all arms (pre-pilot ${pilot_spend.total():.4f} included; {spend.estimated_runs()} run(s) counted at "
                 f"their reservation because their cost was not recorded), ${spend.budget - total:.4f} left. "
                 "A run starts only if spend + its reservation stays within the budget.")
    lines.append(f"Per-run reservation: the configured worst case until an arm has a completed paid run, then "
                 f"max(worst case, {spend.factor:g} x its highest observed run cost). Pre-pilot spend counts "
                 "against each arm's cap.")
    for arm in arms.values():
        highest = spend.highest_observed(arm["id"])
        lines.append(f"  {arm['id']:22s} reservation ${spend.reservation(arm['id']):.4f} (worst case "
                     f"${spend.worst(arm['id']):.4f}, highest observed "
                     f"{f'${highest:.4f}' if highest is not None else 'n/a'}); spent ${spend.arm(arm['id']):.4f} "
                     f"(pre-pilot ${pilot_spend.arm(arm['id']):.4f}) of cap ${arm['cap_usd']:.2f}, "
                     f"${arm['cap_usd'] - spend.arm(arm['id']):.4f} left")

    for key, arm in arms.items():
        if not arm["free"]:
            continue
        per_day, worst_requests = free_limits(arm, requests_per_day)
        todo = sum(r["arm"] == key for r in pending)
        runs_per_day, days = free_tier.estimate_days(todo, per_day, worst_requests)
        day = free_tier.utc_day()
        halted = ledger.halted(arm["id"])
        problems = free_tier.config_problems(arm)
        state = (f"HALTED ({halted['reason']})" if halted else
                 f"REFUSED ({'; '.join(problems)})" if problems else "active")
        lines.append(f"\nFree arm {arm['id']} ({arm['model_id']}): cap ${arm['cap_usd']:.2f}, ':free' only, no "
                     f"fallbacks, max_price 0/0, zero-cost guard, data_collection={arm['data_collection']} "
                     f"(user-approved, this arm only). Status: {state}.")
        lines.append(f"  budget {per_day} requests/UTC day, worst case {worst_requests} requests/run -> "
                     f"{runs_per_day} runs/day; {todo} pending runs -> ~{days} UTC days. "
                     f"Used today ({day}): {ledger.used(arm['id'], day)}/{per_day}.")
    hours = totals["todo"] * MINUTES_PER_RUN / 60
    paid_hours = totals["paid_todo"] * MINUTES_PER_RUN / 60
    lines.append(f"\nPending runs: {totals['todo']} (~{hours:.1f} h machine time at ~{MINUTES_PER_RUN} min/run on one "
                 f"stack; paid arms {totals['paid_todo']} runs ~{paid_hours:.1f} h, never waiting on the free arm, "
                 "whose runs are interleaved while its daily budget lasts).")
    pilot = pilot_schedule(config, arms, None)
    pilot_typical = sum(preregister.estimate_run_usd(arms[r["arm"]], tokens, "typical") or 0 for r in pilot)
    pilot_worst = sum(preregister.estimate_run_usd(arms[r["arm"]], tokens, "worst") or 0 for r in pilot)
    lines.append(f"Pre-pilot P1 (--pilot): {len(pilot)} runs ({len(arms)} arms x "
                 f"{', '.join(config['matrix'].get('pilot_cases', ['healthy']))}), typical ${pilot_typical:.4f}, "
                 f"worst ${pilot_worst:.4f}, ~{len(pilot) * MINUTES_PER_RUN / 60:.1f} h.")
    for status in cost_switch_status(config, pilot_dir):
        median = f"${status['pilot_median_usd']:.4f}" if status["pilot_median_usd"] is not None else "n/a"
        lines.append(f"Cost switch {status['arm']}: pre-pilot median {median} over {status['pilot_runs_with_cost']} "
                     f"run(s) vs ${status['threshold_usd']:.3f} -> {status['status']} (pin {status['pin']}).")
    lines.append(f"Caps total ${totals['cap']:.2f}, pre-pilot included. Keep the OpenRouter key limit above the global "
                 f"budget (${spend.budget:.2f}); a key-limit 403 stops the batch.")
    return "\n".join(lines)


# --- one run --------------------------------------------------------------

def stack_cli(action: str, timeout: int = 900) -> None:
    done = subprocess.run([sys.executable, "-m", "experiments.stack", action], cwd=runner.ROOT,
                          env=runner.isolated_env(), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError(f"stack {action} exited {done.returncode}: {done.stderr.strip()[-500:]}")


def prepare_stack() -> None:
    """A fresh stack with its clean-stack marker and traffic flowing."""
    stack_cli("down")
    stack_cli("up")
    ready, detail = runner.wait_ready(READY_SECONDS)
    if not ready:
        raise RuntimeError(detail)
    runner.initialize_stack()
    stack_cli("start-traffic")
    if not runner.wait_for_growth("raw_transactions", TRAFFIC_SECONDS)["grew"]:
        raise RuntimeError(f"no raw traffic within {TRAFFIC_SECONDS}s of start-traffic")


@contextmanager
def arm_environment(config: dict, arm: dict):
    """Set one arm's agent settings in os.environ for the run, then restore them."""
    saved = {name: os.environ.get(name) for name in ARM_VARIABLES}
    os.environ.update(preregister.arm_environment(config, arm))
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def execute(row: dict, arm: dict, config: dict, runs_dir: Path, mode: str, prereg_sha: str | None) -> Path:
    extra = {"batch": mode, "schedule_row": row["row"], "schedule_block": row["block"], "schedule_rep": row["rep"],
             "schedule_phase": row.get("phase"), "prereg_sha256": prereg_sha,
             "arm": {key: arm.get(key) for key in MANIFEST_ARM_KEYS}}
    case = runner.configured_case(row["case"])
    with arm_environment(config, arm):
        try:
            for attempt in (1, 2):
                try:
                    prepare_stack()
                    break
                except Exception as error:
                    log(runs_dir, {"event": "stack_setup_failed", "row": row["row"], "attempt": attempt,
                                   "error": str(error)[-500:]})
                    if attempt == 2:
                        raise BatchStop(f"stack setup failed twice before {row['row']}: {error}") from None
            return runner.run_case(case, runs_dir, extra)
        finally:
            try:
                stack_cli("down")
            except Exception as error:
                log(runs_dir, {"event": "stack_down_failed", "row": row["row"], "error": str(error)[-500:]})


def log(runs_dir: Path, event: dict) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (runs_dir / "batch_log.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps({"at_utc": utc_now(), **event}) + "\n")


# --- preflight and loop ---------------------------------------------------

def resolved_api_key() -> str | None:
    """MODEL_API_KEY, else OPENROUTER_API_KEY; environment first, then .env, as the agent resolves it."""
    return api_key_from({**dotenv_api_keys(runner.DOTENV), **os.environ})


def preflight(rows: list[dict], mode: str, allow_dirty: bool, prereg: dict | None, config: dict | None = None) -> list[str]:
    problems = []
    if not resolved_api_key():
        problems.append("no API key: set MODEL_API_KEY or OPENROUTER_API_KEY (environment or .env)")
    if mode == "matrix":
        if prereg is None:
            problems.append("experiments/prereg/prereg.json is missing: run `python -m experiments.preregister freeze`")
        else:
            changed = preregister.protocol_mismatches(prereg)
            if changed:
                problems.append(f"protocol files changed since freezing: {changed}")
        dirty = runner.command(["git", "status", "--porcelain"])
        if dirty.stdout.strip() and not allow_dirty:
            problems.append("source tree is dirty (manifests would record source_dirty=true); commit first")
        for status in cost_switch_status(config) if config else []:
            if "UNDECIDED" in status["status"]:
                problems.append(f"cost switch for {status['arm']}: pre-pilot median ${status['pilot_median_usd']:.4f} "
                                f"> ${status['threshold_usd']:.3f}; set its pin to {status['alternative_pin']} or "
                                "cost_switch.decision to 'accept' (then re-freeze)")
    project = runner.stack.project()
    images = [f"{project}-{service}" for service in STACK_IMAGES]
    if any(runner.CASES[row["case"]].active_state == "netem_delay" for row in rows):
        images.append(runner.NETSHOOT_IMAGE)
    for image in images:
        if runner.command(["docker", "image", "inspect", image]).returncode != 0:
            hint = (f"docker pull {image}" if image == runner.NETSHOOT_IMAGE
                    else "python -m experiments.stack build (with the bench stack file)")
            problems.append(f"image {image} is not available locally: {hint}")
    return problems


def halt_free_arm(ledger: free_tier.Ledger, runs_dir: Path, arm: dict, reason: str, run_id: str | None) -> str:
    record = ledger.halt(arm["id"], reason, run_id)
    path = free_tier.write_halt(runs_dir, record)
    log(runs_dir, {"event": "free_arm_halted", **record})
    print(f"free arm {arm['id']} HALTED permanently: {reason}. {free_tier.HOLM_NOTE}. Recorded in {path}",
          file=sys.stderr)
    return f"halted: {reason}"


def after_free_run(attempt: Attempt, arm: dict, ledger: free_tier.Ledger, runs_dir: Path, day: str,
                   reservation: str) -> str | None:
    """Settle the request count and apply the free-arm halt/pause rules; returns a block reason or None."""
    _, worst_requests = free_limits(arm)
    ledger.settle(arm["id"], day, reservation, attempt.manifest["run_id"],
                  free_tier.requests_used(attempt.result, worst_requests))
    accepted = (arm.get("free_tier") or {}).get("accepted_served_models") or [arm["model_id"]]
    violation = free_tier.cost_violation(attempt.manifest, attempt.result, accepted)
    if violation:
        return halt_free_arm(ledger, runs_dir, arm, violation, attempt.manifest["run_id"])
    kind = attempt.availability_failure
    if kind == "gone":
        return halt_free_arm(ledger, runs_dir, arm, "model or endpoints unavailable (permanent)",
                             attempt.manifest["run_id"])
    if kind:
        if kind == "daily_limit":
            ledger.mark_exhausted(arm["id"], day)
        days = ledger.note_availability(arm["id"], day)
        log(runs_dir, {"event": "free_arm_paused", "arm": arm["id"], "kind": kind, "run_id": attempt.manifest["run_id"],
                       "availability_days": days})
        if days >= free_tier.UNAVAILABLE_AFTER_DAYS:
            return halt_free_arm(ledger, runs_dir, arm, f"availability failures on {days} distinct UTC days without "
                                 "a completed run", attempt.manifest["run_id"])
        print(f"free arm {arm['id']} paused ({kind}): availability failure, not a model failure; the row stays "
              "pending and resumes on a later invocation")
        return f"paused: {kind}"
    if attempt.completed:
        ledger.note_completed(arm["id"])
    return None


def free_gate(arm: dict, ledger: free_tier.Ledger, day: str, requests_per_day: int | None) -> str | None:
    """Why a free arm cannot start a run now (halted or not enough budget today), or None."""
    halted = ledger.halted(arm["id"])
    if halted:
        return f"halted: {halted['reason']}"
    limit, worst = free_limits(arm, requests_per_day)
    remaining = ledger.remaining(arm["id"], day, limit)
    if remaining < worst:
        return (f"waiting: {remaining} of {limit} requests left on UTC day {day}, a run may need {worst}; "
                "resumes on the next UTC day")
    return None


def run_batch(config: dict, rows: list[dict], runs_dir: Path, mode: str, max_runs: int | None,
              prereg_sha: str | None, executor=execute, ledger: free_tier.Ledger | None = None,
              today=free_tier.utc_day, requests_per_day: int | None = None,
              other_spend_dirs: tuple[Path, ...] | None = None) -> int:
    arms = preregister.enabled_arms(config)
    folders = spend_folders(runs_dir, other_spend_dirs)
    tokens = config["token_model"]
    ledger = ledger or free_tier.Ledger()
    worst = {key: preregister.estimate_run_usd(arm, tokens, "worst") for key, arm in arms.items()}
    unpriced = sorted({row["arm"] for row in rows if worst[row["arm"]] is None})
    if unpriced:
        print(f"refusing: no price for the selected pin of {unpriced} (arms.json)", file=sys.stderr)
        return 2
    blocked: dict[str, str] = {}
    for key in sorted({row["arm"] for row in rows}):
        arm = arms[key]
        if arm["free"]:
            problems = free_tier.config_problems(arm)
            reason = (f"refused, it could cost money: {'; '.join(problems)}" if problems
                      else free_gate(arm, ledger, today(), requests_per_day))
            if reason:
                blocked[key] = reason
                print(f"free arm {arm['id']} not run now: {reason}")
                log(runs_dir, {"event": "free_arm_blocked", "arm": arm["id"], "reason": reason})
    started, prefer_free = 0, True
    log(runs_dir, {"event": "batch_started", "mode": mode, "rows_selected": len(rows)})
    while True:
        attempts = load_attempts(runs_dir)
        pending = pending_rows(rows, arms, attempts)
        todo = [row for row in pending if row["arm"] not in blocked]
        if not todo:
            if pending:
                print(f"{len(pending)} rows still pending for stopped or waiting arms: "
                      + "; ".join(f"{key}: {reason}" for key, reason in sorted(blocked.items())))
            else:
                print("nothing left to run for this selection")
            return 0
        if max_runs is not None and started >= max_runs:
            print(f"stopped after --max-runs {max_runs}; {len(todo)} rows still pending")
            return 0
        if STOP_FILE.exists():
            print(f"stop file {STOP_FILE} found; stopping cleanly ({len(todo)} rows pending)")
            return 0
        free_rows = [row for row in todo if arms[row["arm"]]["free"]]
        paid_rows = [row for row in todo if not arms[row["arm"]]["free"]]
        row = None
        # Free rows alternate with paid rows only while the free quota lasts, so paid arms never wait on it.
        if free_rows and (prefer_free or not paid_rows):
            candidate = free_rows[0]
            reason = free_gate(arms[candidate["arm"]], ledger, today(), requests_per_day)
            if reason:
                blocked[candidate["arm"]] = reason
                print(f"free arm {arms[candidate['arm']]['id']} not run now: {reason}")
                log(runs_dir, {"event": "free_arm_blocked", "arm": arms[candidate["arm"]]["id"], "reason": reason})
                continue
            row = candidate
        row = row or paid_rows[0]
        arm = arms[row["arm"]]
        spend = Spend(config, folders)
        spent, reserve, total = spend.arm(arm["id"]), spend.reservation(arm["id"]), spend.total()
        # Reserve a worst case before starting, so a run whose cost is unknown cannot overshoot the cap.
        if spent + reserve > arm["cap_usd"]:
            blocked[row["arm"]] = f"capped at ${arm['cap_usd']:.2f}"
            print(f"arm {arm['id']} stopped: spent ${spent:.4f} (pre-pilot included) + reservation ${reserve:.4f} "
                  f"> cap ${arm['cap_usd']:.2f}")
            log(runs_dir, {"event": "arm_capped", "arm": arm["id"], "spent_usd": spent, "reservation_usd": reserve})
            continue
        if total + reserve > spend.budget:
            blocked[row["arm"]] = f"global budget ${spend.budget:.2f}"
            print(f"arm {arm['id']} not run: recorded spend ${total:.4f} (all arms, pre-pilot included) + "
                  f"reservation ${reserve:.4f} > global budget ${spend.budget:.2f}")
            log(runs_dir, {"event": "global_budget", "arm": arm["id"], "total_spent_usd": total,
                           "reservation_usd": reserve, "budget_usd": spend.budget})
            continue
        day, reservation = today(), f"reserved {row['row']}"
        if arm["free"]:
            ledger.record(arm["id"], day, reservation, free_limits(arm, requests_per_day)[1])
        print(f"[{utc_now()}] {row['row']} {row.get('phase', '')} block {row['block']} {arm['id']} {row['case']} "
              f"(spent ${spent:.4f} of ${arm['cap_usd']:.2f}, reservation ${reserve:.4f}; all arms ${total:.4f} "
              f"of ${spend.budget:.2f})", flush=True)
        started += 1
        prefer_free = not arm["free"]
        try:
            run_dir = executor(row, arm, config, runs_dir, mode, prereg_sha)
        except KeyboardInterrupt:
            log(runs_dir, {"event": "interrupted", "row": row["row"]})
            print("interrupted: the run was restored and recorded; re-run the same command to resume")
            return 130  # a free arm's worst-case reservation stays counted for the day
        except BatchStop as error:
            log(runs_dir, {"event": "batch_stopped", "row": row["row"], "reason": str(error)})
            print(f"batch stopped: {error}", file=sys.stderr)
            return 2
        attempt = Attempt(json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")), None, run_dir)
        result_path = run_dir / f"{attempt.manifest['run_id']}.result.json"
        attempt.result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        cost = run_cost(attempt.manifest, attempt.result)
        if needs_estimate(attempt):
            # Spend that never reached the record still counts against the caps, at this reservation.
            attempt.manifest.update({"cost_estimated": True, "cost_estimated_usd": reserve})
            runner.write_manifest(run_dir / "manifest.json", attempt.manifest)
        counted, estimated = attempt_spend(attempt, reserve)
        log(runs_dir, {"event": "run_finished", "row": row["row"], "arm": arm["id"], "case": row["case"],
                       "run_id": attempt.manifest["run_id"], "status": attempt.manifest["status"],
                       "validity_errors": attempt.manifest.get("validity_errors"), "cost_usd": cost,
                       "counted_usd": counted, "cost_estimated": estimated})
        print(f"    {attempt.manifest['status']}: {run_dir.name} cost ${cost if cost is not None else 'unknown'}"
              + (f" (counted at the reservation: ${counted:.4f})" if estimated else "")
              + (f" errors {attempt.manifest['validity_errors']}" if attempt.manifest.get("validity_errors") else ""))
        failure = attempt.key_failure
        if failure:
            log(runs_dir, {"event": "batch_stopped", "row": row["row"], "reason": f"key failure {failure}"})
            print(f"batch stopped: OpenRouter refused the key ({failure}); raise the key limit or fix the key, "
                  "then re-run the same command", file=sys.stderr)
            return 3
        if arm["free"]:
            reason = after_free_run(attempt, arm, ledger, runs_dir, day, reservation)
            if reason:
                blocked[row["arm"]] = reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="schedule, progress and cost estimate only")
    mode.add_argument("--run", action="store_true", help="run the pre-registered matrix (resumes)")
    mode.add_argument("--pilot", action="store_true",
                      help="pre-pilot P1: every arm x matrix.pilot_cases into experiments/runs/pilot")
    parser.add_argument("--arm", help="only this arm key")
    parser.add_argument("--case", choices=[name for name in runner.CASES], help="only this case")
    parser.add_argument("--phase", choices=("P2", "P3"), help="only this phase of the matrix")
    parser.add_argument("--max-runs", type=int, help="stop after starting this many runs")
    parser.add_argument("--runs-dir", type=Path, help="default experiments/runs/v2 (matrix) or experiments/runs/pilot")
    parser.add_argument("--arms", type=Path, default=preregister.ARMS)
    parser.add_argument("--schedule", type=Path, default=preregister.SCHEDULE)
    parser.add_argument("--allow-dirty", action="store_true", help="matrix only: accept uncommitted source changes")
    parser.add_argument("--free-requests-per-day", type=int,
                        help="free arm daily request budget (default arms.json free_tier, 50)")
    args = parser.parse_args()

    os.environ.setdefault("EXPERIMENT_STACK_FILE", str(STACK_FILE))
    config = preregister.load_config(args.arms)
    arms = preregister.enabled_arms(config)
    if args.arm and args.arm not in arms:
        parser.error(f"--arm must be one of the enabled arms {sorted(arms)}")
    if args.pilot:
        schedule = pilot_schedule(config, arms, args.case)
        runs_dir = args.runs_dir or PILOT_RUNS
        rows = select(schedule, args.arm, None)
    else:
        schedule = preregister.read_schedule(args.schedule)
        runs_dir = args.runs_dir or runner.RUNS
        rows = select(schedule, args.arm, args.case, args.phase)
    if args.plan:
        print(plan(config, rows, runs_dir, requests_per_day=args.free_requests_per_day))
        return 0

    mode_name = "pilot" if args.pilot else "matrix"
    prereg = json.loads(preregister.PREREG.read_text(encoding="utf-8")) if preregister.PREREG.exists() else None
    problems = preflight(rows, mode_name, args.allow_dirty, prereg, config)
    if problems:
        print("refusing to start:\n- " + "\n- ".join(problems), file=sys.stderr)
        return 2
    print(f"stack profile {os.environ['EXPERIMENT_STACK_FILE']} (project {runner.stack.project()}), runs -> {runs_dir}")
    prereg_sha = preregister.sha256_text(preregister.PREREG) if prereg else None
    try:
        return run_batch(config, rows, runs_dir, mode_name, args.max_runs, prereg_sha,
                         requests_per_day=args.free_requests_per_day)
    except KeyboardInterrupt:
        print("interrupted between runs; re-run the same command to resume")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
