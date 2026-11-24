"""Pre-registered non-LLM baselines R0-R3, applied offline to each run and paired with the agent.

R0-R2 read evidence entry 0 (exactly what the model saw first); R3 reads the model-free runbook snapshots.
Each rule is scored strict (target plus state or mechanism) and for localization (target only).
Usage: python -m experiments.baseline_rule [--runs DIR] [--output CSV]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from experiments import runbook_rule
from experiments.runner import CASES, REQUIRED_SERVICES, RUNS, TIMEOUT_VARIABLES

LINE = re.compile(r"^(?P<service>[^:\s]+): state=(?P<state>[^,\s]+)", re.MULTILINE)
STATUS = re.compile(r'^(?P<service>[^:\s]+): state=running, status="Up (?P<up>[^"(]+)', re.MULTILINE)
UPTIME = re.compile(r"^(?:about )?(?P<count>an?|\d+) (?P<unit>second|minute|hour|day|week|month|year)s?$")
UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800, "month": 2592000,
                "year": 31536000}
CONTAINER_STATE_OF = {"stopped": "exited", "paused": "paused"}
HEALTHY = "healthy"
RECENT_SECONDS = 120   # R2: "recently restarted" (a run's stack is up for about 5-8 minutes at entry 0)
RULES = ("r1", "r0", "r2", "r3")
# R3 strict scoring: the mechanism each running-target fault must be named with.
R3_MECHANISM = {"netem_delay": "network_delay", "cpu_limited": "cpu", "redis_timeouts_low": "config"}


def parse_states(observation: str) -> list[tuple[str, str]]:
    return [(match["service"], match["state"]) for match in LINE.finditer(observation)]


def rule_r1(observation: str) -> str:
    """The first required service (infrastructure first, a dependency prior) that is not running, else healthy."""
    states = dict(parse_states(observation))
    for service in REQUIRED_SERVICES:
        state = states.get(service, "absent")
        if state != "running":
            return f"{service}:{state}"
    return HEALTHY


def rule_r0(observation: str) -> str:
    """Naive floor: every exited or paused container (it false-alarms on the one-shot redis-cluster-setup job)."""
    flagged = [f"{service}:{state}" for service, state in parse_states(observation) if state in ("exited", "paused")]
    return "+".join(flagged) or HEALTHY


def uptime_seconds(text: str) -> float | None:
    """Seconds in a Docker "Up ..." status ("45 seconds", "About a minute", "Less than a second")."""
    text = text.strip().lower()
    if text.startswith("less than a second"):
        return 0.5
    match = UPTIME.match(text)
    if not match:
        return None
    count = 1 if match["count"] in ("a", "an") else int(match["count"])
    return count * UNIT_SECONDS[match["unit"]]


def rule_r2(observation: str) -> str:
    """R1, else the single required service up for less than RECENT_SECONDS (a recreated container)."""
    r1 = rule_r1(observation)
    if r1 != HEALTHY:
        return r1
    uptimes = {match["service"]: uptime_seconds(match["up"]) for match in STATUS.finditer(observation)}
    known = {service: up for service, up in uptimes.items() if service in REQUIRED_SERVICES and up is not None}
    recent = [service for service in REQUIRED_SERVICES if known.get(service, RECENT_SECONDS) < RECENT_SECONDS]
    if len(recent) == 1 and len(known) == len(REQUIRED_SERVICES):
        return f"{recent[0]}:recently_restarted"
    return HEALTHY


def rule_r3(before: dict | None, after: dict | None, thresholds: dict | None = None) -> str | None:
    """None when either runbook snapshot is missing (the rule cannot be applied)."""
    return runbook_rule.decide(before, after, thresholds)


def reports_nothing(prediction: str) -> bool:
    return prediction in (HEALTHY, runbook_rule.ABSTAIN)


def localizes(case_name: str, prediction: str) -> bool:
    """Localization: nothing reported on a control, else exactly the case's target component."""
    case = CASES[case_name]
    if case.script is None:
        return reports_nothing(prediction)
    return "+" not in prediction and prediction.split(":")[0] == case.service


def r3_correct(case_name: str, prediction: str) -> bool:
    """R3 strict: the target and the registered mechanism (abstain on a control)."""
    case = CASES[case_name]
    if case.script is None:
        return prediction == runbook_rule.ABSTAIN
    mechanism = R3_MECHANISM.get(case.active_state)
    service, _, named = prediction.partition(":")
    if service != case.service or mechanism is None or not named.startswith(mechanism):
        return False
    if mechanism == "config":
        settings = named.partition(":")[2].split("+")
        return any(setting in TIMEOUT_VARIABLES for setting in settings)
    return named == mechanism


def is_correct(case_name: str, prediction: str) -> bool:
    case = CASES[case_name]
    if case.script is None:
        return prediction == HEALTHY
    expected = CONTAINER_STATE_OF.get(case.active_state)
    return expected is not None and f"{case.service}:{expected}" in prediction.split("+")


def entry_zero(trace: list[dict]) -> str | None:
    """The result text of the first baseline get_container_state call, if the trace has one."""
    for event in trace:
        if event.get("event") == "tool_call" and event.get("node") == "baseline":
            return event.get("result") if event.get("tool_name") == "get_container_state" else None
    return None


def load_trace(run_dir: Path, run_id: str) -> list[dict]:
    path = run_dir / f"{run_id}.trace.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def apply_rules(run_dir: Path, manifest: dict, thresholds: dict | None = None) -> dict:
    """Predictions and correctness of every rule; "" where a rule cannot be applied to this run."""
    case = manifest["case"]
    observation = entry_zero(load_trace(run_dir, manifest["run_id"]))
    row = {"run_id": manifest["run_id"], "case": case,
           "arm": (manifest.get("arm") or {}).get("id") or manifest.get("model_id") or ""}
    for name in RULES:
        row[f"{name}_prediction"] = row[f"{name}_correct"] = row[f"{name}_localized"] = ""
    if observation is not None:
        for name, rule in (("r1", rule_r1), ("r0", rule_r0), ("r2", rule_r2)):
            prediction = rule(observation)
            row.update({f"{name}_prediction": prediction, f"{name}_correct": is_correct(case, prediction),
                        f"{name}_localized": localizes(case, prediction)})
    r3 = rule_r3(manifest.get("runbook_snapshot_before"), manifest.get("runbook_snapshot"), thresholds)
    if r3 is not None:
        row.update({"r3_prediction": r3, "r3_correct": r3_correct(case, r3), "r3_localized": localizes(case, r3)})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path, help="CSV file (default: stdout)")
    args = parser.parse_args()
    rows = [apply_rules(path.parent, json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(args.runs.glob("*/manifest.json"))]
    fields = ["run_id", "case", "arm",
              *(f"{name}_{part}" for name in RULES for part in ("prediction", "correct", "localized"))]
    out = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
