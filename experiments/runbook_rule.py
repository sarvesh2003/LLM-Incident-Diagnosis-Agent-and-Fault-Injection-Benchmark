"""Rule R3: a model-free config/metrics runbook that names a cause or abstains, from the runbook snapshots.

It reads only signals the agent's own tools can observe, never the case name or the agent's answer.
Thresholds in prereg/r3_thresholds.json come only from healthy-run data (MARGIN x the largest healthy value).
Usage: python -m experiments.runbook_rule {derive,show} [--runs DIR ...] [--gate-file JSON ...] [--min-pairs 4]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.runner import ROOT

THRESHOLDS = ROOT / "experiments" / "prereg" / "r3_thresholds.json"
ABSTAIN = "abstain"
MARGIN = 1.5
SERVICES = ("go-server", "go-enricher")
KEYS = {
    "go-server": {"p50": "server_p50_s", "requests": "server_requests_per_s", "errors": "server_errors_per_s",
                  "cpu": "server_cpu_per_s", "gomaxprocs": "server_gomaxprocs"},
    "go-enricher": {"p50": "enricher_p50_s", "requests": "enricher_processed_per_s",
                    "errors": "enricher_errors_per_s", "cpu": "enricher_cpu_per_s",
                    "gomaxprocs": "enricher_gomaxprocs"},
}
PROVISIONAL = {"status": "provisional", "margin": MARGIN, "latency_ratio": 3.0, "error_ratio": 0.01,
               "cpu_rate": {"go-server": None, "go-enricher": None}, "derived_from": None,
               "note": "provisional values (gate G2 levels); derive from healthy runs before the main phase"}


def load_thresholds(path: Path = THRESHOLDS) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(PROVISIONAL)


def _number(red: dict | None, key: str) -> float | None:
    value = (red or {}).get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def error_ratio(red: dict | None, service: str) -> float | None:
    errors, throughput = _number(red, KEYS[service]["errors"]), _number(red, KEYS[service]["requests"])
    if errors is None:
        return None
    # go-server counts every received request; the enricher counts successes only.
    total = (throughput or 0.0) + (errors if service == "go-enricher" else 0.0)
    return errors / total if total > 0 else None


def latency_ratio(before: dict | None, after: dict | None, service: str) -> float | None:
    b, a = _number(before, KEYS[service]["p50"]), _number(after, KEYS[service]["p50"])
    return a / b if b and a is not None else None


def config_changes(before: dict, after: dict, service: str) -> list[str]:
    b, a = (before.get("config") or {}).get(service), (after.get("config") or {}).get(service)
    if not isinstance(b, dict) or not isinstance(a, dict):
        return []
    return sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))


def cpu_signal(before: dict | None, after: dict | None, service: str, thresholds: dict) -> bool:
    keys = KEYS[service]
    b_procs, a_procs = _number(before, keys["gomaxprocs"]), _number(after, keys["gomaxprocs"])
    if b_procs is not None and a_procs is not None and a_procs < b_procs:
        return True
    limit, cpu = (thresholds.get("cpu_rate") or {}).get(service), _number(after, keys["cpu"])
    return limit is not None and cpu is not None and cpu > limit


def decide(before: dict | None, after: dict | None, thresholds: dict | None = None) -> str | None:
    """First match wins: config diff, then an anomaly as cpu, network_delay or errors; None if a snapshot is missing."""
    if not before or not after:
        return None
    thresholds = thresholds or load_thresholds()
    for service in SERVICES:
        changed = config_changes(before, after, service)
        if changed:
            return f"{service}:config:{'+'.join(changed)}"
    red_before, red_after = before.get("red"), after.get("red")
    for service in SERVICES:
        ratio, errors = latency_ratio(red_before, red_after, service), error_ratio(red_after, service)
        slow = ratio is not None and ratio > thresholds["latency_ratio"]
        failing = errors is not None and errors > thresholds["error_ratio"]
        if not (slow or failing):
            continue
        if cpu_signal(red_before, red_after, service, thresholds):
            return f"{service}:cpu"
        return f"{service}:network_delay" if slow else f"{service}:errors"
    return ABSTAIN


# --- thresholds from healthy data only ------------------------------------

def healthy_pairs(runs_dirs: list[Path], gate_files: list[Path]) -> list[tuple[dict, dict, str]]:
    """(before red, after red, source) from healthy runs and from gate reference pairs."""
    pairs = []
    for runs in runs_dirs:
        for path in sorted(runs.glob("*/manifest.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            before, after = manifest.get("runbook_snapshot_before"), manifest.get("runbook_snapshot")
            if manifest.get("case") == "healthy" and before and after:
                pairs.append((before.get("red"), after.get("red"), f"run {manifest['run_id']}"))
    for path in gate_files:
        for entry in json.loads(path.read_text(encoding="utf-8")).get("cases", []):
            if entry.get("symptoms_reference") and entry.get("symptoms_before"):
                pairs.append((entry["symptoms_reference"], entry["symptoms_before"],
                              f"gate {path.name} {entry['case']} reference"))
    return pairs


def derive(pairs: list[tuple[dict, dict, str]], margin: float = MARGIN) -> dict:
    ratios, errors, cpu = [], [], {service: [] for service in SERVICES}
    for before, after, _ in pairs:
        for service in SERVICES:
            ratio = latency_ratio(before, after, service)
            if ratio is not None:
                ratios.append(ratio)
            for red in (before, after):
                value = error_ratio(red, service)
                if value is not None:
                    errors.append(value)
                rate = _number(red, KEYS[service]["cpu"])
                if rate is not None:
                    cpu[service].append(rate)
    if not ratios:
        raise ValueError("no healthy pair has a latency value on both sides")
    return {"status": "derived", "margin": margin, "latency_ratio": round(margin * max(ratios), 4),
            "error_ratio": round(margin * max(errors or [0.0]), 6),
            "cpu_rate": {service: round(margin * max(values), 6) if values else None
                         for service, values in cpu.items()},
            "derived_from": {"pairs": len(pairs), "sources": [source for _, _, source in pairs],
                             "max_healthy_latency_ratio": max(ratios), "max_healthy_error_ratio": max(errors or [0.0])},
            "derived_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "each threshold = margin x the largest healthy value; a signal must exceed it"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("derive", "show"))
    parser.add_argument("--runs", type=Path, action="append", default=[], help="runs folder with healthy runs")
    parser.add_argument("--gate-file", type=Path, action="append", default=[], help="Phase 0 gate JSON")
    parser.add_argument("--min-pairs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=THRESHOLDS)
    args = parser.parse_args()
    if args.action == "show":
        print(json.dumps(load_thresholds(args.output), indent=2))
        return 0
    pairs = healthy_pairs(args.runs, args.gate_file)
    if len(pairs) < args.min_pairs:
        print(f"refusing: {len(pairs)} healthy snapshot pairs, need at least {args.min_pairs}")
        return 2
    derived = derive(pairs)
    args.output.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(derived, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
