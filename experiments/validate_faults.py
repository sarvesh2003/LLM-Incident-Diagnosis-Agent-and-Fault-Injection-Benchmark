"""Exercise the reversible registered faults on a fresh isolated stack without calling a model.

Default: inject, sustain, recover. --gate: Phase 0 fault-validity gates G1-G5 per case, written as JSON and CSV;
--summarize combines gate files from separate stacks (a case passes only if it passed on every stack).
Usage: python -m experiments.validate_faults [--gate] [--case NAME ...] [--set CASE.KEY=VALUE] | --summarize A.json B.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments import symptoms
from experiments.runner import (
    CASES, RECOVERY_SECONDS, ROOT, case_role, configured_case, consume_marker, fault_is_live, restore_and_wait,
    script_action, stack_health, utc_now, verify_stack_marker, wait_for_growth,
)
from experiments.stack import isolated_env, project

NAMES = ("redis_node_stopped", "ingestion_paused", "broker_stopped", "enricher_paused",
         "client_timeout_too_low", "cpu_throttled", "latency_injected", "latency_enricher")
HIDDEN = tuple(name for name in NAMES if case_role(name) == "hidden")
GATES = ("G1", "G2", "G3", "G4", "G5")
GATE_SUSTAIN_SECONDS = 360


def snapshot() -> dict:
    return symptoms.snapshot(isolated_env()["PROMETHEUS_URL"])


def mcp_python() -> str:
    configured = os.environ.get("MCP_PYTHON")
    if configured:
        return configured
    bundled = ROOT / "mcp-server" / "venv" / "Scripts" / "python.exe"
    return str(bundled) if bundled.exists() else sys.executable


def run_probe(name: str, overrides: dict[str, str]) -> dict:
    """experiments.observability_probe in the MCP server's interpreter; `matched` None if it could not run."""
    settings = [arg for key, value in overrides.items() for arg in ("--setting", f"{key}={value}")]
    try:
        done = subprocess.run([mcp_python(), "-m", "experiments.observability_probe", "--case", name, *settings],
                              cwd=ROOT, env=isolated_env(), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=180)
        if done.returncode == 0:
            return json.loads(done.stdout)
        return {"matched": None, "error": f"probe exited {done.returncode}: {done.stderr.strip()[-500:]}"}
    except Exception as error:
        return {"matched": None, "error": f"probe failed: {error}"}


# --- mechanics mode ---------------------------------------------------------

def validate(name: str, log_dir: Path, sustain_seconds: int = 5, overrides: dict | None = None) -> dict:
    case = configured_case(name, overrides)
    case_dir = log_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    entry = {"case": name, "case_env": case.script_env(), "started_at_utc": utc_now(), "status": "invalid",
             "errors": [], "symptoms_before": snapshot()}
    try:
        result = script_action(case, "inject", case_dir)
        if result.returncode != 0:
            raise RuntimeError(f"inject exited {result.returncode}: {result.stderr[-500:]}")
        time.sleep(case.settle_seconds)
        live, state = fault_is_live(case)
        entry["live_probe"] = {"at_utc": utc_now(), "live": live, "state": state}
        if not live:
            raise RuntimeError("fault did not reach its expected active state")
        time.sleep(sustain_seconds)
        still_live, state = fault_is_live(case)
        entry["sustained_probe"] = {"at_utc": utc_now(), "live": still_live, "state": state}
        entry["symptoms_sustained"] = snapshot()
        if not still_live:
            raise RuntimeError(f"fault was not sustained for {sustain_seconds} seconds")
    except Exception as error:
        entry["errors"].append(str(error))
    finally:
        restore_errors(case, case_dir, entry)
    if not entry["errors"]:
        entry["status"] = "passed"
    return entry


def restore_errors(case, case_dir: Path, entry: dict) -> None:
    """Restore, wait for recovery and for enriched output; errors go to entry['errors'] and entry['restore_errors']."""
    errors = []
    try:
        restored, entry["recovery_probe"] = restore_and_wait(case, case_dir)
        errors.extend(restored)
        entry["processing_probe"] = wait_for_growth("enriched_transactions", RECOVERY_SECONDS)
        if not entry["processing_probe"]["grew"]:
            errors.append("enriched output did not resume")
    except Exception as error:
        errors.append(f"recovery: {error}")
    entry["restore_errors"] = errors
    entry["errors"].extend(errors)
    entry["finished_at_utc"] = utc_now()


# --- Phase 0 gate -----------------------------------------------------------

def gate(name: str, log_dir: Path, sustain_seconds: int = GATE_SUSTAIN_SECONDS, overrides: dict | None = None,
         reference_gap: int = 30, check_every: int = 30, probe=run_probe, sleep=time.sleep,
         clock=time.monotonic) -> dict:
    """G1-G5 for one case; returns the full record with `gates` and `verdict`."""
    overrides = overrides or {}
    case = configured_case(name, overrides)
    case_dir = log_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    entry = {"case": name, "role": case_role(name), "target": case.service, "case_env": case.script_env(),
             "sustain_seconds": sustain_seconds, "started_at_utc": utc_now(), "errors": [], "live_checks": []}
    entry["symptoms_reference"] = snapshot()
    sleep(reference_gap)
    entry["symptoms_before"] = snapshot()
    entry["probe_before"] = probe(name, overrides)
    try:  # restore always, even if injection fails part way
        result = script_action(case, "inject", case_dir)
        if result.returncode != 0:
            raise RuntimeError(f"inject exited {result.returncode}: {result.stderr[-500:]}")
        sleep(case.settle_seconds)
        deadline = clock() + sustain_seconds
        while True:
            live, state = fault_is_live(case)
            entry["live_checks"].append({"at_utc": utc_now(), "live": live, "state": state})
            if not live:
                raise RuntimeError("fault was not live at a check inside the sustain window")
            remaining = deadline - clock()
            if remaining <= 0:
                break
            sleep(min(check_every, remaining))
        entry["symptoms_sustained"] = snapshot()
        entry["probe_during"] = probe(name, overrides)
    except Exception as error:
        entry["errors"].append(str(error))
    finally:
        restore_errors(case, case_dir, entry)
    entry["gates"] = evaluate_gates(entry)
    entry["verdict"] = "pass" if all(g["pass"] for g in entry["gates"].values()) else "fail"
    return entry


def evaluate_gates(entry: dict) -> dict:
    """G1 live throughout, G2 symptom, G3 observable via a tool, G4 silent on healthy, G5 clean restore."""
    live = entry["live_checks"]
    sustained = "symptoms_sustained" in entry
    found = symptoms.symptoms_present(entry["symptoms_before"], entry.get("symptoms_sustained")) if sustained else {}
    reference = symptoms.symptoms_present(entry["symptoms_reference"], entry["symptoms_before"])
    during, before = entry.get("probe_during") or {}, entry.get("probe_before") or {}
    return {
        "G1": {"pass": sustained and len(live) >= 2 and all(c["live"] for c in live),
               "detail": f"{sum(c['live'] for c in live)}/{len(live)} live checks over {entry['sustain_seconds']} s"},
        "G2": {"pass": bool(found), "detail": found or "no component met a symptom criterion"},
        "G3": {"pass": during.get("matched") is True,
               "detail": during.get("error") or [c.get("detail") for c in during.get("calls", [])] or "not probed"},
        "G4": {"pass": not reference and before.get("matched") is False,
               "detail": {"reference_symptoms": reference,
                          "probe_before": before.get("error") or [c.get("detail") for c in before.get("calls", [])]}},
        "G5": {"pass": "restore_errors" in entry and not entry["restore_errors"],
               "detail": entry.get("restore_errors", "not restored")},
    }


def phase0_decision(verdicts: dict[str, bool]) -> dict:
    """Pre-registered go/no-go over hidden cases (a case passes only if it passed on every stack)."""
    passed = [name for name in HIDDEN if verdicts.get(name)]
    off_server = [name for name in passed if CASES[name].service != "go-server"]
    go = (len(passed) >= 2 and bool(off_server)) or len(passed) >= 3
    return {"passed_hidden": passed, "passed_not_go_server": off_server, "decision": "GO" if go else "NO-GO",
            "rule": "GO if >= 2 hidden cases pass and one targets something other than go-server, or >= 3 pass"}


def summarize(paths: list[Path]) -> dict:
    files = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    names = sorted({entry["case"] for data in files for entry in data["cases"]})
    verdicts = {name: all(any(e["case"] == name and e["verdict"] == "pass" for e in data["cases"]) for data in files)
                for name in names}
    return {"files": [str(path) for path in paths], "case_passed_on_every_stack": verdicts,
            **phase0_decision(verdicts)}


def write_gate_report(output: Path, results: list[dict]) -> Path:
    table = output.with_suffix(".csv")
    with table.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["case", "target", "case_env", *GATES, "verdict", "errors"])
        for entry in results:
            writer.writerow([entry["case"], entry["target"], json.dumps(entry["case_env"], sort_keys=True),
                             *("pass" if entry["gates"][g]["pass"] else "FAIL" for g in GATES), entry["verdict"],
                             "; ".join(entry["errors"])])
    return table


def parse_sets(items: list[str]) -> dict[str, dict[str, str]]:
    overrides: dict[str, dict[str, str]] = {}
    for item in items:
        target, _, value = item.partition("=")
        name, _, key = target.partition(".")
        if not (name in CASES and key and value):
            raise SystemExit(f"--set expects CASE.KEY=VALUE, got {item!r}")
        overrides.setdefault(name, {})[key] = value
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="result file (default: a new file under experiments/runs/fault-validation/)")
    parser.add_argument("--case", choices=NAMES, action="append", help="only these cases (repeatable)")
    parser.add_argument("--sustain", type=int, metavar="SECONDS",
                        help=f"sustained-state window (default 5; {GATE_SUSTAIN_SECONDS} with --gate)")
    parser.add_argument("--gate", action="store_true", help="Phase 0 fault-validity gates G1-G5 (default: hidden cases)")
    parser.add_argument("--set", action="append", default=[], metavar="CASE.KEY=VALUE",
                        help="script setting override, e.g. cpu_throttled.LOW_CPUS=0.01")
    parser.add_argument("--reference-gap", type=int, default=30, help="gate: seconds between the two healthy snapshots")
    parser.add_argument("--check-every", type=int, default=30, help="gate: seconds between live checks")
    parser.add_argument("--between", type=int, default=60, help="gate: seconds to wait after a restore before the next case")
    parser.add_argument("--summarize", type=Path, nargs="+", metavar="GATE_JSON",
                        help="combine gate files from separate fresh stacks into the Phase 0 decision (no Docker)")
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.summarize), indent=2))
        return 0
    overrides = parse_sets(args.set)
    names = tuple(args.case or (HIDDEN if args.gate else NAMES))
    for name in names:
        configured_case(name, overrides.get(name))  # reject a bad setting before touching the stack
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "-gate" if args.gate else ""
    output = args.output or ROOT / "experiments" / "runs" / "fault-validation" / f"{stamp}{suffix}.json"

    healthy, states = stack_health()
    marker_valid, marker_error = verify_stack_marker()
    if not healthy or not marker_valid:
        raise RuntimeError(f"isolated stack is not ready: healthy={healthy}, marker={marker_error}, states={states}")
    consume_marker(f"fault-validation-{stamp}")

    log_dir = output.parent / f"{output.stem}_logs"
    results = []
    for index, name in enumerate(names):
        if args.gate:
            if index:
                time.sleep(args.between)
            results.append(gate(name, log_dir, args.sustain or GATE_SUSTAIN_SECONDS, overrides.get(name),
                                args.reference_gap, args.check_every))
            if results[-1]["restore_errors"]:
                break  # the stack is no longer known to be clean
        else:
            results.append(validate(name, log_dir, args.sustain or 5, overrides.get(name)))
            if results[-1]["status"] != "passed":
                break
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.gate:
        verdicts = {entry["case"]: entry["verdict"] == "pass" for entry in results}
        summary = {"schema_version": "3.0", "purpose": "Phase 0 fault-validity gates G1-G5, no model",
                   "compose_project": project(), "created_at_utc": utc_now(), "cases_requested": list(names),
                   "cases": results, "this_stack": phase0_decision(verdicts),
                   "status": "passed" if len(results) == len(names) and all(verdicts.values()) else "failed"}
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        table = write_gate_report(output, results)
        print(table.read_text(encoding="utf-8"), end="")
        print(json.dumps({"status": summary["status"], "this_stack": summary["this_stack"]["decision"],
                          "result": str(output), "table": str(table)}))
        return 0 if summary["status"] == "passed" else 1
    passed = len(results) == len(names) and all(r["status"] == "passed" for r in results)
    summary = {"schema_version": "2.2", "purpose": "fault mechanics and processing recovery, no model diagnosis",
               "compose_project": project(), "created_at_utc": utc_now(), "cases": results,
               "status": "passed" if passed else "failed"}
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "cases": {r["case"]: r["status"] for r in results},
                      "result": str(output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
