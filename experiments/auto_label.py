"""Automatic second rater for diagnoses (NEVER a score): correct, abstain, wrong or unrated against the registered case.

Label every run by hand before looking at this, then compare with --agreement (Cohen's kappa).
Usage: python -m experiments.auto_label [--runs DIR] [--output CSV] [--agreement]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from experiments.runner import CASES, REQUIRED_SERVICES, RUNS
from experiments.score import LABELS, load_labels

# Failure-mode terms per registered active state; every pattern must match.
MODE_TERMS = {
    "stopped": [r"stop|exit|down|not running|offline|dead|crash"],
    "paused": [r"pause|frozen|freez|suspend"],
    "redis_timeouts_low": [r"time ?out|timed out|deadline", r"redis"],
    "cpu_limited": [r"cpu|throttl|quota|nanocpu|cfs"],
    "netem_delay": [r"latenc|delay|netem|round.trip|rtt"],
}
NONE_LIKE = re.compile(r"^(none|n/?a|unknown|no[-_ ]?(issue|fault|problem|incident)s?(-found|-detected)?|"
                       r"(pipeline[-_ ])?healthy|all[-_ ]healthy|inconclusive)$")
FAULT_TERMS = re.compile("|".join(term for terms in MODE_TERMS.values() for term in terms))


def mentions(text: str, service: str) -> bool:
    """Whole-name match, treating '-', '_' and spaces alike ('go server' names go-server)."""
    flat = re.sub(r"[\s_]+", "-", text.lower())
    return re.search(rf"(?<![a-z0-9]){re.escape(service)}(?![a-z0-9])", flat) is not None


def auto_label(case_name: str, result: dict | None) -> str:
    """Correct needs the exact target, a matching failure-mode term and no other service as root cause."""
    if not result or result.get("status") != "completed" or result.get("root_cause_id") in (None, "PARSE_FAILURE"):
        return "unrated"
    root = str(result.get("root_cause_id") or "").strip().lower()
    text = f"{root} {result.get('summary') or ''}".lower()
    root_is_none = bool(NONE_LIKE.match(root))
    case = CASES[case_name]
    if case.script is None:
        named_fault = not root_is_none and (any(mentions(root, s) for s in REQUIRED_SERVICES)
                                            or FAULT_TERMS.search(root) is not None)
        return "wrong" if named_fault else "correct"
    names_target = mentions(text, case.service)
    mode_matches = all(re.search(term, text) for term in MODE_TERMS[case.active_state])
    other_cause = any(mentions(root, s) for s in REQUIRED_SERVICES if s != case.service)
    if not root_is_none and names_target and mode_matches and not other_cause:
        return "correct"
    if root_is_none or (result.get("confidence") == "inconclusive" and not any(
            mentions(root, s) for s in REQUIRED_SERVICES)):
        return "abstain"
    return "wrong"


def label_runs(runs: Path) -> list[dict]:
    rows = []
    for path in sorted(runs.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        result_path = path.parent / f"{manifest['run_id']}.result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        rows.append({"run_id": manifest["run_id"], "case": manifest["case"],
                     "arm": (manifest.get("arm") or {}).get("id") or manifest.get("model_id") or "",
                     "auto_label": auto_label(manifest["case"], result),
                     "root_cause_id": (result or {}).get("root_cause_id") or ""})
    return rows


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Unweighted kappa over the label categories present; None if undefined."""
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(a == b for a, b in pairs) / total
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[c] * second[c] for c in set(first) | set(second)) / (total * total)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 4)


def agreement(rows: list[dict], labels: dict[str, dict[str, str]]) -> dict:
    pairs, disagreements = [], []
    for row in rows:
        human = labels.get(row["run_id"])
        if not human or human.get("run_validity") != "valid" or row["auto_label"] == "unrated":
            continue
        pairs.append((human["diagnosis_label"], row["auto_label"]))
        if pairs[-1][0] != pairs[-1][1]:
            disagreements.append({"run_id": row["run_id"], "case": row["case"],
                                  "human": pairs[-1][0], "auto": pairs[-1][1]})
    return {"compared": len(pairs), "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
            "cohen_kappa": cohen_kappa(pairs), "disagreements": disagreements}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path, help="CSV (default: <runs>/auto_labels.csv)")
    parser.add_argument("--agreement", action="store_true", help="also print agreement with human labels")
    parser.add_argument("--labels", type=Path, default=LABELS)
    args = parser.parse_args()
    rows = label_runs(args.runs)
    output = args.output or args.runs / "auto_labels.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["run_id", "case", "arm", "auto_label", "root_cause_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} runs auto-labelled (second rater, not a score): {output}")
    if args.agreement:
        json.dump(agreement(rows, load_labels(args.labels)), sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
