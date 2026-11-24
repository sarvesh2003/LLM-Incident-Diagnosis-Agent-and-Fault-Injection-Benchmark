"""Model-blind label sheet: export shuffled, arm-redacted runs for human labelling, then merge the labels.

The labeller sees the case and its rubric but never the model, arm, provider, cost, tokens or timings.
Usage:
  python -m experiments.label_sheet export [--runs DIR] [--out DIR] [--seed N] [--include-invalid]
  python -m experiments.label_sheet merge --sheet FILLED.csv --key label_key.csv [--labels CSV]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from experiments.runner import RUNS, case_role
from experiments.score import LABELS, load_labels

FILL_FIELDS = ("reviewer", "run_validity", "validity_reason", "diagnosis_label", "evidence_label",
               "detected", "localized", "notes")
SHEET_FIELDS = ("blind_id", "case", "case_role", "rubric", "root_cause_id", "summary", "confidence",
                "self_reported_confidence", "packet", *FILL_FIELDS)
LABEL_FIELDS = ("run_id", "reviewer", "run_validity", "validity_reason", "diagnosis_label", "evidence_label",
                "detected", "localized", "notes")
# Keys that identify or hint at the arm; removed at any depth of the packet.
REDACTED_KEYS = {"model", "model_id", "env_label", "served_model", "served_models", "cost_usd", "cost_usd_total",
                 "cost_reported_calls", "generation_ids", "provider", "endpoint_host", "model_settings", "arm",
                 "llm_ms", "llm_ms_total", "input_tokens", "output_tokens", "reasoning_tokens",
                 "input_tokens_total", "output_tokens_total", "reasoning_tokens_total", "system_prompt",
                 "tool_specs", "tool_schema_sha256", "vram_pre", "vram_post", "wall_clock_s"}
DEFAULT_SEED = 20261123


def redact(value, names: list[str]):
    """A copy without REDACTED_KEYS, with any arm or model name in a string replaced by [model]."""
    if isinstance(value, dict):
        return {key: redact(item, names) for key, item in value.items() if key not in REDACTED_KEYS}
    if isinstance(value, list):
        return [redact(item, names) for item in value]
    if isinstance(value, str) and names:
        return re.sub("|".join(re.escape(name) for name in names), "[model]", value, flags=re.IGNORECASE)
    return value


def identity_names(manifest: dict) -> list[str]:
    arm = manifest.get("arm") or {}
    names = {manifest.get("model_id"), arm.get("id"), arm.get("model_id"), arm.get("key")}
    for model in (manifest.get("model_id"), arm.get("model_id")):
        if model and "/" in model:
            names.add(model.split("/", 1)[1].split(":")[0])
    # Longest first, and nothing so short that it would blank ordinary words.
    return sorted((name for name in names if name and len(name) >= 4), key=len, reverse=True)


def packet(run_dir: Path, manifest: dict, rubric: str) -> dict:
    run_id = manifest["run_id"]
    result_path, trace_path = run_dir / f"{run_id}.result.json", run_dir / f"{run_id}.trace.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    trace = ([json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
             if trace_path.exists() else [])
    evidence = [{"index": index, "node": e.get("node"), "tool_name": e.get("tool_name"), "args": e.get("args"),
                 "observed_at": e.get("observed_at"), "result": e.get("result")}
                for index, e in enumerate(e for e in trace if e.get("event") == "tool_call")]
    decisions = [{"action": e.get("action"), "model_response": e.get("model_response")}
                 for e in trace if e.get("event") == "action_selected"]
    conclusion = {key: result.get(key) for key in ("root_cause_id", "summary", "reasoning", "confidence",
                                                     "model_self_reported_confidence", "downgrade_reason",
                                                     "cited_evidence_indices", "status", "calls_used")}
    data = {"case": manifest["case"], "case_role": case_role(manifest["case"]), "rubric": rubric,
            "machine_validity": {"status": manifest.get("status"), "validity_errors": manifest.get("validity_errors"),
                                 "live_probe_before": manifest.get("live_probe_before"),
                                 "live_probe_after": manifest.get("live_probe_after")},
            "conclusion": conclusion, "evidence_log": evidence, "agent_decisions": decisions}
    return redact(data, identity_names(manifest))


def export(runs: Path, out: Path, rubrics: dict[str, str], seed: int = DEFAULT_SEED,
           include_invalid: bool = False) -> int:
    manifests = []
    for path in sorted(runs.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if include_invalid or (manifest.get("status") == "completed" and not manifest.get("validity_errors")):
            manifests.append((path.parent, manifest))
    random.Random(seed).shuffle(manifests)
    (out / "packets").mkdir(parents=True, exist_ok=True)
    sheet_rows, key_rows = [], []
    for number, (run_dir, manifest) in enumerate(manifests, start=1):
        blind_id = f"L{number:03d}"
        data = packet(run_dir, manifest, rubrics.get(manifest["case"], ""))
        (out / "packets" / f"{blind_id}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        conclusion = data["conclusion"]
        sheet_rows.append({"blind_id": blind_id, "case": data["case"], "case_role": data["case_role"],
                           "rubric": data["rubric"], "root_cause_id": conclusion.get("root_cause_id") or "",
                           "summary": conclusion.get("summary") or "", "confidence": conclusion.get("confidence") or "",
                           "self_reported_confidence": conclusion.get("model_self_reported_confidence") or "",
                           "packet": f"packets/{blind_id}.json", **{field: "" for field in FILL_FIELDS}})
        key_rows.append({"blind_id": blind_id, "run_id": manifest["run_id"]})
    write_csv(out / "label_sheet.csv", SHEET_FIELDS, sheet_rows)
    write_csv(out / "label_key.csv", ("blind_id", "run_id"), key_rows)
    return len(sheet_rows)


def merge(sheet: Path, key: Path, labels_path: Path) -> int:
    mapping = {row["blind_id"]: row["run_id"] for row in read_csv(key)}
    existing = load_labels(labels_path) if labels_path.exists() else {}
    new_rows = []
    for row in read_csv(sheet):
        if not (row.get("reviewer") or "").strip():
            continue
        run_id = mapping.get(row["blind_id"])
        if run_id is None:
            raise ValueError(f"blind id {row['blind_id']} is not in {key}")
        if run_id in existing or any(r["run_id"] == run_id for r in new_rows):
            raise ValueError(f"{row['blind_id']} is already labelled in {labels_path}")
        new_rows.append({"run_id": run_id, **{field: (row.get(field) or "").strip() for field in LABEL_FIELDS[1:]}})
    if not labels_path.exists() or not labels_path.read_text(encoding="utf-8").strip():
        write_csv(labels_path, LABEL_FIELDS, [])
    with labels_path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=LABEL_FIELDS, lineterminator="\n").writerows(new_rows)
    return len(new_rows)


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> int:
    from experiments.preregister import RUBRIC

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("export", "merge"))
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, help="export: default <runs>/labelling")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--include-invalid", action="store_true", help="export machine-invalid runs too")
    parser.add_argument("--sheet", type=Path, help="merge: the filled label_sheet.csv")
    parser.add_argument("--key", type=Path, help="merge: label_key.csv from the export")
    parser.add_argument("--labels", type=Path, default=LABELS)
    args = parser.parse_args()
    if args.action == "export":
        out = args.out or args.runs / "labelling"
        count = export(args.runs, out, RUBRIC, args.seed, args.include_invalid)
        print(f"{count} runs -> {out / 'label_sheet.csv'} (packets in {out / 'packets'}; do not open label_key.csv)")
        return 0
    if not (args.sheet and args.key):
        parser.error("merge needs --sheet and --key")
    print(f"{merge(args.sheet, args.key, args.labels)} labels appended to {args.labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
