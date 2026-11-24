"""Benchmark tables per model arm and case from manifests, results and human labels only (analysis_plan.json).

Every metric is computed for hidden faults, visible faults, healthy controls and each fault on its own.
Usage: python -m experiments.aggregate [--runs DIR] [--labels CSV] [--output JSON] [--allow-unlabelled]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from experiments import free_tier
from experiments.baseline_rule import RULES, apply_rules
from experiments.runbook_rule import load_thresholds
from experiments.runner import TARGET_DOWN, case_role
from experiments.score import LABELS, RUNS, load_labels, load_manifests, validated_runs, wilson

MODEL_ERRORS = ("agent conclusion could not be parsed", "model response ended at its token limit")
# Provider/API errors of a model call: the openai classes, langchain-openai's wrapped OpenAI* classes, and
# agent.graph.ProviderErrorResponse (an HTTP 200 whose body is an `error`).
PROVIDER_ERROR_TYPES = {"APIStatusError", "APIConnectionError", "APITimeoutError", "RateLimitError",
                        "InternalServerError", "AuthenticationError", "PermissionDeniedError",
                        "NotFoundError", "BadRequestError", "APIError", "ProviderErrorResponse",
                        "OpenAIAuthenticationError", "OpenAIPermissionDeniedError", "OpenAIInvalidRequestError",
                        "OpenAIModelNotFoundError", "OpenAIRateLimitError", "OpenAIAPIError", "OpenAIConnectionError",
                        "OpenAITimeoutError", "OpenAIContextOverflowError", "OpenAIAPIContextOverflowError"}
HTTP_ERROR = re.compile(r"Error code: [45]\d\d\b")
PASS_K = (1, 2, 4)
REGISTERED_COMPARISONS = {
    "localization": ("latency_injected", "latency_enricher"),
    "mechanism": ("latency_injected", "cpu_throttled"),
}
SENSITIVITY_EXCLUDED_CASE = "client_timeout_too_low"


def load_result(runs: Path, manifest: dict) -> dict | None:
    path = runs / manifest["run_id"] / f"{manifest['run_id']}.result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def arm_of(manifest: dict) -> str:
    return (manifest.get("arm") or {}).get("id") or manifest.get("model_id") or "unknown"


def run_cost(manifest: dict, result: dict | None) -> float | None:
    """The larger of the audited cost and the cost the replies reported; None if neither is known."""
    audit = manifest.get("generation_audit") or {}
    known = [float(audit["total_cost_usd"])] if audit.get("retrieved") else []
    if result and result.get("cost_usd_total") is not None:
        known.append(float(result["cost_usd_total"]))
    return max(known) if known else None


def invalid_cause(manifest: dict, result: dict | None) -> str:
    errors = manifest.get("validity_errors") or []
    if any(error in MODEL_ERRORS for error in errors):
        return "model"
    if any(error.startswith(TARGET_DOWN) for error in errors):
        return "target_down"
    if result and result.get("error_type") == "ZeroCostViolation":
        return "cost_guard"
    if free_tier.availability_failure(manifest, result):
        return "availability"
    if any(error.startswith(("provider drift", "provider audit")) for error in errors):
        return "provider"
    if result and result.get("status") == "crashed" and (result.get("error_type") in PROVIDER_ERROR_TYPES
                                                          or HTTP_ERROR.search(result.get("traceback_tail") or "")):
        return "provider"
    return "infrastructure"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for discordant pairs b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]] (sum of tables no more likely)."""
    row1, row2, col1, n = a + b, c + d, a + c, a + b + c + d
    if n == 0:
        return 1.0

    def probability(x):
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = probability(a)
    low, high = max(0, col1 - row2), min(row1, col1)
    total = sum(p for x in range(low, high + 1) if (p := probability(x)) <= observed * (1 + 1e-9))
    return round(min(1.0, total), 4)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (family = the keys)."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted, running, m = {}, 0.0, len(ordered)
    for index, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - index) * p))
        adjusted[name] = round(running, 4)
    return adjusted


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 6) if values else None


def pass_hat_k(correct: int, n: int, k: int) -> float | None:
    """Unbiased pass^k for one cell: all k of k i.i.d. trials correct."""
    return math.comb(correct, k) / math.comb(n, k) if n >= k else None


class ArmData:
    """Everything known about one arm's runs; metric blocks are computed on subsets of its valid runs."""

    def __init__(self, runs: Path, manifests: list[dict], labels: dict, valid_ids: set[str],
                 machine_valid_ids: set[str], thresholds: dict):
        self.manifests = manifests
        self.results = {m["run_id"]: load_result(runs, m) for m in manifests}
        self.labels = labels
        self.valid = [m for m in manifests if m["run_id"] in valid_ids]
        self.machine_valid_ids = machine_valid_ids
        self.rules = {m["run_id"]: apply_rules(runs / m["run_id"], m, thresholds) for m in self.valid}
        self.costs = {m["run_id"]: run_cost(m, self.results[m["run_id"]]) for m in manifests}
        self.causes = {m["run_id"]: invalid_cause(m, self.results[m["run_id"]])
                       for m in manifests if m["run_id"] not in machine_valid_ids}

    def label(self, manifest: dict, field: str = "diagnosis_label") -> str:
        return self.labels[manifest["run_id"]].get(field, "")

    def correct(self, manifest: dict) -> bool:
        return self.label(manifest) == "correct"

    def model_invalid(self, cases: set[str]) -> int:
        return sum(1 for m in self.manifests if m["case"] in cases and self.causes.get(m["run_id"]) == "model")

    def block(self, runs: list[dict]) -> dict:
        """Every per-run metric on one subset of valid runs (one fault, a role, or all)."""
        cases = {m["case"] for m in runs}
        faults = [m for m in runs if m["case"] != "healthy"]
        healthy = [m for m in runs if m["case"] == "healthy"]
        correct = sum(self.correct(m) for m in faults)
        answered = [m for m in faults if self.label(m) != "abstain"]
        out = {
            "n_valid": len(runs),
            "strict_accuracy": rate(correct, len(faults)),
            "strict_accuracy_itt": rate(correct, len(faults) + self.model_invalid(cases - {"healthy"})),
            "localized": rate(sum(self.label(m, "localized") == "yes" for m in faults), len(faults)),
            "detected": rate(sum(self.label(m, "detected") == "yes" for m in faults), len(faults)),
            "abstain": rate(sum(self.label(m) == "abstain" for m in runs), len(runs)),
            "selective_accuracy": rate(sum(self.correct(m) for m in answered), len(answered)),
            "coverage": rate(len(answered), len(faults)),
            "healthy_false_alarm": rate(sum(self.label(m) == "wrong" for m in healthy), len(healthy)),
            "evidence_labels": dict(sorted(Counter(self.label(m, "evidence_label") for m in runs).items())),
            "confidence": self.confidence(runs),
            "rules": self.rule_block(runs),
            "manipulation_check": self.manipulation(runs),
            "pass_k": self.pass_k(faults),
            "cost_usd": self.cost(runs),
        }
        return out

    def confidence(self, runs: list[dict]) -> dict:
        """Q2b and Q4: P(correct | confirmed) under each policy and the downgrade rule both ways."""
        result = lambda m: self.results[m["run_id"]] or {}
        enforced = [m for m in runs if result(m).get("confidence") == "confirmed"]
        self_reported = [m for m in runs if result(m).get("model_self_reported_confidence") == "confirmed"]
        downgraded = [m for m in self_reported if result(m).get("confidence") != "confirmed"]
        by_level = {level: rate(sum(self.correct(m) for m in at), len(at)) for level in
                    ("confirmed", "best_guess", "inconclusive")
                    for at in [[m for m in runs if result(m).get("confidence") == level]]}
        return {
            "p_correct_given_confirmed_enforced": rate(sum(self.correct(m) for m in enforced), len(enforced)),
            "p_correct_given_confirmed_self_reported": rate(sum(self.correct(m) for m in self_reported),
                                                            len(self_reported)),
            "downgraded": len(downgraded),
            "wrong_confirmations_caught": sum(not self.correct(m) for m in downgraded),
            "correct_confirmations_downgraded": sum(self.correct(m) for m in downgraded),
            "confirmed_but_wrong_after_gate": sum(not self.correct(m) for m in enforced),
            "by_enforced_level": by_level,
        }

    def rule_block(self, runs: list[dict]) -> dict:
        """Each rule's strict and localization accuracy, and paired McNemar against the agent."""
        out = {}
        for name in RULES:
            usable = [m for m in runs if self.rules[m["run_id"]][f"{name}_correct"] != ""]
            faults = [m for m in usable if m["case"] != "healthy"]
            healthy = [m for m in usable if m["case"] == "healthy"]
            rule = lambda m, part: self.rules[m["run_id"]][f"{name}_{part}"] is True
            record = {"usable": len(usable),
                      "strict_accuracy": rate(sum(rule(m, "correct") for m in faults), len(faults)),
                      "localized": rate(sum(rule(m, "localized") for m in faults), len(faults)),
                      "healthy_false_alarm": rate(sum(not rule(m, "correct") for m in healthy), len(healthy))}
            for level, agent in (("strict", self.correct), ("localized",
                                                             lambda m: self.label(m, "localized") == "yes")):
                part = "correct" if level == "strict" else "localized"
                b = sum(agent(m) and not rule(m, part) for m in faults)
                c = sum(rule(m, part) and not agent(m) for m in faults)
                record[f"paired_{level}"] = {"agent_right_rule_wrong": b, "rule_right_agent_wrong": c,
                                             "mcnemar_exact_p": mcnemar_exact(b, c)}
            out[name] = record
        return out

    def manipulation(self, runs: list[dict]) -> dict:
        checks = [(m.get("manipulation_check") or {}).get("passed") for m in runs]
        known = [c for c in checks if c is not None]
        return {"passed": rate(sum(known), len(known)), "unknown": len(checks) - len(known)}

    def pass_k(self, faults: list[dict]) -> dict:
        cells = defaultdict(list)
        for m in faults:
            cells[m["case"]].append(self.correct(m))
        out = {}
        for k in PASS_K:
            values = [v for outcomes in cells.values() if (v := pass_hat_k(sum(outcomes), len(outcomes), k)) is not None]
            out[f"pass^{k}"] = round(sum(values) / len(values), 4) if values else None
        modal = [max(Counter(o).values()) / len(o) for o in cells.values() if o]
        out["consistency"] = round(sum(modal) / len(modal), 4) if modal else None
        return out

    def cost(self, runs: list[dict]) -> dict:
        spend = sum(self.costs[m["run_id"]] or 0 for m in runs)
        correct = sum(self.correct(m) for m in runs)
        known = [self.costs[m["run_id"]] for m in runs if self.costs[m["run_id"]] is not None]
        return {"spend_valid": round(spend, 6), "median_per_run": median(known),
                "per_correct_diagnosis": round(spend / correct, 6) if correct else None}


def summarize_arm(data: ArmData) -> dict:
    valid = data.valid
    by_role = {role: [m for m in valid if case_role(m["case"]) == role] for role in ("hidden", "visible", "control")}
    hidden = by_role["hidden"]
    causes = Counter(data.causes.values())
    for m in data.manifests:
        if m["run_id"] in data.machine_valid_ids and m not in valid:
            causes["human review" if m["run_id"] in data.labels else "awaiting label"] += 1
    per_fault = {case: data.block([m for m in valid if m["case"] == case])
                 for case in sorted({m["case"] for m in valid})}
    comparisons = {}
    for name, (first, second) in REGISTERED_COMPARISONS.items():
        a, b = per_fault.get(first), per_fault.get(second)
        record = {"cases": [first, second]}
        for metric in ("strict_accuracy", "localized"):
            if a and b:
                ka, na, kb, nb = a[metric]["k"], a[metric]["n"], b[metric]["k"], b[metric]["n"]
                record[metric] = {first: f"{ka}/{na}", second: f"{kb}/{nb}",
                                  "fisher_exact_p": fisher_exact(ka, na - ka, kb, nb - kb)}
        comparisons[name] = record
    results = [data.results[m["run_id"]] for m in valid if data.results[m["run_id"]]]
    without_case = [m for m in hidden if m["case"] != SENSITIVITY_EXCLUDED_CASE]
    with_symptom = [m for m in hidden if (m.get("manipulation_check") or {}).get("passed") is not False]
    return {
        "attempted": len(data.manifests), "machine_valid": len(data.machine_valid_ids & {m["run_id"] for m in data.manifests}),
        "human_valid": len(valid), "invalid_by_cause": dict(sorted(causes.items())),
        "availability_failures": causes.get("availability", 0),
        "hidden": data.block(hidden), "visible": data.block(by_role["visible"]),
        "healthy": data.block(by_role["control"]), "all": data.block(valid),
        "per_fault": per_fault,
        "registered_comparisons": comparisons,
        "sensitivity": {f"hidden_without_{SENSITIVITY_EXCLUDED_CASE}": data.block(without_case),
                        "hidden_without_failed_manipulation_check": data.block(with_symptom)},
        "median_calls_used": median([r["calls_used"] for r in results if r.get("calls_used") is not None]),
        "median_agent_wall_clock_s": median([r["wall_clock_s"] for r in results if r.get("wall_clock_s") is not None]),
        "tokens": {key: sum(r.get(key) or 0 for r in results)
                   for key in ("input_tokens_total", "output_tokens_total", "reasoning_tokens_total")},
        "cost_usd_all_attempts": round(sum(c or 0 for c in data.costs.values()), 6),
        "attempts_without_cost": sum(c is None for c in data.costs.values()),
        "tool_schema_sha256": sorted({str(m.get("agent_tool_schema_sha256")) for m in valid}),
    }


def kappa_block(datas: dict[str, ArmData]) -> dict:
    from experiments.auto_label import auto_label, cohen_kappa

    pairs, per_arm = [], {}
    for arm, data in datas.items():
        arm_pairs = [(data.label(m), auto) for m in data.valid
                     if (auto := auto_label(m["case"], data.results[m["run_id"]])) != "unrated"]
        per_arm[arm] = {"compared": len(arm_pairs), "cohen_kappa": cohen_kappa(arm_pairs)}
        pairs.extend(arm_pairs)
    return {"note": "auto-labeller is a second rater, never a score", "compared": len(pairs),
            "cohen_kappa": cohen_kappa(pairs), "per_arm": per_arm}


def aggregate(runs: Path, labels: dict, allow_unlabelled: bool = False, thresholds: dict | None = None) -> dict:
    manifests = load_manifests(runs)
    machine_valid, valid = validated_runs(manifests, labels, allow_unlabelled, require_levels=True)
    valid_ids, machine_valid_ids = {m["run_id"] for m in valid}, {m["run_id"] for m in machine_valid}
    thresholds = thresholds or load_thresholds()
    grouped = defaultdict(list)
    for manifest in manifests:
        grouped[arm_of(manifest)].append(manifest)
    datas = {arm: ArmData(runs, items, labels, valid_ids, machine_valid_ids, thresholds)
             for arm, items in sorted(grouped.items())}
    arms = {arm: summarize_arm(data) for arm, data in datas.items()}
    halt = free_tier.read_halt(runs)
    # A permanently halted free arm leaves the Holm family but stays in the report.
    family = [arm for arm in arms if not (halt and halt.get("arm") == arm)]
    for name in RULES:
        for level in ("strict", "localized"):
            adjusted = holm({arm: arms[arm]["hidden"]["rules"][name][f"paired_{level}"]["mcnemar_exact_p"]
                             for arm in family})
            for arm in arms:
                arms[arm]["hidden"]["rules"][name][f"paired_{level}"]["holm_p"] = adjusted.get(arm)
    pooled = lambda members: rate(sum(arms[a]["healthy"]["healthy_false_alarm"]["k"] for a in members),
                                  sum(arms[a]["healthy"]["healthy_false_alarm"]["n"] for a in members))
    hidden = {arm: arms[arm]["hidden"]["strict_accuracy"] for arm in family}
    arm_vs_arm = {f"{a} vs {b}": {a: f"{hidden[a]['k']}/{hidden[a]['n']}", b: f"{hidden[b]['k']}/{hidden[b]['n']}",
                                  "fisher_exact_p": fisher_exact(hidden[a]["k"], hidden[a]["n"] - hidden[a]["k"],
                                                                 hidden[b]["k"], hidden[b]["n"] - hidden[b]["k"])}
                  for a, b in combinations(sorted(family), 2)}
    return {
        "schema_version": "2.0",
        "scoring": "human labels only; per-protocol accuracy is primary, ITT is secondary",
        "caveat": "n is small: arms whose intervals overlap cannot be ranked; per-fault lines next to every pooled value",
        "r3_thresholds_status": thresholds.get("status"),
        "headline_family": {"arms": family, "holm_tests_per_rule": len(family),
                            "free_arm_halt": halt,
                            "note": ("free arm halted: family reduced to the remaining arms" if halt
                                     else "all arms in the data")},
        "pooled_healthy_false_alarm": pooled(family),
        "pooled_healthy_false_alarm_all_arms": pooled(list(arms)),
        "arm_vs_arm_hidden_strict": arm_vs_arm,
        "auto_label_agreement": kappa_block(datas),
        "unlabelled_machine_valid": sorted(machine_valid_ids - set(labels)),
        "arms": arms,
    }


def table(report: dict) -> str:
    def fmt(r):
        return f"{r['k']}/{r['n']}" + (f" [{r['wilson95'][0]:.2f},{r['wilson95'][1]:.2f}]" if r["wilson95"] else "")

    lines = ["arm | attempted/valid | hidden strict | hidden localized | healthy FA | visible strict | "
             "R1/R2/R3 hidden strict | $/correct | availability failures"]
    for arm, s in report["arms"].items():
        rules = "/".join(f"{s['hidden']['rules'][r]['strict_accuracy']['k']}" for r in ("r1", "r2", "r3"))
        lines.append(f"{arm} | {s['attempted']}/{s['human_valid']} | {fmt(s['hidden']['strict_accuracy'])} | "
                     f"{fmt(s['hidden']['localized'])} | {fmt(s['healthy']['healthy_false_alarm'])} | "
                     f"{fmt(s['visible']['strict_accuracy'])} | {rules} of {s['hidden']['strict_accuracy']['n']} | "
                     f"{s['all']['cost_usd']['per_correct_diagnosis']} | {s['availability_failures']}")
    lines.append(f"pooled healthy FA (family {report['headline_family']['arms']}): "
                 f"{fmt(report['pooled_healthy_false_alarm'])}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--output", type=Path, help="JSON file (default: stdout)")
    parser.add_argument("--allow-unlabelled", action="store_true",
                        help="progress view: leave unlabelled completed runs out instead of failing")
    args = parser.parse_args()
    report = aggregate(args.runs, load_labels(args.labels), args.allow_unlabelled)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(table(report))
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
