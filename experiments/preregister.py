"""Pre-registration in experiments/prereg/: seeded run schedule, frozen protocol hashes, OpenRouter catalog snapshot.

The batch refuses a matrix run whose protocol files no longer match prereg.json.
Usage: python -m experiments.preregister {schedule,freeze,catalog} [--tool-schema-sha256 HEX]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from experiments.runner import CASES, ROOT, load_fault_settings

PREREG_DIR = ROOT / "experiments" / "prereg"
ARMS = PREREG_DIR / "arms.json"
SCHEDULE = PREREG_DIR / "schedule.csv"
PREREG = PREREG_DIR / "prereg.json"
CATALOG = PREREG_DIR / "catalog_snapshot.json"
SCHEDULE_FIELDS = ("row", "phase", "block", "arm", "case", "rep")

# Files whose change alters what is measured; the batch checks these hashes.
PROTOCOL_FILES = (
    "agent/system_prompt.md", "agent/graph.py", "agent/tools.py", "agent/config.py", "agent/run.py",
    "agent/report.py", "mcp-server/server.py", "experiments/runner.py", "experiments/openrouter_audit.py",
    "experiments/stack.py", "experiments/bench_stack.json", "docker-compose.yaml", "prometheus/prometheus.yml",
    "fault-injection/common.sh", "experiments/prereg/arms.json", "experiments/prereg/schedule.csv",
    "experiments/symptoms.py", "experiments/fault_settings.json",
)
# Analysis and driver code and the analysis plan, recorded so later changes are visible; not enforced by the batch.
ANALYSIS_FILES = ("experiments/baseline_rule.py", "experiments/runbook_rule.py", "experiments/auto_label.py",
                  "experiments/aggregate.py", "experiments/score.py", "experiments/label_sheet.py",
                  "experiments/batch.py", "experiments/free_tier.py", "experiments/validate_faults.py",
                  "experiments/observability_probe.py", "experiments/prereg/r3_thresholds.json",
                  "experiments/prereg/analysis_plan.json")
# Pre-registered strict "correct" rubric per case; shown on the label sheet.
RUBRIC = {
    "healthy": "Any asserted fault is wrong (false alarm); 'no supported issue' or an abstention is correct.",
    "latency_injected": "Names go-server AND network delay/latency on it. Blaming Redis or Kafka as the cause is wrong.",
    "latency_enricher": "Names go-enricher AND network delay/latency on it. Blaming Redis, Kafka or go-server is wrong.",
    "client_timeout_too_low": "Names go-server, its Redis client timeout setting(s) AND that the value is too low.",
    "cpu_throttled": "Names go-server AND its CPU limit/quota or CPU starvation.",
    "redis_node_stopped": "Names redis2 (not just 'a redis node') AND that it is stopped/exited.",
    "ingestion_paused": "Names go-server AND that its container is paused/frozen.",
    "broker_stopped": "Names kafka AND that the broker is stopped/exited.",
    "enricher_paused": "Names go-enricher AND that its container is paused/frozen.",
}
RULES = {
    "primary": ["Q1 strict hidden accuracy = human 'correct' / human-valid hidden-fault runs per arm, Wilson 95%, "
                "with per-fault k/n; paired exact McNemar against rules R1, R2 and R3 on the same runs, "
                "Holm-corrected over the headline arm family",
                "Q2 healthy false-alarm rate = human 'wrong' / human-valid healthy runs, pooled over the headline "
                "arms, Wilson 95% (per-arm values descriptive only)"],
    "headline_family": "the 4 enabled arms (luna, glm, gemma and the free nemotron arm): Holm over 4 tests per rule, "
                       "so an arm needs at least 8 discordant pairs (alpha .05/4). If the free arm halts "
                       "permanently, the family becomes the 3 paid arms (at least 7 pairs) and the decision is "
                       "recorded in <runs>/free_arm_halt.json",
    "secondary": ["localization and detection accuracy (labels localized, detected)",
                  "ITT hidden accuracy: runs invalid because of the model (unparseable or token-limited conclusion) "
                  "count as wrong; infrastructure, provider and availability invalids stay excluded",
                  "P(correct | confirmed) under enforced and self-reported confidence (Q2b, Q4); the downgrade "
                  "rule reported both ways: wrong confirmations caught and correct confirmations downgraded",
                  "selective accuracy with coverage", "cost per correct diagnosis (Q3), Fisher exact arm vs arm",
                  "pass^k for k = 1, 2, 4, unbiased estimator C(c,k)/C(n,k) (Q5)",
                  "Cohen's kappa auto-labeller vs blind human labels (Q6); manipulation-check pass share",
                  "registered comparisons: localization (latency_injected vs latency_enricher) and mechanism "
                  "(latency_injected vs cpu_throttled)",
                  "sensitivity: hidden accuracy without client_timeout_too_low; without runs failing the "
                  "manipulation check",
                  "every metric also per fault", "invalid attempts by cause; availability failures separately"],
    "scoring": "human labels in experiments/human_labels.csv (filled through the model-blind label sheet, "
               "experiments.label_sheet) are the only score; the auto-labeller is a second rater reported as "
               "agreement (Cohen's kappa), never a score",
    "rubric": RUBRIC,
    "rule_r1": "first service in zookeeper,kafka,redis1,redis2,redis3,go-server,go-enricher,prometheus whose "
               "state in evidence entry 0 is not 'running' (absent counts); else healthy",
    "rule_r0": "every container whose state in evidence entry 0 is 'exited' or 'paused'; else healthy",
    "rule_r2": "R1; if healthy, the single required service whose entry-0 status is 'Up' < 120 s while every other "
               "required service is older (recently restarted); else healthy",
    "rule_r3": "cause-naming runbook on the model-free runbook snapshots (experiments.runbook_rule): (a) Go service "
               "config differs from the pre-injection snapshot -> service and setting; (b) p50 latency ratio or "
               "error ratio above threshold -> service, then (c) CPU signal (GOMAXPROCS dropped or CPU rate above "
               "threshold) -> cpu, else latency -> network delay, else errors; otherwise abstain. Thresholds come "
               "only from healthy-run data (r3_thresholds.json); scored for localization and strict diagnosis",
    "validity": "runner validity_errors (a running-target fault whose container is not running after injection "
                "is invalid as 'target not running after injection'), plus provider drift and an incomplete "
                "generation audit; invalid attempts stay in the attempted count",
    "requeue": "an invalid attempt not caused by the model is re-run once at the end of the schedule; "
               "interrupted, key-limit and free-arm availability attempts do not count; model-caused invalids "
               "are outcomes, not re-run",
    "stop": "an arm stops before a run when its spend (pre-pilot included) plus its per-run reservation would "
            "exceed its cap; the reservation is the token-model worst case until the arm has a completed paid "
            "run, then max(worst case, 1.5 x its highest observed run cost); no run starts when the recorded "
            "spend of all arms (pre-pilot included) plus its reservation would exceed the global budget "
            "(arms.json spend_guard, $5.25 under a $5.50 key limit); a run whose cost was not recorded (agent "
            "killed at the runner timeout or crashed first) counts at the reservation, or its audited cost if "
            "larger (cost_estimated); an OpenRouter HTTP 401/402/403 or 'Key limit exceeded' stops the batch "
            "and does not use up a re-queue (a 429 does not stop it)",
    "free_arm": "nemotron runs only as ':free' with $0 price, $0 cap, no fallbacks, max_price 0/0 and the agent's "
                "zero-cost guard; data_collection 'allow' for this arm only (user-approved). It halts permanently "
                "if any run reports cost > 0, a served model outside the free variant (its base id and dated "
                "'-YYYYMMDD' variants count as the free variant only with a verified $0), an unverifiable $0, a "
                "permanent unavailability, or availability failures on 3 distinct UTC days without a completed "
                "run. A 429 or transient provider error pauses it (availability failure, not a model failure, no "
                "re-queue used). It starts a run only if the UTC day's request budget (50/day, worst case 16 "
                "requests per run) covers a worst-case run. 'If Nemotron would be paid, don't execute it; use it "
                "only if free; otherwise keep 3 models.'",
    "glm_cost_switch": "if GLM's median cost per run in the pre-pilot exceeds $0.109, before the main phase either "
                       "switch GLM to the novita-fp8 pin (new arm id glm@novita-fp8, $0.63/$1.98) or accept that "
                       "GLM may stop early at its $3.40 cap (arms.json cost_switch.decision 'accept'); the batch "
                       "refuses the main phase until one is recorded. z-ai fp8 stays the default",
    "frozen": "prompt, tools, budget (15 incl. 2 baseline), max_tokens 8192, client max_retries 1 and the arms "
              "are frozen for the matrix; the pre-pilot may change infrastructure only, judged on tokens and "
              "errors, never on diagnoses",
}


def sha256_text(path: Path) -> str:
    """sha256 with CRLF normalized to LF, so a Windows checkout matches a Linux one."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# --- arms -----------------------------------------------------------------

def load_config(path: Path = ARMS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_arm(config: dict, arm: dict) -> dict:
    """One arm with its selected pin and defaults flattened; `id` is key@pin."""
    pin = arm["pins"][arm["pin"]]
    defaults = config.get("defaults", {})
    free = bool(arm.get("free"))
    return {
        "id": f"{arm['key']}@{arm['pin']}", "key": arm["key"], "pin": arm["pin"], "model_id": arm["model_id"],
        "provider_order": pin["provider_order"], "quantizations": pin.get("quantizations", []),
        "price_usd_per_million": pin.get("price_usd_per_million"),
        "allow_fallbacks": arm.get("allow_fallbacks", defaults.get("allow_fallbacks", False)),
        "data_collection": arm.get("data_collection", defaults.get("data_collection")),
        "seed": arm.get("seed", defaults.get("seed")),
        "reasoning_effort": arm.get("reasoning_effort", defaults.get("reasoning_effort")),
        "reasoning": bool(arm.get("reasoning")), "effective_temperature": arm.get("effective_temperature"),
        "cap_usd": float(arm["cap_usd"]),
        "max_price": pin.get("max_price_usd_per_million"), "free": free, "require_zero_cost": free,
        "free_tier": arm.get("free_tier") if free else None, "cost_switch": arm.get("cost_switch"),
    }


def enabled_arms(config: dict) -> dict[str, dict]:
    return {arm["key"]: resolve_arm(config, arm) for arm in config["arms"] if arm.get("enabled")}


def arm_environment(config: dict, arm: dict) -> dict[str, str]:
    """Agent settings for one arm; never contains the API key."""
    # The provider is pinned with fallbacks off so each arm is one endpoint; the generation audit enforces it.
    env = {
        "MODEL_BASE_URL": config["base_url"], "MODEL_ID": arm["model_id"], "MODEL_PROVIDER": "openrouter",
        "ENV_LABEL": arm["id"], "MODEL_PROVIDER_ORDER": ",".join(arm["provider_order"]),
        "MODEL_QUANTIZATIONS": ",".join(arm["quantizations"]),
        "MODEL_ALLOW_FALLBACKS": "true" if arm["allow_fallbacks"] else "false",
        "MODEL_DATA_COLLECTION": arm["data_collection"] or "",
        "MODEL_SEED": "" if arm["seed"] is None else str(arm["seed"]),
        "MODEL_REASONING_EFFORT": arm["reasoning_effort"] or "",
        "MODEL_MAX_PRICE": (f"{arm['max_price']['prompt']},{arm['max_price']['completion']}"
                            if arm.get("max_price") else ""),
        "MODEL_REQUIRE_ZERO_COST": "true" if arm.get("require_zero_cost") else "false",
    }
    return env


def estimate_run_usd(arm: dict, token_model: dict, scenario: str) -> float | None:
    """Estimated cost of one run; None when the pin has no price."""
    price = arm["price_usd_per_million"]
    if not price:
        return None
    tokens = token_model[scenario]
    output = tokens["output"] + (tokens["reasoning"] if arm["reasoning"] else 0)
    return (tokens["input"] * price["input"] + output * price["output"]) / 1e6


# --- schedule -------------------------------------------------------------

def phase_blocks(arm_keys: list[str], case_reps: dict[str, int]) -> list[list[tuple[str, str]]]:
    """Blocks for one phase: every arm x case once per block; extra repetitions round robin."""
    count = min(case_reps.values())
    blocks = [[(arm, case) for arm in arm_keys for case in case_reps] for _ in range(count)]
    for arm in arm_keys:
        for case, reps in case_reps.items():
            for extra in range(reps - count):
                blocks[extra % count].append((arm, case))
    return blocks


def build_schedule(paid_keys: list[str], free_keys: list[str], cases: dict[str, dict], seed: int) -> list[dict]:
    """Rows for every phase: the paid arms' blocks first, then each free arm's own blocks."""
    # Every block holds each arm x case once, shuffled, so stopping after any block leaves a balanced dataset.
    unknown = [case for case in cases if case not in CASES]
    if unknown:
        raise ValueError(f"matrix cases must be registered cases: {unknown}")
    phases = sorted({spec["phase"] for spec in cases.values()})
    tracks = [paid_keys] + [[key] for key in free_keys]
    rng = random.Random(seed)
    rows, reps, number = [], {}, 0
    for keys in tracks:
        if not keys:
            continue
        for phase in phases:
            case_reps = {case: int(spec["reps"]) for case, spec in cases.items() if spec["phase"] == phase}
            for block in phase_blocks(keys, case_reps):
                number += 1
                rng.shuffle(block)
                for arm, case in block:
                    reps[(arm, case)] = reps.get((arm, case), 0) + 1
                    rows.append({"row": f"r{len(rows) + 1:03d}", "phase": phase, "block": number, "arm": arm,
                                 "case": case, "rep": reps[(arm, case)]})
    return rows


def matrix_schedule(config: dict) -> list[dict]:
    arms = enabled_arms(config)
    paid = [key for key, arm in arms.items() if not arm["free"]]
    free = [key for key, arm in arms.items() if arm["free"]]
    return build_schedule(paid, free, config["matrix"]["cases"], config["matrix"]["seed"])


def write_schedule(rows: list[dict], path: Path = SCHEDULE) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SCHEDULE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_schedule(path: Path = SCHEDULE) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return [{**row, "block": int(row["block"]), "rep": int(row["rep"])} for row in csv.DictReader(file)]


def case_phase(config: dict, case: str) -> str | None:
    return (config["matrix"]["cases"].get(case) or {}).get("phase")


# --- freeze ---------------------------------------------------------------

def protocol_hashes() -> dict[str, str]:
    scripts = sorted(f"fault-injection/{case.script}" for case in CASES.values() if case.script)
    return {path: sha256_text(ROOT / path) for path in (*PROTOCOL_FILES, *scripts)}


def freeze(tool_schema_sha256: str | None = None) -> dict:
    config = load_config()
    return {
        "schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hash_note": "sha256 of each file with CRLF normalized to LF",
        "matrix": config["matrix"], "arms": enabled_arms(config),
        "schedule_rows": len(read_schedule()),
        "fault_settings": load_fault_settings(),
        "protocol_sha256": protocol_hashes(),
        "analysis_sha256": {path: sha256_text(ROOT / path) for path in ANALYSIS_FILES},
        # From a pre-pilot result's tool_schema_sha256; every scored run must report the same value.
        "tool_schema_sha256": tool_schema_sha256,
        "rules": RULES,
    }


def protocol_mismatches(prereg: dict) -> list[str]:
    current = protocol_hashes()
    frozen = prereg.get("protocol_sha256", {})
    return sorted(path for path in set(current) | set(frozen) if current.get(path) != frozen.get(path))


# --- catalog --------------------------------------------------------------

def catalog_snapshot(config: dict, opener=urllib.request.urlopen) -> dict:
    """Public endpoint data per enabled arm model (prices, quantization, supported parameters)."""
    models = {}
    for arm in enabled_arms(config).values():
        url = f"{config['base_url']}/models/{arm['model_id']}/endpoints"
        with opener(urllib.request.Request(url), timeout=30) as response:
            models[arm["model_id"]] = json.load(response).get("data")
    return {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "source": "GET /api/v1/models/{id}/endpoints",
            "models": models}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("schedule", "freeze", "catalog"))
    parser.add_argument("--tool-schema-sha256", help="freeze: the pre-pilot result's tool_schema_sha256")
    args = parser.parse_args()
    if args.action == "schedule":
        rows = matrix_schedule(load_config())
        write_schedule(rows)
        print(f"{len(rows)} rows -> {SCHEDULE}")
    elif args.action == "freeze":
        PREREG.write_text(json.dumps(freeze(args.tool_schema_sha256), indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        print(f"frozen -> {PREREG}")
    else:
        CATALOG.write_text(json.dumps(catalog_snapshot(load_config()), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
        print(f"catalog -> {CATALOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
