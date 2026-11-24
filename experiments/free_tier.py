"""Guards for a free (":free", $0) arm: configuration check, daily request budget, permanent halt and pause.

Any cost above zero, an unaccepted served model or an unverifiable $0 halts the arm; a 429 only pauses it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "experiments" / "state" / "free_tier.json"
HALT_FILE = "free_arm_halt.json"
UNAVAILABLE_AFTER_DAYS = 3
AVAILABILITY = re.compile(r"Error code: (429|500|502|503|504)\b|RateLimitError|rate[- ]?limit|InternalServerError|"
                          r"APIConnectionError|APITimeoutError|overloaded|temporarily unavailable|"
                          r"provider returned error", re.IGNORECASE)
GONE = re.compile(r"No endpoints found|not a valid model|model .{0,80}(does not exist|not found)|Error code: 404\b",
                  re.IGNORECASE)
DAILY_LIMIT = re.compile(r"per[- ]day|free-models-per-day|daily", re.IGNORECASE)
HOLM_NOTE = ("the free arm is halted permanently: the benchmark continues with the paid arms only, the headline "
             "Holm family becomes those arms (3 arms: at least 7 discordant pairs at alpha .05/3), and the free "
             "arm's completed runs are reported descriptively")


def utc_day(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()


def is_free(arm: dict | None) -> bool:
    return bool((arm or {}).get("free"))


# --- configuration --------------------------------------------------------

def config_problems(arm: dict) -> list[str]:
    """Why this free arm could cost money; [] when every guard is in place."""
    problems = []
    if not arm["model_id"].endswith(":free"):
        problems.append(f"model id {arm['model_id']!r} is not a ':free' variant")
    price = arm.get("price_usd_per_million") or {}
    if not price or any(float(value) != 0 for value in price.values()):
        problems.append(f"price {price} is not zero")
    if float(arm.get("cap_usd", 1)) != 0:
        problems.append(f"cap ${arm.get('cap_usd')} is not zero")
    if arm.get("allow_fallbacks"):
        problems.append("allow_fallbacks must be false (no paid fallback)")
    if arm.get("max_price") != {"prompt": 0, "completion": 0}:
        problems.append(f"max_price {arm.get('max_price')} must be prompt 0, completion 0")
    if not arm.get("require_zero_cost"):
        problems.append("require_zero_cost (the agent's zero-cost guard) must be on")
    tier = arm.get("free_tier") or {}
    if not tier.get("requests_per_day") or not tier.get("worst_case_requests_per_run"):
        problems.append("free_tier needs requests_per_day and worst_case_requests_per_run")
    return problems


# --- what a finished run shows --------------------------------------------

def failure_text(manifest: dict, result: dict | None, stderr: str = "") -> str:
    return "\n".join([*(manifest.get("validity_errors") or []), (result or {}).get("error_type") or "",
                      (result or {}).get("traceback_tail") or "", stderr])


def availability_failure(manifest: dict, result: dict | None, stderr: str = "") -> str | None:
    """'gone' (permanent), 'daily_limit', 'rate_limit_or_unavailable', or None; free arms only."""
    if not is_free(manifest.get("arm")) or (manifest.get("status") == "completed" and not manifest.get("validity_errors")):
        return None
    if result and result.get("status") == "completed":
        return None  # the agent finished; any invalidity is not about availability
    text = failure_text(manifest, result, stderr)
    if GONE.search(text):
        return "gone"
    if AVAILABILITY.search(text):
        return "daily_limit" if DAILY_LIMIT.search(text) else "rate_limit_or_unavailable"
    return None


def served_variant(model: str, accepted_models: list[str]) -> bool:
    """An accepted ':free' id's base id or a dated variant of it ('<base>-YYYYMMDD', with or without ':free')."""
    return any(re.fullmatch(re.escape(accepted.removesuffix(":free")) + r"(-\d{8})?(:free)?", model)
               for accepted in accepted_models)


def cost_violation(manifest: dict, result: dict | None, accepted_models: list[str]) -> str | None:
    """The reason a free arm must halt after this run, or None."""
    result = result or {}
    audit = manifest.get("generation_audit") or {}
    if result.get("error_type") == "ZeroCostViolation":
        return "the agent's zero-cost guard stopped the run: a reply reported a cost above zero"
    if (result.get("cost_usd_total") or 0) > 0:
        return f"replies reported cost ${result['cost_usd_total']}"
    if audit.get("retrieved") and float(audit.get("total_cost_usd") or 0) > 0:
        return f"generation audit reports cost ${audit['total_cost_usd']}"
    served = set(result.get("served_models") or []) | set(audit.get("models_served") or [])
    outside = sorted(served - set(accepted_models))
    unknown = [model for model in outside if not served_variant(model, accepted_models)]
    if unknown:
        return f"served model(s) {unknown} are not the accepted free variant {accepted_models}"
    if outside:
        # The base or dated id is accepted only with a verified $0 on every call and every generation.
        gaps = []
        if not (result.get("model_settings") or {}).get("require_zero_cost"):
            gaps.append("the zero-cost guard was not on")
        if result.get("cost_usd_total") != 0 or result.get("cost_reported_calls") != result.get("llm_calls"):
            gaps.append("not every call reported a cost of exactly 0")
        if not (audit.get("generations") and audit.get("retrieved") == audit.get("generations")
                and not audit.get("missing") and float(audit.get("total_cost_usd") or 0) == 0):
            gaps.append("the generation audit did not confirm $0 for every generation")
        if gaps:
            return (f"served model(s) {outside} are accepted only with a verified $0, but "
                    + "; ".join(gaps))
    if (result.get("llm_calls") or 0) > 0 and result.get("cost_usd_total") is None and not audit.get("retrieved"):
        return "a $0 cost cannot be verified: no reply reported a cost and the generation audit retrieved nothing"
    return None


def requests_used(result: dict | None, worst_case: int) -> int:
    """Measured requests of a run; a run with no result keeps its worst-case reservation."""
    if not result:
        return worst_case
    used = int(result.get("llm_attempts") or 0)
    if result.get("status") != "completed":
        retries = (result.get("model_settings") or {}).get("max_retries")
        retries = 2 if retries is None else int(retries)  # unrecorded: assume the openai client default
        used += 1 + retries  # the call that failed, with the client's hidden retries
    return used


def estimate_days(pending_runs: int, requests_per_day: int, worst_case: int) -> tuple[int, int | None]:
    """(runs per UTC day, days needed) for the free arm's pending runs."""
    per_day = requests_per_day // worst_case if worst_case else 0
    if not pending_runs:
        return per_day, 0
    return per_day, (-(-pending_runs // per_day) if per_day else None)


# --- local ledger -----------------------------------------------------------

class Ledger:
    """Requests per arm and UTC day, availability days and permanent halts (experiments/state/free_tier.json)."""

    def __init__(self, path: Path = LEDGER):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"arms": {}}

    def arm(self, arm_id: str) -> dict:
        return self.data["arms"].setdefault(arm_id, {"days": {}, "availability_days": [], "halted": None})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def day(self, arm_id: str, day: str) -> dict:
        return self.arm(arm_id)["days"].setdefault(day, {"requests": {}, "exhausted": False})

    def used(self, arm_id: str, day: str) -> int:
        return sum(self.day(arm_id, day)["requests"].values())

    def remaining(self, arm_id: str, day: str, limit: int) -> int:
        record = self.day(arm_id, day)
        return 0 if record["exhausted"] else max(0, limit - self.used(arm_id, day))

    def record(self, arm_id: str, day: str, key: str, requests: int) -> None:
        self.day(arm_id, day)["requests"][key] = requests
        self.save()

    def settle(self, arm_id: str, day: str, reservation: str, key: str, requests: int) -> None:
        self.day(arm_id, day)["requests"].pop(reservation, None)
        self.record(arm_id, day, key, requests)

    def mark_exhausted(self, arm_id: str, day: str) -> None:
        self.day(arm_id, day)["exhausted"] = True
        self.save()

    def note_availability(self, arm_id: str, day: str) -> int:
        """Distinct UTC days with an availability failure since the last completed run."""
        days = self.arm(arm_id)["availability_days"]
        if day not in days:
            days.append(day)
        self.save()
        return len(days)

    def note_completed(self, arm_id: str) -> None:
        self.arm(arm_id)["availability_days"] = []
        self.save()

    def halt(self, arm_id: str, reason: str, run_id: str | None) -> dict:
        record = {"arm": arm_id, "reason": reason, "run_id": run_id,
                  "at_utc": datetime.now(timezone.utc).isoformat(), "consequence": HOLM_NOTE}
        self.arm(arm_id)["halted"] = record
        self.save()
        return record

    def halted(self, arm_id: str) -> dict | None:
        return self.arm(arm_id)["halted"]


def write_halt(runs_dir: Path, record: dict) -> Path:
    """The recorded decision next to the runs, read by experiments.aggregate."""
    path = runs_dir / HALT_FILE
    runs_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_halt(runs_dir: Path) -> dict | None:
    path = runs_dir / HALT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
