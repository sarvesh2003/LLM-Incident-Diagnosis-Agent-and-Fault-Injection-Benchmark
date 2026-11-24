"""Per-run audit of OpenRouter generation records: served provider, model, cost and reasoning tokens.

The chat reply does not name its provider, so a pinned run is invalid if any generation was served elsewhere
("provider drift") or cannot be retrieved. The API key goes only in the Authorization header, never into a log.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ATTEMPTS = 5           # a generation can take a few seconds to become queryable
RETRY_SECONDS = 2.0    # linear backoff: at most 20 s of waiting per missing generation
KEY_FAILURE_CODES = {401, 402, 403}  # bad key, credits, or the key's own limit: stop, never retry


class KeyFailure(RuntimeError):
    """OpenRouter refused the key (HTTP 401/402/403); the batch stops on this."""


def generation_ids(trace: list[dict]) -> list[str]:
    return [gid for event in trace if event.get("event") in ("action_selected", "conclusion")
            for gid in event.get("generation_ids", [])]


def fetch_generation(base_url: str, api_key: str, generation_id: str, *, opener=urllib.request.urlopen,
                     sleep=time.sleep, attempts: int = ATTEMPTS) -> dict | None:
    """The generation record, or None if it was still unavailable after all attempts."""
    url = f"{base_url.rstrip('/')}/generation?" + urllib.parse.urlencode({"id": generation_id})
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    for attempt in range(attempts):
        try:
            with opener(request, timeout=20) as response:
                return json.load(response).get("data")
        except urllib.error.HTTPError as error:
            error.close()
            if error.code in KEY_FAILURE_CODES:
                raise KeyFailure(f"generation lookup refused: HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass
        if attempt + 1 < attempts:
            sleep(RETRY_SECONDS * (attempt + 1))
    return None


def normalize(name: str | None) -> str:
    """'Z.AI' and 'z-ai' both become 'zai'; quantization suffixes ('novita/bf16') are dropped."""
    return re.sub(r"[^a-z0-9]", "", (name or "").split("/")[0].lower())


def drift_errors(records: list[dict], provider_order: list[str] | tuple[str, ...]) -> list[str]:
    pinned = {normalize(slug) for slug in provider_order}
    errors = []
    for record in records:
        if normalize(record.get("provider_name")) not in pinned:
            errors.append(f"provider drift: generation {record.get('id')} served by "
                          f"{record.get('provider_name')!r}, pinned {list(provider_order)}")
    return errors


def audit_run(run_dir: Path, run_id: str, base_url: str, api_key: str,
              provider_order: list[str] | tuple[str, ...], **fetch_options) -> tuple[dict, list[str]]:
    """Look up every generation of one run, write generations.json; returns (summary, validity errors)."""
    trace_path = run_dir / f"{run_id}.trace.jsonl"
    trace = ([json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
             if trace_path.exists() else [])
    ids = generation_ids(trace)
    records, missing = [], []
    for generation_id in ids:
        record = fetch_generation(base_url, api_key, generation_id, **fetch_options)
        if record is None:
            missing.append(generation_id)
        else:
            records.append(record)
    summary = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "pinned_provider_order": list(provider_order),
        "generations": len(ids), "retrieved": len(records), "missing": missing,
        "providers_served": sorted({str(r.get("provider_name")) for r in records}),
        "models_served": sorted({str(r.get("model")) for r in records}),
        "total_cost_usd": sum(float(r.get("total_cost") or 0) for r in records),
        "native_tokens_reasoning": sum(int(r.get("native_tokens_reasoning") or 0) for r in records),
    }
    (run_dir / "generations.json").write_text(
        json.dumps({"run_id": run_id, **summary, "records": records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    # With no pinned provider (a free arm) there is no drift rule; the audit still records cost and models.
    errors = drift_errors(records, provider_order) if provider_order else []
    if missing:
        errors.append(f"provider audit incomplete: {len(missing)} of {len(ids)} generations not retrievable")
    return summary, errors
