"""Model-free RED symptom snapshots and the pre-registered symptom criteria (never shown to the agent).

One definition serves gate G2/G4, the per-run manipulation check and rule R3, so they cannot drift apart.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone

LATENCY_FACTOR = 3.0
RATE_FACTOR = 0.5
ERROR_RATIO = 0.01

QUERIES = {
    "server_requests_per_s": "sum(rate(fraud_server_transactions_total[30s]))",
    "server_errors_per_s": "sum(rate(fraud_server_transaction_errors_total[30s]))",
    "server_p50_s": "histogram_quantile(0.5, sum by (le) (rate(fraud_server_transaction_duration_seconds_bucket[30s])))",
    "enricher_processed_per_s": "sum(rate(fraud_enricher_messages_processed_total[30s]))",
    "enricher_errors_per_s": "sum(rate(fraud_enricher_errors_total[30s]))",
    "enricher_p50_s": "histogram_quantile(0.5, sum by (le) (rate(fraud_enricher_message_duration_seconds_bucket[30s])))",
    # CPU signals for rule R3 (both are visible to the agent through get_metrics).
    "server_cpu_per_s": 'sum(rate(process_cpu_seconds_total{job="go-server"}[30s]))',
    "enricher_cpu_per_s": 'max(rate(process_cpu_seconds_total{job="go-enricher"}[30s]))',
    "server_gomaxprocs": 'max(go_sched_gomaxprocs_threads{job="go-server"})',
    "enricher_gomaxprocs": 'max(go_sched_gomaxprocs_threads{job="go-enricher"})',
}
# Component -> (latency, throughput, errors) snapshot keys; the order is the R3 check order.
COMPONENTS = {
    "go-server": ("server_p50_s", "server_requests_per_s", "server_errors_per_s"),
    "go-enricher": ("enricher_p50_s", "enricher_processed_per_s", "enricher_errors_per_s"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def query(prometheus_url: str, promql: str, opener=urllib.request.urlopen) -> float | None:
    """Instant value of a scalar-like query; None for no series or NaN. Raises on HTTP or API errors."""
    url = prometheus_url.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": promql})
    with opener(url, timeout=10) as response:
        payload = json.load(response)
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus rejected {promql!r}: {payload.get('error')}")
    result = payload["data"]["result"]
    if not result:
        return None
    value = float(result[0]["value"][1])
    return None if math.isnan(value) or math.isinf(value) else value


def snapshot(prometheus_url: str, opener=urllib.request.urlopen) -> dict:
    """Every QUERIES value at one moment; a failed query is recorded as 'query failed: ...', never raised."""
    taken = {"at_utc": utc_now()}
    for name, promql in QUERIES.items():
        try:
            taken[name] = query(prometheus_url, promql, opener)
        except Exception as error:
            taken[name] = f"query failed: {error}"
    return taken


def _number(snapshot_: dict | None, key: str) -> float | None:
    value = (snapshot_ or {}).get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _failed(snapshot_: dict | None, key: str) -> bool:
    return isinstance((snapshot_ or {}).get(key), str)


def component_symptoms(before: dict | None, after: dict | None, component: str) -> list[str]:
    """Criteria met for one component: latency up LATENCY_FACTOR x, throughput down to RATE_FACTOR x, or errors."""
    latency, throughput, errors = COMPONENTS[component]
    met = []
    before_latency, after_latency = _number(before, latency), _number(after, latency)
    if before_latency and after_latency is not None and after_latency >= LATENCY_FACTOR * before_latency:
        met.append(f"{latency} {after_latency:.4g} >= {LATENCY_FACTOR:g} x {before_latency:.4g}")
    before_rate = _number(before, throughput)
    if before_rate and not _failed(after, throughput):
        after_rate = _number(after, throughput) or 0.0  # no samples: nothing was processed or scraped
        if after_rate <= RATE_FACTOR * before_rate:
            met.append(f"{throughput} {after_rate:.4g} <= {RATE_FACTOR:g} x {before_rate:.4g}")
    after_errors, after_rate = _number(after, errors), _number(after, throughput)
    if after_errors:
        # go-server counts every received request; the enricher counts only successes.
        total = (after_rate or 0.0) + (after_errors if component == "go-enricher" else 0.0)
        if total > 0 and after_errors / total >= ERROR_RATIO:
            met.append(f"error ratio {after_errors / total:.4g} >= {ERROR_RATIO:g}")
    return met


def symptoms_present(before: dict | None, after: dict | None) -> dict[str, list[str]]:
    """Component -> criteria met, only for components with at least one."""
    found = {component: component_symptoms(before, after, component) for component in COMPONENTS}
    return {component: met for component, met in found.items() if met}


def manipulation_check(case_name: str, has_script: bool, before: dict | None, after: dict | None) -> dict:
    """A fault run passes with a symptom, healthy without; None if unknown. Reported, never a validity gate."""
    if not before or not after or any(_failed(s, key) for s in (before, after)
                                      for keys in COMPONENTS.values() for key in keys):
        return {"case": case_name, "symptoms": None, "passed": None,
                "note": "snapshot missing or a query failed"}
    found = symptoms_present(before, after)
    return {"case": case_name, "symptoms": found, "passed": bool(found) if has_script else not found}
