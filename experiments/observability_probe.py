"""Gates G3/G4: does a fixed, pre-registered MCP tool call show the case's discriminating fact? No model involved.

Calls the MCP server's tool functions in-process with the agent's 2,000-character reply limit; needs that interpreter.
Usage: mcp-server\\venv\\Scripts\\python.exe -m experiments.observability_probe --case latency_injected
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

from experiments.runner import CASES, LOW_TIMEOUT_MS, ROOT, configured_case

# Frozen before any model run: the 200 ms delay adds at least 0.15 s to every request or message.
LATENCY_THRESHOLD_S = 0.15
CPU_QUOTA_FRACTION = 0.8
REPLY_LIMIT = 2000
SERVER_P50 = "histogram_quantile(0.5, sum by (le) (rate(fraud_server_transaction_duration_seconds_bucket[1m])))"
ENRICHER_P50 = "histogram_quantile(0.5, sum by (le) (rate(fraud_enricher_message_duration_seconds_bucket[1m])))"
SERVER_CPU = 'rate(process_cpu_seconds_total{job="go-server"}[1m])'
CONTAINER_STATE_OF = {"stopped": "exited", "paused": "paused"}


def probes(case_name: str) -> list[dict]:
    """The pre-registered tool calls and checks for one case ([] for healthy)."""
    case = CASES[case_name]
    if case.script is None:
        return []
    if case.active_state in CONTAINER_STATE_OF:
        state = CONTAINER_STATE_OF[case.active_state]
        return [{"tool": "get_container_state", "args": {}, "check": "regex",
                 "pattern": rf"(?m)^{re.escape(case.service)}: state={state}\b"}]
    latency = {"go-server": SERVER_P50, "go-enricher": ENRICHER_P50}[case.service]
    latency_probe = {"tool": "get_metrics", "args": {"promql": latency, "window": "5m"},
                     "check": "latest_at_least", "threshold": LATENCY_THRESHOLD_S}
    if case.active_state == "netem_delay":
        return [latency_probe]
    if case.active_state == "cpu_limited":
        return [{"tool": "get_metrics", "args": {"promql": SERVER_CPU, "window": "5m"},
                 "check": "latest_at_least_cpu_quota", "fraction": CPU_QUOTA_FRACTION}, latency_probe]
    if case.active_state == "redis_timeouts_low":
        return [{"tool": "read_config", "args": {"service": case.service}, "check": "regex",
                 "pattern": rf"REDIS_(DIAL|READ|WRITE)_TIMEOUT_MS={LOW_TIMEOUT_MS}\b"}]
    raise ValueError(f"no observability probe for {case_name}")


def latest_values(reply: str) -> list[float]:
    """The newest value of each series in a get_metrics reply ('... -> latest ts:v, ts:v; range=...')."""
    values = []
    for line in reply.splitlines():
        _, marker, rest = line.partition(" -> latest ")
        if not marker:
            continue
        newest = rest.split(";")[0].split(",")[-1].strip()
        try:
            value = float(newest.rpartition(":")[2])
        except ValueError:
            continue
        if value == value:  # not NaN
            values.append(value)
    return values


def check(probe: dict, reply: str, script_env: dict[str, str]) -> tuple[bool, str]:
    if len(reply) > REPLY_LIMIT:
        return False, f"reply has {len(reply)} characters, more than the agent's {REPLY_LIMIT}"
    if probe["check"] == "regex":
        found = re.search(probe["pattern"], reply)
        return bool(found), f"matched {found.group(0)!r}" if found else "pattern not in reply"
    values = latest_values(reply)
    if probe["check"] == "latest_at_least":
        threshold = probe["threshold"]
    elif probe["check"] == "latest_at_least_cpu_quota":
        threshold = probe["fraction"] * float(script_env.get("LOW_CPUS", "0.1"))
    else:
        raise ValueError(f"unknown check {probe['check']}")
    if not values:
        return False, "no numeric latest value in reply"
    return max(values) >= threshold, f"latest {max(values):.4g} vs threshold {threshold:.4g}"


def evaluate(case_name: str, replies: list[str], script_env: dict[str, str] | None = None) -> dict:
    """Verdict for one probe pass: `matched` is true if any reply shows the discriminating fact."""
    calls = []
    for probe, reply in zip(probes(case_name), replies):
        matched, detail = check(probe, reply, script_env or {})
        calls.append({"tool": probe["tool"], "args": probe["args"], "reply": reply, "chars": len(reply),
                      "matched": matched, "detail": detail})
    return {"case": case_name, "at_utc": datetime.now(timezone.utc).isoformat(), "calls": calls,
            "matched": any(call["matched"] for call in calls)}


def call_tools(case_name: str) -> list[str]:
    """Run the case's probe calls through the MCP server functions (needs the MCP interpreter)."""
    sys.path.insert(0, str(ROOT / "mcp-server"))
    import server  # noqa: E402  (reads PROMETHEUS_URL and the Compose project from the environment)

    return [getattr(server, probe["tool"])(**probe["args"]) for probe in probes(case_name)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--setting", action="append", default=[], metavar="KEY=VALUE",
                        help="script setting override, e.g. LOW_CPUS=0.01 (as validate_faults --set)")
    args = parser.parse_args()
    overrides = dict(item.split("=", 1) for item in args.setting)
    case = configured_case(args.case, overrides)
    from experiments.stack import isolated_env
    os.environ.update(isolated_env())  # PROMETHEUS_URL and COMPOSE_PROJECT_NAME of the benchmark stack
    print(json.dumps(evaluate(args.case, call_tools(args.case), case.script_env()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
