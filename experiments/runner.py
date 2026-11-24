"""Run one registered fault (or the healthy control) on a fresh isolated stack, with a replayable manifest.

Preflight, inject, run the agent, check the fault stayed live, then always restore and verify recovery.
Usage: python -m experiments.runner --case NAME | --list | --init-stack | --preflight | --wait-ready SECONDS
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.config import AgentConfig, dotenv_api_keys
from agent.tools import REQUIRED_TOOLS
from experiments import openrouter_audit, stack, symptoms
from experiments.stack import ROOT, isolated_env

RUNS = ROOT / "experiments" / "runs" / "v2"
STATE_DIR = ROOT / "experiments" / "state"
DOTENV = ROOT / ".env"
PROTOCOL_VERSION = "2.1-draft"
REQUIRED_SERVICES = ("zookeeper", "kafka", "redis1", "redis2", "redis3", "go-server", "go-enricher", "prometheus")
TRAFFIC_SERVICE = "python-producer"
RECOVERY_SECONDS = 120
FAULT_SETTINGS = ROOT / "experiments" / "fault_settings.json"


@dataclass(frozen=True)
class Case:
    name: str
    script: str | None
    service: str | None
    active_state: str | None
    fault_id: int | None
    # Lets a recreated container that crashes on start fail before the first live probe.
    settle_seconds: int = 0
    env: tuple[tuple[str, str], ...] = ()

    def script_env(self) -> dict[str, str]:
        return dict(self.env)


# Only vetted, reversible faults; a script that flushes Redis must never be added.
CASES = {
    "healthy": Case("healthy", None, None, None, None),
    "redis_node_stopped": Case("redis_node_stopped", "redis/01_node_stopped.sh", "redis2", "stopped", 1),
    "ingestion_paused": Case("ingestion_paused", "network/11_container_paused.sh", "go-server", "paused", 11),
    "broker_stopped": Case("broker_stopped", "kafka/17_broker_stopped.sh", "kafka", "stopped", 17),
    "enricher_paused": Case("enricher_paused", "network/18_enricher_paused.sh", "go-enricher", "paused", 18),
    # The target keeps running in these; mechanism_probe checks the cause itself.
    "client_timeout_too_low": Case("client_timeout_too_low", "config/13_client_timeout_too_low.sh", "go-server",
                                   "redis_timeouts_low", 13, settle_seconds=15),
    "cpu_throttled": Case("cpu_throttled", "network/12_cpu_throttle.sh", "go-server", "cpu_limited", 12),
    "latency_injected": Case("latency_injected", "network/09_latency_injection.sh", "go-server", "netem_delay", 9),
    "latency_enricher": Case("latency_enricher", "network/09_latency_injection.sh", "go-enricher", "netem_delay", 9,
                             env=(("TARGET", "go-enricher"),)),
}
# Per-case script settings fault_settings.json may set, with their pre-registered allowed values.
CONFIGURABLE_SETTINGS = {"cpu_throttled": {"LOW_CPUS": ("0.1", "0.01")}}

# Script defaults the mechanism probes expect; tests check them against the scripts.
LOW_TIMEOUT_MS = "1"                     # config/13 LOW_TIMEOUT_MS
TIMEOUT_VARIABLES = ("REDIS_DIAL_TIMEOUT_MS", "REDIS_READ_TIMEOUT_MS", "REDIS_WRITE_TIMEOUT_MS")
LOW_NANO_CPUS = 100_000_000              # network/12 default LOW_CPUS=0.1
NETSHOOT_IMAGE = "nicolaka/netshoot"     # network/09 sidecar image; must already be pulled
NETEM_DELAY = re.compile(r"qdisc netem \S+ root\b.*\bdelay 200(\.0+)?ms")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def case_role(name: str) -> str:
    """control (healthy), visible (container stopped or paused: entry 0 shows it) or hidden (target keeps running)."""
    case = CASES[name]
    if case.script is None:
        return "control"
    return "visible" if case.active_state in ("stopped", "paused") else "hidden"


def load_fault_settings(path: Path | None = None) -> dict[str, dict[str, str]]:
    path = FAULT_SETTINGS if path is None else path
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {name: values for name, values in data.items() if not name.startswith("_")}


def configured_case(name: str, overrides: dict[str, str] | None = None, settings: dict | None = None) -> Case:
    """The case with fault_settings.json then `overrides` applied; only CONFIGURABLE_SETTINGS may change."""
    case = CASES[name]
    settings = load_fault_settings() if settings is None else settings
    chosen = {**settings.get(name, {}), **(overrides or {})}
    allowed = CONFIGURABLE_SETTINGS.get(name, {})
    for key, value in chosen.items():
        if key not in allowed:
            raise ValueError(f"{name}: script setting {key} is not configurable (allowed: {sorted(allowed)})")
        if str(value) not in allowed[key]:
            raise ValueError(f"{name}: {key}={value} is not one of the registered values {allowed[key]}")
    env = {**{key: str(value) for key, value in chosen.items()}, **case.script_env()}
    return dataclasses.replace(case, env=tuple(sorted(env.items())))


def expected_nano_cpus(case: Case) -> int:
    """NanoCpus that network/12 sets for this case's LOW_CPUS (default 0.1)."""
    return round(float(case.script_env().get("LOW_CPUS", "0.1")) * 1e9)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str], *, timeout: int = 60, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, env=isolated_env(env), text=True, encoding="utf-8",
                          errors="replace", capture_output=True, timeout=timeout)


def compose(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return command(["docker", "compose", "-f", str(stack.COMPOSE_FILE), *args], timeout=timeout)


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


# --- stack observations ---------------------------------------------------

def container_state(service: str) -> dict | None:
    ids = compose("ps", "-a", "-q", service)
    if ids.returncode != 0 or not ids.stdout.strip():
        return None
    states = []
    for container_id in ids.stdout.splitlines():
        inspected = command(["docker", "inspect", "--format", "{{json .State}}", container_id])
        if inspected.returncode != 0:
            return None
        states.append(json.loads(inspected.stdout))
    health = [(s.get("Health") or {}).get("Status") for s in states]
    return {"running": any(s.get("Running") and not s.get("Paused") for s in states),
            "paused": any(s.get("Paused") for s in states),
            "healthy": all(not status or status == "healthy" for status in health),
            "count": len(states)}


def one_healthy(state: dict | None) -> bool:
    return bool(state and state["running"] and state["healthy"] and state["count"] == 1)


def stack_health() -> tuple[bool, dict]:
    states = {name: container_state(name) for name in (*REQUIRED_SERVICES, TRAFFIC_SERVICE)}
    return all(one_healthy(state) for state in states.values()), states


def prometheus_health() -> dict[str, float]:
    url = isolated_env()["PROMETHEUS_URL"] + "/api/v1/query?query=up"
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.load(response)
    if payload.get("status") != "success":
        raise RuntimeError("Prometheus up query failed")
    jobs: dict[str, list[float]] = {}
    for series in payload["data"]["result"]:
        jobs.setdefault(series["metric"].get("job", ""), []).append(float(series["value"][1]))
    required = {"prometheus", "redis", "kafka", "go-server", "go-enricher"}
    if not required <= jobs.keys() or any(value != 1 for name in required for value in jobs[name]):
        raise RuntimeError(f"Prometheus targets not all up: {jobs}")
    return {name: min(values) for name, values in jobs.items()}


def topic_end_offsets() -> dict[str, int]:
    try:
        return {topic: sum(stack.topic_offsets(topic).values()) for topic in stack.TOPICS}
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"cannot inspect Kafka topics: {error.stderr.strip()}") from None


def wait_for_growth(topic: str, seconds: int) -> dict:
    """Poll until a topic gains records (the processing check behind recovery claims)."""
    first = topic_end_offsets()[topic]
    deadline = time.monotonic() + seconds
    while True:
        time.sleep(2)
        try:
            last = topic_end_offsets()[topic]
        except RuntimeError:
            last = first  # broker still coming back
        if last > first or time.monotonic() >= deadline:
            return {"topic": topic, "before": first, "after": last, "grew": last > first,
                    "at_utc": utc_now()}


def traffic_probe() -> dict:
    result = wait_for_growth("raw_transactions", 5)
    if not result["grew"]:
        raise RuntimeError("python-producer produced no raw transactions in five seconds")
    return result


def container_ids() -> dict[str, list[str]]:
    return {name: compose("ps", "-a", "-q", name).stdout.splitlines() for name in REQUIRED_SERVICES}


def container_images() -> dict[str, list[str]]:
    images = {}
    for name, ids in container_ids().items():
        images[name] = []
        for container_id in ids:
            inspected = command(["docker", "inspect", "--format", "{{.Image}}", container_id])
            if inspected.returncode != 0:
                raise RuntimeError(f"cannot inspect image for {name}")
            images[name].append(inspected.stdout.strip())
    return images


def redis_key_counts() -> dict[str, int]:
    counts = {}
    for service in ("redis1", "redis2", "redis3"):
        result = compose("exec", "-T", service, "redis-cli", "DBSIZE", timeout=15)
        if result.returncode != 0:
            raise RuntimeError(f"cannot inspect {service} keys: {result.stderr.strip()}")
        counts[service] = int(result.stdout.strip())
    return counts


# --- clean-stack marker ---------------------------------------------------

def marker_path() -> Path:
    """One marker per Compose project, so a throwaway stack never reuses another's."""
    return STATE_DIR / f"{stack.project()}.stack_baseline.json"


def initialize_stack() -> Path:
    states = {name: container_state(name) for name in REQUIRED_SERVICES}
    if not all(one_healthy(state) for state in states.values()):
        raise RuntimeError(f"start one instance of each required service before initialization: {states}")
    producer = container_state(TRAFFIC_SERVICE)
    if producer and producer["running"]:
        raise RuntimeError("stop python-producer before initializing the baseline")
    offsets = topic_end_offsets()
    keys = redis_key_counts()
    # One key per node is go-enricher's own health_check write.
    if any(offsets.values()) or any(count > 1 for count in keys.values()):
        raise RuntimeError(f"stack contains prior data; Kafka offsets={offsets}, Redis keys={keys}")
    path = marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(path, {"schema_version": 1, "compose_project": stack.project(), "created_at_utc": utc_now(),
                          "container_ids": container_ids(), "topic_offsets": offsets, "redis_key_counts": keys})
    return path


def verify_stack_marker() -> tuple[bool, str]:
    path = marker_path()
    if not path.exists():
        return False, f"stack baseline marker {path.name} absent; initialize a clean stack first"
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return False, f"stack baseline marker unreadable: {error}"
    if marker.get("container_ids") != container_ids():
        return False, "required containers changed since baseline initialization"
    if marker.get("used_run_id"):
        return False, f"stack baseline already used by {marker['used_run_id']}; recreate the isolated stack"
    return True, ""


def consume_marker(run_id: str) -> dict:
    marker = json.loads(marker_path().read_text(encoding="utf-8"))
    write_manifest(marker_path(), {**marker, "used_run_id": run_id, "used_at_utc": utc_now()})
    return marker


# --- faults ---------------------------------------------------------------

def first_container_id(service: str) -> str | None:
    ids = compose("ps", "-a", "-q", service)
    return ids.stdout.split()[0] if ids.returncode == 0 and ids.stdout.split() else None


def inspect_container(container_id: str, template: str) -> str | None:
    inspected = command(["docker", "inspect", "--format", template, container_id])
    return inspected.stdout.strip() if inspected.returncode == 0 else None


def netem_qdisc(container_id: str) -> subprocess.CompletedProcess:
    """`tc qdisc show` inside the target's network namespace (same sidecar as the script, never pulled)."""
    return command(["docker", "run", "--rm", "--pull", "never", "--net", f"container:{container_id}",
                    NETSHOOT_IMAGE, "tc", "qdisc", "show", "dev", "eth0"], timeout=60)


def mechanism_probe(case: Case) -> dict:
    """For running-target faults: `active` if the injected setting is in force, `cleared` if it is gone."""
    container_id = first_container_id(case.service)
    if not container_id:
        return {"active": False, "cleared": False, "error": "no container"}
    if case.active_state == "redis_timeouts_low":
        env_json = inspect_container(container_id, "{{json .Config.Env}}")
        if env_json is None:
            return {"active": False, "cleared": False, "error": "docker inspect failed"}
        env = dict(entry.partition("=")[::2] for entry in json.loads(env_json))
        values = {name: env.get(name) for name in TIMEOUT_VARIABLES}
        return {"env": values, "active": all(v == LOW_TIMEOUT_MS for v in values.values()),
                "cleared": all(v == "" for v in values.values())}
    if case.active_state == "cpu_limited":
        nano_cpus = inspect_container(container_id, "{{.HostConfig.NanoCpus}}")
        if nano_cpus is None or not nano_cpus.isdigit():
            return {"active": False, "cleared": False, "error": "docker inspect failed"}
        return {"nano_cpus": int(nano_cpus), "active": int(nano_cpus) == expected_nano_cpus(case),
                "cleared": int(nano_cpus) == 0}
    if case.active_state == "netem_delay":
        shown = netem_qdisc(container_id)
        if shown.returncode != 0:
            return {"active": False, "cleared": False, "error": shown.stderr.strip()[-300:]}
        return {"qdisc": shown.stdout.strip(), "active": bool(NETEM_DELAY.search(shown.stdout)),
                "cleared": "netem" not in shown.stdout}
    raise ValueError(f"no mechanism probe for {case.name}")


def fault_is_live(case: Case) -> tuple[bool, dict | None]:
    if case.script is None:
        return True, None
    state = container_state(case.service)
    if not state:
        return False, None
    if case.active_state == "stopped":
        return not state["running"] and not state["paused"], state
    if case.active_state == "paused":
        return state["paused"], state
    if not state["running"]:
        return False, state
    probe = mechanism_probe(case)
    return probe["active"], {**state, "mechanism": probe}


TARGET_DOWN = "target not running after injection"


def live_failure(case: Case, state: dict | None) -> str:
    """Validity error for a failed live probe; an exited running-target fault is named as such, never scored."""
    if case.active_state not in ("stopped", "paused") and (state is None or not state.get("running")):
        return (f"{TARGET_DOWN}: {case.service} is not running (no restart policy; e.g. a failed startup "
                f"ping), so {case.name} was not live")
    return "fault injection returned success but its live-state probe failed"


def fault_is_restored(case: Case) -> tuple[bool, dict | None]:
    """Target container back (plus cluster/broker API). Processing is checked separately."""
    if case.script is None:
        return True, None
    state = container_state(case.service)
    if not state or not state["running"] or not state["healthy"]:
        return False, state
    if case.service in ("redis1", "redis2", "redis3"):
        probe = compose("exec", "-T", "redis1", "redis-cli", "CLUSTER", "INFO")
        return probe.returncode == 0 and "cluster_state:ok" in probe.stdout, state
    if case.service == "kafka":
        probe = compose("exec", "-T", "kafka", "kafka-broker-api-versions", "--bootstrap-server", "kafka:29092")
        return probe.returncode == 0, state
    if case.active_state not in ("stopped", "paused"):
        probe = mechanism_probe(case)
        return probe["cleared"], {**state, "mechanism": probe}
    return True, state


def bash_binary() -> str:
    configured = os.environ.get("BASH_BIN")
    if configured:
        return configured
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    return str(git_bash) if os.name == "nt" and git_bash.exists() else "bash"


def script_action(case: Case, action: str, log_dir: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, **case.script_env()} if case.env else None
    output = command([bash_binary(), str(ROOT / "fault-injection" / case.script), action], timeout=240, env=env)
    (log_dir / f"{action}.stdout.log").write_text(output.stdout, encoding="utf-8")
    (log_dir / f"{action}.stderr.log").write_text(output.stderr, encoding="utf-8")
    return output


def restore_and_wait(case: Case, log_dir: Path) -> tuple[list[str], dict]:
    errors = []
    restored = script_action(case, "restore", log_dir)
    if restored.returncode != 0:
        errors.append(f"restore script exited {restored.returncode}")
    recovered, state = False, None
    for _ in range(RECOVERY_SECONDS):
        recovered, state = fault_is_restored(case)
        if recovered:
            break
        time.sleep(1)
    if not recovered:
        errors.append("target did not recover")
    return errors, {"at_utc": utc_now(), "restore_exit_code": restored.returncode,
                    "recovered": recovered, "state": state}


# --- agent ----------------------------------------------------------------

def agent_python() -> str:
    configured = os.environ.get("AGENT_PYTHON")
    if configured:
        return configured
    bundled = ROOT / "agent" / "venv" / "Scripts" / "python.exe"
    return str(bundled) if bundled.exists() else sys.executable


def agent_environment(run_id: str, run_dir: Path) -> dict[str, str]:
    env = isolated_env()
    for name in ("FAULT_ID", "FAULT_NAME", "FAULT_FAMILY"):
        env.pop(name, None)
    # Same precedence as the agent's load_dotenv: .env fills only key variables not already set.
    for name, value in dotenv_api_keys(DOTENV).items():
        env.setdefault(name, value)
    env.update({"RUN_TAG": run_id, "RESULTS_DIR": str(run_dir)})
    return env


def check_agent_artifacts(run_dir: Path, run_id: str) -> tuple[dict, list[dict]]:
    """The agent's result and trace, or RuntimeError naming what is missing."""
    result_path = run_dir / f"{run_id}.result.json"
    if not result_path.exists():
        raise RuntimeError("agent result is absent")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise RuntimeError(f"agent result status is {result.get('status')!r}, not completed")
    if result.get("root_cause_id") == "PARSE_FAILURE":
        raise RuntimeError("agent conclusion could not be parsed")
    trace_path = run_dir / f"{run_id}.trace.jsonl"
    if not trace_path.exists():
        raise RuntimeError("agent trace is absent")
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    if not trace or trace[0].get("event") != "investigation_started":
        raise RuntimeError("agent trace header is absent")
    if any(key in trace[0] for key in ("fault_id", "fault_name", "fault_family")):
        raise RuntimeError("fault metadata leaked into the agent trace header")
    tool_names = {spec.get("name") for spec in trace[0].get("tool_specs", [])}
    if tool_names != REQUIRED_TOOLS:
        raise RuntimeError(f"agent received an unexpected tool set: {sorted(tool_names)}")
    if not any(event.get("event") == "action_selected" and event.get("agent_visible_prompt") for event in trace):
        raise RuntimeError("agent-visible investigation prompt is absent")
    if any(event.get("finish_reason") == "length" for event in trace
           if event.get("event") in ("action_selected", "conclusion")):
        raise RuntimeError("model response ended at its token limit")
    return result, trace


# --- model-free runbook snapshot (never shown to the agent) ---------------

GO_SERVICES = ("go-server", "go-enricher")
CONFIG_PREFIXES = ("REDIS_", "KAFKA_", "GRPC_")
SECRET_MARKERS = ("KEY", "SECRET", "PASSWORD", "TOKEN", "LICENSE")


def state_name(state: dict | None) -> str:
    if not state:
        return "absent"
    if state["paused"]:
        return "paused"
    return "running" if state["running"] else "exited"


def go_config(service: str) -> dict[str, str] | str:
    """A Go service's REDIS_/KAFKA_/GRPC_ environment without secret-like names, as read_config shows it."""
    container_id = first_container_id(service)
    env_json = inspect_container(container_id, "{{json .Config.Env}}") if container_id else None
    if env_json is None:
        return "unavailable"
    pairs = (entry.partition("=") for entry in json.loads(env_json))
    return {key: value for key, _, value in pairs
            if key.startswith(CONFIG_PREFIXES) and not any(marker in key.upper() for marker in SECRET_MARKERS)}


def runbook_snapshot() -> dict:
    """Container states, RED metrics and Go config at one moment, for rule R3 and the manipulation check."""
    taken = {"at_utc": utc_now(),
             "states": {name: state_name(container_state(name)) for name in REQUIRED_SERVICES},
             "red": symptoms.snapshot(isolated_env()["PROMETHEUS_URL"])}
    taken["config"] = {service: go_config(service) for service in GO_SERVICES}
    return taken


def record_snapshot(manifest: dict, key: str) -> None:
    """Take a runbook snapshot into manifest[key]; a failure is noted, never a validity error."""
    try:
        manifest[key] = runbook_snapshot()
    except Exception as error:
        manifest[key] = None
        manifest.setdefault("runbook_errors", []).append(f"{key}: {error}")


# --- one run --------------------------------------------------------------

def new_manifest(case: Case, run_id: str) -> dict:
    source = command(["git", "rev-parse", "HEAD"])
    dirty = command(["git", "status", "--porcelain"])
    env = isolated_env()
    return {
        "schema_version": PROTOCOL_VERSION, "run_id": run_id, "compose_project": stack.project(),
        "case": case.name, "case_role": case_role(case.name), "case_env": case.script_env(),
        "fault_id": case.fault_id, "fault_script": case.script,
        "fault_script_sha256": sha256(ROOT / "fault-injection" / case.script) if case.script else None,
        "source_commit": source.stdout.strip() if source.returncode == 0 else None,
        "source_dirty": bool(dirty.stdout.strip()),
        "model_id": env.get("MODEL_ID"), "model_settings": None,
        "geoip_source": env.get("GEOIP_DATA") or "maxmind volume",
        "traffic_profile": {key: env[key] for key in ("TRAFFIC_SEED", "TRAFFIC_INTERVAL_SECONDS",
                                                      "TRAFFIC_FRAUD_PROBABILITY", "TRAFFIC_USER_LIMIT")},
        "created_at_utc": utc_now(), "status": "created", "validity_errors": [],
    }


def preflight(manifest: dict) -> None:
    healthy, states = stack_health()
    manifest["preflight_at_utc"] = utc_now()
    manifest["preflight_states"] = states
    if not healthy:
        raise RuntimeError("required Compose services are not all running")
    marker_valid, marker_error = verify_stack_marker()
    if not marker_valid:
        raise RuntimeError(marker_error)
    manifest["container_image_ids"] = container_images()
    manifest["prometheus_up_before"] = prometheus_health()
    manifest["traffic_probe"] = traffic_probe()
    manifest["topic_end_offsets_before"] = topic_end_offsets()
    manifest["redis_key_counts_before"] = redis_key_counts()


def run_case(case: Case, runs_dir: Path = RUNS, extra: dict | None = None) -> Path:
    """One attempt; any failure goes to validity_errors and the run is kept as invalid, so it stays countable."""
    run_id = "inc-" + uuid.uuid4().hex
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest = {**new_manifest(case, run_id), **(extra or {})}
    write_manifest(manifest_path, manifest)
    injected = interrupted = False
    agent_config = None
    try:
        # Validate model settings exactly as the agent will, before touching the stack.
        agent_env = agent_environment(run_id, run_dir)
        agent_config = AgentConfig.from_env(agent_env)
        manifest["model_id"] = agent_config.model_id
        manifest["model_settings"] = agent_config.model_settings.as_dict()

        # The clean-stack marker is single use, so no run inherits another run's state.
        preflight(manifest)
        manifest["stack_baseline"] = consume_marker(run_id)
        manifest["status"] = "preflight_passed"
        record_snapshot(manifest, "runbook_snapshot_before")
        write_manifest(manifest_path, manifest)

        if case.script:
            injected = True  # restore even if injection fails part way
            manifest["inject_started_at_utc"] = utc_now()
            injected_result = script_action(case, "inject", run_dir)
            manifest["inject_ended_at_utc"] = utc_now()
            if injected_result.returncode != 0:
                raise RuntimeError(f"fault injection failed with exit {injected_result.returncode}")
            time.sleep(case.settle_seconds)
            live, state = fault_is_live(case)
            manifest["live_probe_before"] = {"at_utc": utc_now(), "live": live, "state": state}
            if not live:
                raise RuntimeError(live_failure(case, state))

        manifest["agent_started_at_utc"] = utc_now()
        write_manifest(manifest_path, manifest)
        result = command([agent_python(), "-m", "agent.run"], timeout=1800, env=agent_env)
        manifest["agent_ended_at_utc"] = utc_now()
        (run_dir / "agent.stdout.log").write_text(result.stdout, encoding="utf-8")
        (run_dir / "agent.stderr.log").write_text(result.stderr, encoding="utf-8")
        manifest["agent_exit_code"] = result.returncode
        if result.returncode != 0:
            raise RuntimeError(f"agent exited with code {result.returncode}")
        agent_result, trace = check_agent_artifacts(run_dir, run_id)
        observations = [event["ts"] for event in trace if event.get("event") == "tool_call"]
        manifest["first_observation_at_utc"] = min(observations) if observations else None
        manifest["last_observation_at_utc"] = max(observations) if observations else None
        manifest["agent_tool_schema_sha256"] = agent_result.get("tool_schema_sha256")
        # The fault must still be live when the agent ends, or it may have diagnosed a recovered stack.
        if case.script:
            live, state = fault_is_live(case)
            manifest["live_probe_after"] = {"at_utc": utc_now(), "live": live, "state": state}
            if not live:
                down = live_failure(case, state).startswith(TARGET_DOWN)
                raise RuntimeError(f"{TARGET_DOWN}: {case.service} exited during the agent run" if down
                                   else "fault ended before the agent completed")
        manifest["status"] = "agent_completed"
        # Before restore: what a runbook would have seen during the incident.
        record_snapshot(manifest, "runbook_snapshot")
        before, after = manifest.get("runbook_snapshot_before"), manifest.get("runbook_snapshot")
        manifest["manipulation_check"] = symptoms.manipulation_check(
            case.name, case.script is not None, (before or {}).get("red"), (after or {}).get("red"))
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        manifest["validity_errors"].append("interrupted" if interrupted else str(error))
        manifest["status"] = "invalid"
    finally:
        # Restore and check that processing resumed, whatever happened.
        if injected:
            manifest["restore_started_at_utc"] = utc_now()
            try:
                errors, manifest["recovery_probe"] = restore_and_wait(case, run_dir)
                manifest["validity_errors"].extend(errors)
            except Exception as error:
                manifest["validity_errors"].append(f"restore exception: {error}")
        if "preflight_states" in manifest:
            try:
                manifest["processing_probe"] = wait_for_growth("enriched_transactions", RECOVERY_SECONDS)
                if not manifest["processing_probe"]["grew"]:
                    manifest["validity_errors"].append("enriched output did not resume after the run")
            except Exception as error:
                manifest["validity_errors"].append(f"processing probe failed: {error}")
        audited = agent_config is not None and bool(agent_config.model_settings.provider_order
                                                    or agent_config.model_settings.require_zero_cost)
        if audited and "agent_started_at_utc" in manifest and not interrupted:
            try:
                manifest["generation_audit"], errors = openrouter_audit.audit_run(
                    run_dir, run_id, agent_config.base_url, agent_config.api_key,
                    agent_config.model_settings.provider_order)
                manifest["validity_errors"].extend(errors)
            except Exception as error:
                manifest["validity_errors"].append(f"provider audit failed: {error}")
        try:
            manifest["topic_end_offsets_after"] = topic_end_offsets()
            manifest["redis_key_counts_after"] = redis_key_counts()
        except Exception as error:
            manifest["validity_errors"].append(f"post-run state capture failed: {error}")
        ok = not manifest["validity_errors"] and manifest["status"] == "agent_completed"
        manifest["status"] = "completed" if ok else "invalid"
        manifest["finished_at_utc"] = utc_now()
        write_manifest(manifest_path, manifest)
    if interrupted:
        raise KeyboardInterrupt
    return run_dir


def wait_ready(seconds: int) -> tuple[bool, str]:
    deadline, last_error = time.monotonic() + seconds, ""
    while time.monotonic() < deadline:
        states = {name: container_state(name) for name in REQUIRED_SERVICES}
        try:
            if all(one_healthy(state) for state in states.values()):
                return True, f"ready: {stack.project()} {prometheus_health()}"
            last_error = f"services: {states}"
        except Exception as error:
            last_error = str(error)
        time.sleep(5)
    return False, f"not ready after {seconds}s: {last_error}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--init-stack", action="store_true")
    parser.add_argument("--wait-ready", type=int, metavar="SECONDS",
                        help="wait until each required service is up and Prometheus scrapes them all")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--runs-dir", type=Path, default=RUNS,
                        help="where run folders go (default experiments/runs/v2, the scored population)")
    args = parser.parse_args()
    if args.list:
        for case in CASES.values():
            print(f"{case.name}: {case.script or 'no injection'}")
        return 0
    if args.wait_ready is not None:
        ready, detail = wait_ready(args.wait_ready)
        print(detail, file=sys.stdout if ready else sys.stderr)
        return 0 if ready else 1
    if args.init_stack:
        print(f"initialized clean stack: {initialize_stack()}")
        return 0
    if args.preflight:
        healthy, states = stack_health()
        marker_valid, marker_error = verify_stack_marker()
        observation_error, observations = "", {}
        if healthy and marker_valid:
            try:
                observations = {"prometheus_up": prometheus_health(), "traffic": traffic_probe()}
            except Exception as error:
                observation_error = str(error)
        print(json.dumps({"compose_project": stack.project(), "healthy": healthy, "marker_valid": marker_valid,
                          "marker_error": marker_error, "observation_error": observation_error,
                          "observations": observations, "states": states}, indent=2))
        return 0 if healthy and marker_valid and not observation_error else 1
    if not args.case:
        parser.error("--case is required unless --list, --init-stack or --preflight is used")
    path = run_case(configured_case(args.case), args.runs_dir)
    result = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    print(f"{result['status']}: {path}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
