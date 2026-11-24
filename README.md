# LLM Incident-Diagnosis Agent and Fault-Injection Benchmark

Can an LLM agent find the root cause of a live production incident using only read-only observability tools?

This repo contains:
- **A fault-injection harness.** It breaks a running microservice stack in controlled, reversible ways.
- **A LangGraph diagnostic agent.** It investigates the stack through 5 read-only MCP tools.
- **An evaluation pipeline.** It grades each diagnosis against the fault that was actually injected.

The system under test is a small Go/Kafka/Redis transaction-processing stack (`pipeline/`). It is a testbed only.

## Results at a glance

Four models were tested on 9 live incidents. They saw identical incidents: each fault was injected once and all models investigated it at the same time.

| Finding | Result |
|---|---|
| Visible outages (Kafka broker or Redis node stopped) correctly diagnosed | **13 / 13** valid runs |
| False alarms on a healthy stack | **0 / 4** |
| Hidden network-latency faults fully diagnosed | **0 / 12** valid runs (3 partial) |
| Hidden-fault runs that queried any latency metric | **0 / 12** |
| After a fault-agnostic "golden signals" prompt change: runs that checked latency | **0 / 8 → 7 / 8** |
| Same comparison: runs that flagged the slowed service | **2 / 8 → 6 / 8** |
| Same comparison: runs that named network latency as the root cause | **0 / 8 → 2 / 8** |
| Fault runs where the fault was confirmed live at the end and the stack recovered | **44 / 44** |

**Main finding.** The agents could find outages they could see directly. They missed hidden latency faults because they never looked at latency.
- **The fix helped them look.** Adding a short generic instruction to check the four golden signals got them looking, and most then flagged the right component.
- **It did not help them explain.** Few named the mechanism, because none of the tools exposes the network layer.

These are small samples (2 runs per fault per model). Treat the before/after change as directional (Fisher exact on misses, 6/8 vs 2/8: p ≈ 0.13). Details are in [Results](#results).

## How it works

```
                 ┌──────────────── experiment harness (experiments/) ────────────────┐
                 │  fresh isolated stack → inject fault → verify live → run agents   │
                 │  in parallel → verify fault still live → restore → verify recovery│
                 └───────┬──────────────────────────────────────────────┬────────────┘
                         │ inject / restore                             │ spawn N agents
                         ▼                                              ▼
 ┌──────────── testbed (pipeline/, docker-compose.yaml) ─────┐   ┌──── LangGraph agent (agent/) ───┐
 │ producer ─gRPC─▶ go-server ─▶ Kafka ─▶ go-enricher ─▶ Kafka│   │ baseline → investigate ⇄ tools  │
 │                    │  Redis cluster  │      Prometheus     │◀──│ → conclude (cited evidence)     │
 └───────────────────────────────────────────────────────────┘   └──────────────┬──────────────────┘
                         ▲                                                      │ MCP (stdio)
                         └──── 5 read-only tools (mcp-server/) ◀────────────────┘
```

### Fault injection (`fault-injection/`)

There are 18 `inject` / `restore` shell scripts across Redis, Kafka, network and config faults. Eight are registered as benchmark cases in `experiments/runner.py`.

| Case | Kind | Mechanism |
|---|---|---|
| `healthy` | control | Nothing is injected; the right answer is "no fault" |
| `broker_stopped` | visible | `docker compose stop kafka` |
| `redis_node_stopped` | visible | Stops one Redis cluster master (`redis2`) |
| `latency_injected` | hidden | `tc netem delay 200ms` on go-server, through a netshoot sidecar in its network namespace |
| `latency_enricher` | hidden | The same delay on go-enricher |
| `ingestion_paused`, `enricher_paused` | visible | `docker pause` on the service |
| `client_timeout_too_low`, `cpu_throttled` | hidden | Rejected by the validity gates (below) |

**Visible** means a container is down. **Hidden** means every container stays up and the fault shows only in metrics.

### Agent (`agent/`)

The agent is a LangGraph state machine: `baseline → investigate ⇄ tool_call → conclude`.
- **Tool budget:** 15 calls, including 2 fixed baseline observations.
- **Bounded observations:** each tool reply is capped at 2,000 characters, keeping the head and tail with a truncation marker.
- **Typed conclusion:** `root_cause_id`, a summary, cited evidence indices and a confidence level (`confirmed`, `best_guess` or `inconclusive`).
- **Confirmation rule:** a `confirmed` answer must cite usable evidence from at least two tools.
- **Generic trigger:** the agent is never told which fault was injected.
- **Prompt variants:** `AGENT_PROMPT_VARIANT=v2` appends a 3-line golden-signals addendum (latency, traffic, errors, saturation, compared against a baseline window). The variant and the prompt's sha256 are recorded in every trace.

### Observation tools (`mcp-server/`)

There are 5 read-only MCP tools: `query_logs`, `get_metrics` (PromQL), `list_metric_names`, `get_container_state` and `read_config`.
- Arguments are validated against allow-lists and regexes.
- Docker is called with argument lists, never through a shell.
- Secret-like environment variables are redacted.
- Replies are bounded: at most 500 log lines, 10 series and 60 points per series.
- Errors come back as plain `ERROR: …` text that never hints at a diagnosis.

### Evaluation harness (`experiments/`)

**Paired, concurrent runs.** `quick_eval.py` runs one *group* per incident:
1. Bring up a fresh, isolated Docker Compose project.
2. Inject the fault once and confirm it is live.
3. Run one agent per model at the same time, in a thread pool.
4. Check that the fault was still live when each agent finished.
5. Restore the stack, then require the target to recover and processing to resume.

**Model-free validity gates (Phase 0).** `validate_faults.py --gate` checks each candidate hidden fault before any paid run:
- **G1:** the fault is live.
- **G2:** it causes a measurable symptom.
- **G3:** the symptom is visible through the agent's own tools.
- **G4:** a healthy stack does not trigger the probe.
- **G5:** it restores cleanly.

Two of four candidates failed and were dropped. For example, the CPU-throttle probe also fired on a healthy stack.

**Spend control (OpenRouter).**
- A worst-case cost is reserved before each concurrent run.
- Caps apply per model, per group and globally.
- Providers are pinned with fallbacks disabled.
- Every billed call is reconciled against OpenRouter's generation records.
- A batch stops automatically on key errors, cost-cap overruns or provider drift.
- The free-tier model is checked to really cost $0.

**Offline mode.** `fake_model.py` is a scripted OpenAI-compatible endpoint. It runs the real agent, tools and harness end to end with no API calls.

## Results

Models (via OpenRouter, provider pinned):
- `openai/gpt-6-luna`
- `z-ai/glm-5.3`
- `google/gemma-4-31b-it`
- `nvidia/nemotron-3-ultra-550b-a55b:free`

One row per run is in [`results/runs.csv`](results/runs.csv).

**Grading.** Each run is graded against the injected fault:

| Label | Meaning |
|---|---|
| correct | Right component and right mechanism, or "no fault" on the healthy control |
| partial | Right component as root cause, but explained as throughput or lag rather than latency |
| spotted | No root cause given, but the summary flags an anomaly on the right component |
| miss | Called the stack healthy, blamed the wrong component, or dismissed the anomaly |
| invalid | Crashed or unparseable output. Counted as an availability failure and excluded from accuracy |

### Experiment 1: original prompt (36 runs: 9 incidents × 4 models)

| | luna | GLM | gemma | Nemotron | All |
|---|---|---|---|---|---|
| Valid runs | 9/9 | 7/9 | 4/9 | 9/9 | 29/36 |
| Healthy: no false alarm | 1/1 | 1/1 | 1/1 | 1/1 | 4/4 |
| Visible faults correct | 4/4 | 4/4 | 1/1 | 4/4 | **13/13** |
| Hidden faults correct | 0/4 | 0/2 | 0/2 | 0/4 | **0/12** |
| Hidden faults partial | 1 | 0 | 1 | 1 | 3 |

**What the traces show.**
- None of the 12 valid hidden-fault runs issued a latency query (any PromQL using `duration`, `latency`, `bucket` or `quantile`).
- The agents checked liveness, throughput, Kafka offsets and consumer lag instead. They judged current values without a baseline, although these faults cut request rates by about 70% in the model-free gate measurements.
- The 2,000-character observation cap was not the cause: 3 of the 7 go-server misses had no truncated observation at all.

**Failures were mostly infrastructure.** 7 of 36 runs were invalid:
- gemma: 5, all upstream provider 429s on a shared rate-limit pool.
- GLM: 2, unparseable final answers.

### Experiment 2: golden-signals prompt on the hidden faults (luna and Nemotron, 8 runs vs 8)

Only the two hidden faults were rerun (2 injections each). Healthy and visible faults were not rerun, so this experiment says nothing about regressions on them.

| | Original prompt | Golden-signals prompt |
|---|---|---|
| Queried a latency metric | 0/8 | **7/8** |
| Flagged the slowed service (correct + partial + spotted) | 2/8 | **6/8** |
| Named latency as the root cause (correct) | 0/8 | **2/8** |
| Missed | 6/8 | **2/8** |

**By model.**
- **luna:** located the slowed service in 4/4 runs, but answered "inconclusive" every time. For example, it reported go-server p95 rising from 0.024 s to 0.455 s without naming a cause.
- **Nemotron:** diagnosed the enricher latency correctly twice. It saw the go-server latency rise and dismissed it twice.

**Likely remaining gap: tool coverage (interpretation, not measured).** No tool exposes the network layer (qdisc, netem), so an agent can see *that* a service slowed down but not *why*.

### Cost and reliability

| | luna | GLM | gemma | Nemotron |
|---|---|---|---|---|
| Cost per run (experiment 1) | $0.0078 | $0.193 | $0.0024 | $0 (free tier, audited) |
| Valid runs, both experiments | 13/13 | 7/9 | 4/13 | 13/13 |
| Agent wall time, valid runs | 55–182 s | 307–646 s | 43–263 s | 100–297 s |

### Limitations

- **Small samples:** 1–2 runs per fault per model, so no significance claims. The before/after comparison reuses the same 2 fault types, so the prompt change has not been tested on a held-out fault.
- **Grader:** labels were assigned against the injected fault using the rubric above, by a single grader.
- **Planned design not run:** a larger 152-run design is fully specified in `experiments/prereg/` but was not run.

## Repository layout

| Path | Contents |
|---|---|
| `fault-injection/` | `common.sh` and the numbered `inject` / `restore` scripts |
| `agent/` | LangGraph agent, MCP client tools, system prompt and variants, run entry point |
| `mcp-server/` | The 5 read-only observation tools |
| `experiments/` | Stack profiles, run harness, quick eval, validity gates, symptom and observability probes, OpenRouter audit, fake model, scoring |
| `experiments/prereg/` | Model arms, caps and the pre-registered matrix design |
| `pipeline/`, `docker-compose.yaml`, `prometheus/` | The testbed: Go gRPC ingest, Kafka enricher, Redis cluster, synthetic traffic, Prometheus |
| `tests/` | Unit and integration tests (fakes, scripted endpoint, MCP-to-agent path) |
| `results/runs.csv` | One row per evaluated run |

## Quick start

### Requirements

- Docker Desktop (Linux engine) and Compose 2.20 or newer.
- Python 3.12+. The agent uses its own venv on Python 3.13.
- Git Bash on Windows, for the fault scripts. Set `BASH_BIN` if it is not in the default location.

### Install

```powershell
python -m venv agent\venv;      agent\venv\Scripts\pip install -r agent\requirements.txt
python -m venv mcp-server\venv; mcp-server\venv\Scripts\pip install -r mcp-server\requirements.txt
```

### Bring up the testbed (no credentials needed)

```powershell
$env:EXPERIMENT_STACK_FILE = "experiments/fixture_stack.json"
python -m experiments.stack build
python -m experiments.stack up
python -m experiments.runner --wait-ready 300
python -m experiments.stack down
```

### Run one investigation offline (no API key, no cost)

```powershell
python -m experiments.fake_model            # scripted OpenAI-compatible endpoint on :18999 (separate terminal)
$env:MODEL_BASE_URL = "http://127.0.0.1:18999/v1"; $env:MODEL_ID = "offline-fake"
python -m experiments.stack up
python -m experiments.runner --init-stack
python -m experiments.stack start-traffic
python -m experiments.runner --case redis_node_stopped --runs-dir experiments/runs/offline
python -m experiments.stack down
```

The run writes a manifest, a full agent trace (`*.trace.jsonl`) and the result (`*.result.json`). `python -m experiments.runner --list` shows all cases.

### Validate faults without a model (Phase 0)

On a fresh stack, after `--init-stack` and `start-traffic`:

```powershell
python -m experiments.validate_faults --gate
python -m experiments.validate_faults --summarize <gate1.json> <gate2.json>
```

### Run the multi-model evaluation (OpenRouter)

Put `OPENROUTER_API_KEY` in `.env` (see `.env.example`). Then:

```powershell
$env:EXPERIMENT_STACK_FILE = "experiments/bench_stack.json"
python -m experiments.quick_eval --plan                                     # costs and caps, no Docker
python -m experiments.quick_eval --run --arms luna,glm,gemma,nemotron
python -m experiments.quick_eval --run --prompt-variant v2 --runs-dir experiments/runs/quick-v2 `
    --cases latency_injected,latency_enricher --arms luna,nemotron
python -m experiments.quick_eval --summary
```

Create `experiments/state/batch.stop` to stop after the current group.

## Tests

```powershell
python -m unittest tests.test_experiment_stack tests.test_experiment_runner tests.test_benchmark `
    tests.test_experiment_design tests.test_quick_eval tests.test_v2_scoring
agent\venv\Scripts\python.exe -m unittest tests.test_agent_contract tests.test_agent_graph tests.test_agent_run `
    tests.test_observation_path tests.test_fake_model tests.test_mcp_tools tests.test_mcp_metrics
```

The tests cover:
- tool-budget exhaustion and duplicate calls;
- citation and confidence rules;
- malformed model replies;
- input validation and truncation in the MCP server;
- the harness lifecycle;
- spend reservations, caps and stop rules;
- the OpenRouter audit.

CI (`.github/workflows/verify.yml`) also builds the testbed images and runs its Go and producer tests.
