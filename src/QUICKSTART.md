# Quick Start Guide

This project runs across three environments: a **GPU server** (Ollama), your **local machine** (code + Docker services), and **LangSmith** (browser tracing). Start them in the order below.

---

## 1. GPU Server — Start Ollama

SSH into the GPU server and ensure Ollama is running with your chosen model:

```bash
ollama serve                        # starts the API on port 11434
ollama pull llama3.1                # or whichever model you intend to use
```

Note the server's IP address (e.g. `http://10.x.x.x:11434`). You'll need it for the `.env` file.

---

## 2. Local Machine — Configure Environment

Create `src/.env` with the variables below:

```env
# LangSmith tracing (optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your LangSmith API key>
LANGSMITH_PROJECT="<project name shown in LangSmith>"

# Ollama (point at your GPU server)
OLLAMA_URL=http://<gpu-server-ip>:11434
OLLAMA_MODEL="llama3.1"
OLLAMA_MODEL_MAX_TOKENS=4096
OLLAMA_MODEL_MAX_TIMEOUT=120

# Semantic Search service (started in Step 4)
SEMANTIC_SEARCH_URL=http://localhost:7080

# OrioSearch (web search fallback for Module 2)
ORIOSEARCH_URL=http://localhost:8000

# Per-module recursion limits (raise if a module says "need more steps")
MAPPER_RECURSIVE_LIMIT=60
ML_RECURSIVE_LIMIT=80
LOGIC_RECURSIVE_LIMIT=40
INTEGRATOR_RECURSIVE_LIMIT=120
RECURSIVE_LIMIT=40
```

---

## 3. Local Machine — Install Python Dependencies

From the repo root:

```bash
pip install langgraph-cli langchain-ollama langchain-community python-dotenv
pip install -r requirements.txt
```

---

## 4. Local Machine — Start Required Services

The two convenience scripts under `src/tools/` start and stop both stacks at once:

```bash
cd src/tools
./start_tools.sh        # docker compose up -d (oriosearch + semantic_search)
# ./stop_tools.sh       # docker compose stop  (when you're done)
```

Or start them individually:

### OrioSearch (web search for O-RAN docs — port 8000)

```bash
cd src/tools/oriosearch
docker compose up -d
```

### Semantic Search service (port 7080, ChromaDB on 7000)

```bash
cd src/tools/semantic_search
docker compose up -d --build
```

> **First start takes several minutes.** On startup, the `semantic-api` container clones the FlexRIC repository (per `src/tools/semantic_search/repos.yml`), chunks `.c/.h/.cpp/.hpp/.py` files, and embeds them into ChromaDB using `all-MiniLM-L6-v2`. Track progress with:
> ```bash
> curl http://localhost:7080/status   # is_ready: false → true when done
> ```
> Modules 2, 3, and 6 will return a "system is building the knowledge graph" message until ingestion finishes.

Verify both services once ready:

```bash
curl http://localhost:8000/health      # OrioSearch
curl http://localhost:7080/status      # Semantic Search — wait for "is_ready": true
curl -X POST http://localhost:7080/semantic_search \
     -H "Content-Type: application/json" \
     -d '{"query": "kpm_ind_msg_t", "n_results": 1}'
```

---

## 5. Local Machine — Start the LangGraph Dev Server

```bash
cd src
langgraph dev --no-reload
```

The server starts at `http://localhost:2024`. The graph entry point is `agent.py:graph`.

---

## 6. Browser — Open LangSmith / LangGraph Studio

Go to [https://smith.langchain.com](https://smith.langchain.com) and log in.

- Your project traces appear under the name set in `LANGSMITH_PROJECT`.
- To use the interactive graph UI, open **LangGraph Studio** and connect it to `http://localhost:2024`.

> In LangGraph Studio, set:
> - **API URL**: `http://localhost:2024`
> - **Assistant ID**: `agent`

---

## 7. Run the Agent

In LangGraph Studio (or via API), start a new thread and describe your xApp intent, for example:

```
I need an xApp that monitors per-slice throughput and adjusts PRB allocation
when a slice drops below 10 Mbps, to prevent slice starvation.
```

**Interaction flow (3 human-in-the-loop interrupts):**

1. **Module 1** asks clarifying questions until the blueprint is complete.
2. Type **`CONFIRM`** to approve the blueprint and continue.
3. **Module 2** maps the intent to FlexRIC C-structs via Semantic Search and produces a hierarchical `Telemetry_Variables` schema.
4. The agent prints the validated telemetry schema and asks how to source the data. Reply with one of:
   - **`no`** (or "Generate all synthetic data") — Module 3 synthesizes everything: 100–500 streaming JSON items, 5 000 training rows, 1 000 test rows.
   - **An absolute path** (e.g. `/home/user/my_dataset/`) — Module 3 discovers files, pre-filters columns, RAN-validates additional columns against the FlexRIC codebase, and splits 80/20 train/test.
   - **A mix** — e.g. `Use /path/to/ml_data/ for training/testing, but synthesize streaming data`. Module 3 handles both halves in one pass.
5. **Modules 3–6** run automatically, producing files under `src/workspace/`.
6. The agent asks **"Do you want to proceed with deploying to the testbed?"**
   - Type **`Proceed`** to run Module 7 — copies artifacts into the testbed Docker stack, rebuilds the xApp container, runs it for 20 seconds, and reports the captured container logs.
   - Type anything else to end without deploying.

At the end of the run, check `src/workspace/ml/evaluation_report.json` (ML workflows) for whether the model met the threshold, and `src/workspace/final_xapp.py` for the deployable xApp.

---

## Port Reference

| Service | Port | Purpose |
|---|---|---|
| LangGraph server | 2024 | Agent API + Studio backend |
| OrioSearch API | 8000 | O-RAN web search (Module 2 fallback) |
| SearXNG | 8080 | Backend for OrioSearch |
| Semantic Search API | 7080 | FlexRIC code search (Modules 2, 3, 6) |
| ChromaDB | 7000 | Vector store backing Semantic Search |
| Ollama | 11434 | LLM inference (on GPU server) |

---

## Generated Output

All artifacts are written to `src/workspace/`:

```
src/workspace/
├── data/                          # Module 3
│   ├── build_streaming_datasets.py
│   ├── pre_filter.py              # only when profiling user paths
│   ├── streaming_mock_data.json   # hierarchical, matches Telemetry_Variables
│   ├── historical_training_data.csv   # ML workflows
│   └── test_data.csv                  # ML workflows
├── ml/                            # Module 4
│   ├── train.py                   # pre-written wrapper around auto_train
│   ├── saved_model.pkl
│   └── evaluation_report.json     # always present, even on failure
├── logic/                         # Module 5
│   └── core_logic.py
├── log/                           # Per-module logs
├── testbed/                       # Module 7 (only if deployed)
│   └── nearrtric/...
├── flexric_template.py
└── final_xapp.py                  # Module 6
```

`ml/evaluation_report.json` is always written, even if Module 4 hits the recursion limit — it records the threshold, best metric achieved, and whether the threshold was met so the operator always has a result.

---

## Aborting a Run

### Graceful stop

Press `Ctrl+C` in the terminal running `langgraph dev`. This cleanly shuts down the LangGraph server. Docker containers (OrioSearch, Semantic Search, Chroma) keep running.

### If the port is still bound after a hard kill

A hard kill (`kill -9`) or a crash can leave port 2024 bound. Check and free it:

```bash
lsof -i :2024
fuser -k 2024/tcp        # kill everything on that port
```

### Stale workspace files after an abort

An aborted run may leave partially written files in `src/workspace/`. Wipe everything except the committed template before the next run:

```bash
rm -rf src/workspace/*
git checkout src/workspace/flexric_template.py   # restore the committed stub
```

### What stays running after an abort

| Process | Survives abort? | How to stop |
|---|---|---|
| LangGraph dev server (port 2024) | No — killed with Ctrl+C | `fuser -k 2024/tcp` if port stays bound |
| OrioSearch container | Yes — Docker is independent | `cd src/tools/oriosearch && docker compose down` |
| Semantic Search + Chroma containers | Yes — Docker is independent | `cd src/tools/semantic_search && docker compose down` |
| Ollama (GPU server) | Yes — separate SSH session | `pkill ollama` on the GPU server if needed |
| Module 7 testbed (`nearrtric` stack) | No — Module 7 always runs `docker compose down` after its 20 s observation window | `cd src/workspace/testbed/nearrtric && docker compose down` if a forced stop left it up |

> The `terminal_command` workspace tool runs subprocesses with a 120-second timeout via `subprocess.run`, so those cannot outlive the LangGraph server process itself — no orphan Python scripts to worry about.

---

## Cleanup

### Between runs — clear generated workspace artifacts

Each agent run writes into `src/workspace/`. Remove it before starting a new intent:

```bash
rm -rf src/workspace/*
git checkout src/workspace/flexric_template.py
```

### Stop services

```bash
cd src/tools
./stop_tools.sh
```

Or individually:

```bash
cd src/tools/oriosearch && docker compose down
cd src/tools/semantic_search && docker compose down
```

---

## Troubleshooting

### "Sorry, need more steps to process this request"

This is llama3.1's message when a ReAct agent hits its recursion limit. Each module has a dedicated env var:

| Module | Env var | Default |
|---|---|---|
| Module 2 (mapper) | `MAPPER_RECURSIVE_LIMIT` | 60 |
| Module 3 (data engineer) | `RECURSIVE_LIMIT` | 40 |
| Module 4 (ML dev) | `ML_RECURSIVE_LIMIT` | 80 |
| Module 5 (logic) | `LOGIC_RECURSIVE_LIMIT` | 40 |
| Module 6 (integrator) | `INTEGRATOR_RECURSIVE_LIMIT` | 120 |

Add the relevant variable to `src/.env` and restart `langgraph dev`. Module 6 also surfaces a clear message — `"Set INTEGRATOR_RECURSIVE_LIMIT > N and re-run"` — when it hits the limit.

### Module 4 produced no model but the run completed

Check `src/workspace/ml/evaluation_report.json`. If it contains `"status": "AGENT_FAILED"`, the agent hit its limit before training. Increase `ML_RECURSIVE_LIMIT` (e.g. to `120`) and re-run. The `threshold_met` field and best metric value will be in the report even on partial runs.

Note that `ml/train.py` is **pre-written deterministically** by the orchestrator (it's a thin wrapper around `module_4/auto_train.py`), so a script crash usually means Module 3 produced bad data, not a training-script bug — re-running won't help; check `data/` first.

### Semantic Search returns "system is currently building the AI Knowledge Graph"

The container is still ingesting FlexRIC. Watch progress:

```bash
curl http://localhost:7080/status     # is_ready: false → true when done
docker logs -f semantic_search-semantic-api-1
```

First-run ingestion typically takes 5–15 minutes depending on hardware.

### Semantic Search server returns connection error

Verify the stack is up:

```bash
curl http://localhost:7080/status
docker compose -f src/tools/semantic_search/docker-compose.yml ps
```

If it is not running:

```bash
cd src/tools/semantic_search
docker compose up -d --build
```

If you need to wipe and re-ingest the index, take down the volumes:

```bash
cd src/tools/semantic_search
docker compose down -v
docker compose up -d --build
```

### Module 7 reports an empty log summary

Module 7 expects a pre-existing `src/workspace/testbed/nearrtric/` Docker Compose stack containing a `Dockerfile.xapp`. If that directory is missing, the deployer will skip the build and the captured logs will be empty. Provision the testbed directory before answering `Proceed` at the deploy prompt.
