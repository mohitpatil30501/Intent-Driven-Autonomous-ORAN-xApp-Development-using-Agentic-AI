# Quick Start Guide

This project runs across three environments: a **GPU server** (Ollama), your **local machine** (code + services), and **LangSmith** (browser tracing). Start them in the order below.

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

Copy the example env file and fill in the blanks:

```bash
cp src/.env.example src/.env
```

Edit `src/.env`:

```env
# LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your LangSmith API key>
LANGSMITH_PROJECT="<project name shown in LangSmith>"

# Ollama (point at your GPU server)
OLLAMA_URL=http://<gpu-server-ip>:11434
OLLAMA_MODEL="llama3.1"
OLLAMA_MODEL_MAX_TOKENS=4096
OLLAMA_MODEL_MAX_TIMEOUT=120

# Structural RAG server (started in Step 4)
STRUCTURAL_RAG_URL=http://localhost:7070

# Per-module recursion limits (increase if a module says "need more steps")
ML_RECURSIVE_LIMIT=80
MAPPER_RECURSIVE_LIMIT=40
LOGIC_RECURSIVE_LIMIT=40
INTEGRATOR_RECURSIVE_LIMIT=120
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

### OrioSearch (web search for O-RAN docs — port 8000)

```bash
cd src/tools/oriosearch
docker compose up -d
```

### FlexRIC Structural RAG (pre-built index — port 7070)

The index is already built at `src/structural_rag/flexric_index/`. Start the FastAPI server:

```bash
cd src/structural_rag
FLEXRIC_INDEX_DIR=./flexric_index/index \
uvicorn server:app --host 0.0.0.0 --port 7070
```

> **Why Structural RAG instead of the old semantic search service?**
> The previous Docker-based semantic search returned raw C function bodies (100–300 lines each) which caused 3 M+ token usage in Module 2. The Structural RAG server returns compact, pre-formatted context blocks (signatures + summaries + call-graph edges) hard-capped at ~3 500 characters per call, cutting Module 2 token usage by ~100×.

Verify both services are up:

```bash
curl http://localhost:8000/health      # OrioSearch
curl http://localhost:7070/health      # Structural RAG
# Expected: {"status":"ok",...}
```

---

## 5. Local Machine — Start the LangGraph Dev Server

```bash
cd src
langgraph dev --no-reload
```

The server starts at `http://localhost:2024`. The graph entry point is `agent.py:graph`.

---

## 6. Browser — Open LangSmith Studio

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

**Interaction flow:**

1. **Module 1** asks clarifying questions until the blueprint is complete.
2. Type **`CONFIRM`** to approve the blueprint and continue.
3. **Module 2** maps the intent to FlexRIC service-model variables (2 Structural RAG calls).
4. The agent asks whether you have an existing dataset:
   - Type **`no`** to auto-generate synthetic data (recommended for new projects). Module 3 will generate exactly 5 000 training rows and 1 000 test rows using numpy vectorized generation.
   - Or paste an **absolute path** to your data file or directory (e.g. `/home/user/my_dataset/`). Multi-file and nested-folder datasets are supported — the profiler will discover files, filter to RAN-reportable columns (verified against the FlexRIC codebase), and map them automatically.
5. **Modules 3–6** run automatically, producing files under `src/workspace/`.

At the end of the run, check `src/workspace/ml/evaluation_report.json` (ML workflows) to see whether the model met the acceptance threshold, and `src/workspace/final_xapp.py` for the deployable xApp.

---

## Port Reference

| Service | Port | Purpose |
|---|---|---|
| LangGraph server | 2024 | Agent API + Studio backend |
| OrioSearch API | 8000 | O-RAN web search (Module 2 fallback) |
| SearXNG | 8080 | Backend for OrioSearch |
| FlexRIC Structural RAG | 7070 | FlexRIC code search (Modules 2, 6, 3b) |
| Ollama | 11434 | LLM inference (on GPU server) |

---

## Generated Output

All artifacts are written to `src/workspace/`:

```
src/workspace/
├── data/           # Mock + training CSVs (Module 3)
├── ml/             # Trained model + evaluation_report.json (Module 4)
├── logic/          # Standalone XAppLogic class (Module 5)
├── log/            # Per-module logs
├── flexric_template.py
└── final_xapp.py   # Final FlexRIC xApp (Module 6)
```

`ml/evaluation_report.json` is always written, even if Module 4 hits the recursion limit — it records the threshold, best metric achieved, and whether the threshold was met so the operator always has a result.

---

## Aborting a Run

### Graceful stop

Press `Ctrl+C` in the terminal running `langgraph dev`. This cleanly shuts down the LangGraph server. Docker containers and the Structural RAG uvicorn process are unaffected.

### If the port is still bound after a hard kill

A hard kill (`kill -9`) or a crash can leave port 2024 bound. Check and free it:

```bash
# find what's holding port 2024
lsof -i :2024

# kill it by PID
kill -9 <PID>

# or kill everything on the port in one shot
fuser -k 2024/tcp
```

### Stale workspace files after an abort

An aborted run may leave partially written files in `src/workspace/`. Always wipe the workspace before the next run:

```bash
rm -rf src/workspace/*
git checkout src/workspace/flexric_template.py   # restore the committed stub
```

### What stays running after an abort

| Process | Survives abort? | How to stop |
|---|---|---|
| LangGraph dev server (port 2024) | No — killed with Ctrl+C | `fuser -k 2024/tcp` if port stays bound |
| OrioSearch container | Yes — Docker is independent | `cd src/tools/oriosearch && docker compose down` |
| Structural RAG uvicorn (port 7070) | No — killed with Ctrl+C | `fuser -k 7070/tcp` if port stays bound |
| Ollama (GPU server) | Yes — separate SSH session | `pkill ollama` on the GPU server if needed |

> The `terminal_command` workspace tool runs subprocesses with a 120-second timeout via `subprocess.run`, so those cannot outlive the LangGraph server process itself — no orphan Python scripts to worry about.

---

## Cleanup

### Between runs — clear generated workspace artifacts

Each agent run writes into `src/workspace/`. Remove it before starting a new intent:

```bash
rm -rf src/workspace/*
git checkout src/workspace/flexric_template.py
```

---

### Stop services

```bash
cd src/tools/oriosearch && docker compose down

# Stop the Structural RAG server
fuser -k 7070/tcp
```

---

## Troubleshooting

### "Sorry, need more steps to process this request"

This is llama3.1's message when a ReAct agent hits its recursion limit. Each module now has a dedicated env var to increase the limit independently:

| Module | Env var | Default |
|---|---|---|
| Module 2 (mapper) | `MAPPER_RECURSIVE_LIMIT` | 40 |
| Module 4 (ML dev) | `ML_RECURSIVE_LIMIT` | 80 |
| Module 5 (logic) | `LOGIC_RECURSIVE_LIMIT` | 40 |
| Module 6 (integrator) | `INTEGRATOR_RECURSIVE_LIMIT` | 120 |

Add the relevant variable to `src/.env` and restart `langgraph dev`.

### Module 4 produced no model but the run completed

Check `src/workspace/ml/evaluation_report.json`. If it contains `"status": "AGENT_FAILED"`, the agent hit its limit before training. Increase `ML_RECURSIVE_LIMIT` (e.g. to `120`) and re-run. The `threshold_met` field and best metric value will be in the report even on partial runs.

### Structural RAG server returns connection error

Verify the server is running:

```bash
curl http://localhost:7070/health
```

If it is not running, start it:

```bash
cd src/structural_rag
FLEXRIC_INDEX_DIR=./flexric_index/index \
uvicorn server:app --host 0.0.0.0 --port 7070
```

If the index directory is missing, rebuild it (takes a few minutes):

```bash
cd src/structural_rag
python pipeline.py build \
    --repo ./flexric \
    --out  ./flexric_index \
    --summarizer tfidf
```
