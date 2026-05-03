# 🧠 O-RAN Code Search Engine (Semantic + Exact Match)

This microservice replaces heavy enterprise tools (like local GitLab + Sourcegraph) with a lightweight, AI-native search engine tailored for your LangGraph agents. 

It automatically clones target repositories (like FlexRIC or OAI), chunks the source code (C, C++, Python), embeds it into a local **ChromaDB Vector Database**, and exposes a fast API for both **Semantic Search** (intent-based) and **Exact Search** (`ripgrep`).

---

## 📂 1. Directory Setup
Create a new folder named `semantic_search/` in your project root and place the 5 files provided previously inside it:
```text
semantic_search/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── repos.yml
├── app.py
└── semantic_search_tool.py  # Python wrapper for agents
```

---

## ⚙️ 2. Configuration (`repos.yml`)
Before running, define the repositories you want the AI to learn. Open `repos.yml` and add your targets. 

*Note: The container requires internet access to clone these on startup. You can use standard HTTPS `.git` URLs.*

```yaml
repositories:
  - name: "flexric"
    url: "https://gitlab.eurecom.fr/mosaic5g/flexric.git"
    branch: "dev"
  # Add more repositories here if needed:
  # - name: "ric-plt"
  #   url: "https://gerrit.o-ran-sc.org/r/ric-plt/lib/rmr"
```

---

## 🚀 3. Build & Run

Ensure you have Docker and Docker Compose installed. Open your terminal, navigate to the `semantic_search/` directory, and run:

```bash
docker-compose up -d --build
```

### What happens during startup?
The API server boots **instantly** and runs a background thread to build the AI Knowledge Graph:
1. **Cloning**: It reads `repos.yml` and clones repositories into the `/app/cloned_repos` volume.
2. **Chunking & Embedding**: It parses `.c`, `.h`, `.cpp`, and `.py` files, chunks them, and passes them through a local HuggingFace embedding model (`all-MiniLM-L6-v2`).
3. **Indexing**: Vectors are pushed to the ChromaDB container.

*(⚠️ **Note:** The very first time you run this, it may take 10-20 minutes to embed a large C/C++ codebase purely on CPU. During this time, the search endpoints will return a JSON notice asking you to wait.)*

### 📊 Monitoring Ingestion Progress
To see exactly how many chunks have been processed, you can hit the status endpoint:
```bash
curl http://localhost:7080/status
```

If the CPU is too overloaded to respond to the HTTP request, you can instantly read the internal log file directly from the container:
```bash
docker exec semantic_search-semantic-api-1 cat /app/ingestion.log
```

---

## 📡 4. API Endpoints

The API runs on `http://localhost:7080`. You can test it from your terminal:

**1. Semantic Search** (Finds code based on meaning/intent)
```bash
# Default: returns truncated snippets (800 chars) to save tokens
curl -X POST "http://localhost:7080/semantic_search" \
     -H "Content-Type: application/json" \
     -d '{"query": "how is slice throughput calculated", "n_results": 3}'

# Detailed: returns full code bodies
curl -X POST "http://localhost:7080/semantic_search" \
     -H "Content-Type: application/json" \
     -d '{"query": "how is slice throughput calculated", "return_full_text": true}'
```

**2. Exact Keyword Search** (Extremely fast regex/keyword match via `ripgrep`)
```bash
curl -X POST "http://localhost:7080/exact_search" \
     -H "Content-Type: application/json" \
     -d '{"query": "dl_aggr_tbs", "n_results": 5}'
```

---

## 🤖 5. LangGraph Agent Integration

Use the `semantic_search_tool.py` wrapper to provide two levels of search to your agent. This manages token usage by separating broad exploration from detailed analysis.

### Usage in Agent Modules

```python
from tools.semantic_search.semantic_search_tool import (
    semantic_search_summary,   # Use for broad intent lookup (truncated)
    semantic_search_detailed,  # Use ONLY for deep dive into specific functions (full text)
)

# Add to your tool list alongside Structural RAG
tools = [flexric_rag_context, semantic_search_summary, semantic_search_detailed, ...]
```

### Why two tools?

- **Summary Tool**: Limits every result to 800 characters. This is the "safest" way to search without hitting context window limits (prevents 3M+ token spikes).
- **Detailed Tool**: Returns the entire function body. Agents are instructed to use this only after finding a relevant filename via the summary tool or structural RAG.

---

## 🛑 6. Shutting Down & Data Management

To stop the services:
```bash
docker-compose down
```

**Resetting the Database**:
If you change branches, add new repositories, or want to force a clean re-embedding of the codebase, you must delete the Docker volumes before restarting:
```bash
docker-compose down -v
docker-compose up -d --build
```