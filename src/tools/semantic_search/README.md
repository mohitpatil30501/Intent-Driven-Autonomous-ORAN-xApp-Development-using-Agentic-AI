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
└── app.py
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
1. **Cloning**: The API will read `repos.yml` and run `git clone` (or `git pull` if it already exists) into the `/app/cloned_repos` volume.
2. **Chunking & Embedding**: It will parse all `.c`, `.h`, `.cpp`, and `.py` files, chunk them, and pass them through a HuggingFace embedding model (`all-MiniLM-L6-v2`).
3. **Indexing**: Vectors are saved to the ChromaDB container.

*(⚠️ **Note:** The very first time you run this, it may take 2-5 minutes to download the embedding model, clone the repos, and embed the codebase. Subsequent restarts will be much faster.)*

To monitor the ingestion progress, check the logs:
```bash
docker-compose logs -f semantic-api
```
Look for the message: `Ingestion Complete!`

---

## 📡 4. API Endpoints

The API runs on `http://localhost:7080`. You can test it from your terminal:

**1. Semantic Search** (Finds code based on meaning/intent)
```bash
curl -X POST "http://localhost:7080/semantic_search" \
     -H "Content-Type: application/json" \
     -d '{"query": "how is slice throughput calculated", "n_results": 3}'
```

**2. Exact Keyword Search** (Extremely fast regex/keyword match via `ripgrep`)
```bash
curl -X POST "http://localhost:7080/exact_search" \
     -H "Content-Type: application/json" \
     -d '{"query": "dl_aggr_tbs", "n_results": 5}'
```

---

## 🤖 5. LangGraph Agent Integration

To connect **Module 2 (Technical Template Completion)** to this engine, remove your old Sourcegraph and GitLab tools. Add these two tools to your Python agent script:

```python
import requests
from langchain_core.tools import tool

SEARCH_ENGINE_URL = "http://localhost:7080" # Update if running on a different IP

@tool
def semantic_code_search(nl_query: str, max_results: int = 3) -> str:
    """
    Use this to find code based on intent or concepts. 
    Example: 'Where are MAC Service Model variables defined?'
    """
    try:
        res = requests.post(
            f"{SEARCH_ENGINE_URL}/semantic_search", 
            json={"query": nl_query, "n_results": max_results}
        )
        return res.json().get("results", "No results found.")
    except Exception as e:
        return f"Error connecting to search engine: {e}"

@tool
def exact_keyword_search(keyword: str, max_results: int = 5) -> str:
    """
    Use this to find EXACT references to a specific C-struct, variable, or function name.
    Example: 'dl_aggr_tbs' or 'mac_ind_data'
    """
    try:
        res = requests.post(
            f"{SEARCH_ENGINE_URL}/exact_search", 
            json={"query": keyword, "n_results": max_results}
        )
        return res.json().get("results", "No matches found.")
    except Exception as e:
        return f"Error connecting to search engine: {e}"

# Add them to your tool list for Module 2
tools = [semantic_code_search, exact_keyword_search, ...]
```

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