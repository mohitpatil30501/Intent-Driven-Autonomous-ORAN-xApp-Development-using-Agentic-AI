# FlexRIC Structural RAG

**End-to-end structural retrieval-augmented generation for the FlexRIC O-RAN codebase.**

> Structural RAG treats your codebase as a **call graph**, not a pile of text.
> Every retrieval result brings the function you asked for *plus* its callers,
> callees, and the nearest xApp example — the three things an LLM needs to
> generate correct FlexRIC xApp code without hallucinating API calls.

---

## Table of Contents

1. [Why Not Vanilla RAG?](#1-why-not-vanilla-rag)
2. [Architecture Overview](#2-architecture-overview)
3. [Quick Start](#3-quick-start)
4. [Pipeline Stages](#4-pipeline-stages)
5. [NL Summarization Without API Keys](#5-nl-summarization-without-api-keys)
6. [Configuration Reference](#6-configuration-reference)
7. [Querying](#7-querying)
8. [Project Layout](#8-project-layout)
9. [Extending the Pipeline](#9-extending-the-pipeline)
10. [FlexRIC-Specific Notes](#10-flexric-specific-notes)

---

## 1. Why Not Vanilla RAG?

| Problem with naive RAG | This pipeline's fix |
|---|---|
| Chunks cut across function boundaries | AST-based extraction: every chunk is one complete function or struct |
| No import/call context | Call graph: retrieved chunk includes its callers and callees |
| Dense embed misses exact API names | Hybrid index: BM25 catches `e2sm_kpm_ind_msg_t`, dense catches intent |
| LLM can't see a working xApp example | `is_xapp_example` boost: `examples/xApp/` always surfaces in results |
| All chunks treated equally | FlexRIC layer/SM-type metadata: xApp ↔ E2AP boundary edges flagged |

---

## 2. Architecture Overview

```
FlexRIC repo
     │
     ▼
┌─────────────────────┐
│  Stage 1: AST Parse │  tree-sitter-c / tree-sitter-python (regex fallback)
│  core/parser.py     │  → function, struct, class chunks with metadata
└────────┬────────────┘
         │
         ▼
┌──────────────────────┐
│  Stage 2: Call Graph │  bidirectional call-graph via tree-sitter call exprs
│  core/callgraph.py   │  → calls[], called_by[], boundary edges flagged
└────────┬─────────────┘
         │
         ▼
┌───────────────────────────┐
│  Stage 3: NL Summarize    │  one sentence per function (see §5 for backends)
│  summarize/summarizer.py  │  → nl_summary field on each chunk
└────────┬──────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Dual Embed + Index  (index/builder.py)            │
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────┐  │
│  │ CodeBERT    │   │ MiniLM NL   │   │  BM25 (lexical)  │  │
│  │ code_index  │   │ nl_index    │   │  bm25.pkl        │  │
│  │ .faiss      │   │ .faiss      │   │                  │  │
│  └─────────────┘   └─────────────┘   └──────────────────┘  │
│                   + networkx DiGraph (graph.pkl)             │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 5: Query-time Retrieval  (retrieval/retriever.py)     │
│                                                              │
│   NL-FAISS + Code-FAISS + BM25                               │
│         │ Reciprocal Rank Fusion                             │
│         │ Graph expansion (1-2 hops)                        │
│         │ FlexRIC re-rank (xapp★ + sm_type boost)           │
│         ▼                                                    │
│   top-k chunks with _score, _via_graph provenance            │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Quick Start

### Prerequisites

- Python 3.10+
- FlexRIC cloned locally: `git clone https://github.com/oran-sc/flexric`
- Choose a summarizer (see §5); **no API key needed for any option**

### Install

```bash
# Clone this repo
git clone https://github.com/yourorg/flexric-rag
cd flexric-rag

# Core dependencies (always required)
pip install -r requirements.txt

# Pick ONE summarizer backend (see §5 for all options)
# Option A — Ollama (recommended, best quality)
pip install -r requirements-ollama.txt
ollama pull codellama

# Option B — Zero-dependency TF-IDF (works right now, no model needed)
# (No extra install required)
```

### Build the index

```bash
# With Ollama summarizer (recommended)
python pipeline.py build \
    --repo /path/to/flexric \
    --out  ./flexric_index \
    --summarizer ollama \
    --model codellama

# With TF-IDF fallback (instant, no model required)
python pipeline.py build \
    --repo /path/to/flexric \
    --out  ./flexric_index \
    --summarizer tfidf
```

### Query

```bash
# Single query
python pipeline.py query \
    --index ./flexric_index \
    --q "how do I subscribe to KPM measurements"

# Filter by SM type
python pipeline.py query \
    --index ./flexric_index \
    --q "encode action definition" \
    --sm KPM

# Interactive REPL
python pipeline.py repl --index ./flexric_index
```

### Use in Python

```python
from retrieval.retriever import StructuralRetriever

retriever = StructuralRetriever("./flexric_index/index")

results = retriever.retrieve(
    "how do I subscribe to KPM measurements",
    top_k=8,
    hops=1,
)

# Build a prompt context block for an LLM
context = retriever.build_context(results)
print(context)
```

---

## 4. Pipeline Stages

### Stage 1 — AST Parsing (`core/parser.py`)

Extracts **semantically complete units** from every `.c`, `.h`, and `.py` file:

| Chunk type | Content | When |
|---|---|---|
| `function` | Full function source + signature | Always |
| `struct` | Struct/union/enum definition | C headers |
| `class` | Python class body | Python files |
| `file_summary` | File path + list of function names | One per file |

Each chunk carries rich metadata:

```python
{
  "chunk_id": "examples/xApp/c/kpm_rc/xapp_kpm_rc.c::kpm_subscribe",
  "type":      "function",
  "name":      "kpm_subscribe",
  "signature": "void kpm_subscribe(e2_node_t* node, kpm_sub_t* sub)",
  "body":      "... full source ...",
  "file":      "examples/xApp/c/kpm_rc/xapp_kpm_rc.c",
  "layer":     "xapp",           # xapp | ric | sm | api | util
  "sm_type":   "E2SM_KPM",       # E2SM_KPM | E2SM_RC | E2SM_MAC | E2SM_RLC | none
  "is_xapp_example": True,
  "module":    "kpm_rc",
  "calls":     [],               # filled in Stage 2
  "called_by": [],
  "nl_summary": "",              # filled in Stage 3
}
```

**tree-sitter** (preferred) is used when available; a regex fallback handles the case where it is not installed.

---

### Stage 2 — Call Graph (`core/callgraph.py`)

Builds a **bidirectional directed call graph** across all extracted functions.

- Resolves call names to `chunk_id`s, preferring same SM type when names collide.
- Flags **boundary edges** crossing the xApp ↔ E2AP interface — the highest-hallucination-risk API calls.
- Output: `callgraph.json` with adjacency dict + edge list.

---

### Stage 3 — NL Summarization (`summarize/summarizer.py`)

Generates one English sentence per function describing what it does in O-RAN terms.
This sentence is embedded by the NL model to handle **intent-based queries**.

→ See **§5** for all backends (none require an API key).

---

### Stage 4 — Dual Embed + Index (`index/builder.py`)

| Index | Model | Best for |
|---|---|---|
| `nl_index.faiss` | all-MiniLM-L6-v2 | Intent queries ("how to subscribe to…") |
| `code_index.faiss` | CodeBERT (or MiniLM fallback) | Symbol queries ("e2ap_subscribe") |
| `bm25.pkl` | BM25Okapi | Exact identifier matching |
| `graph.pkl` | networkx DiGraph | Structural expansion |

---

### Stage 5 — Query-time Retrieval (`retrieval/retriever.py`)

```
Query
  │
  ├─ NL embed  → nl_index.search()   → top-20 indices
  ├─ Code embed → code_index.search() → top-20 indices
  └─ BM25 score → argsort            → top-20 indices
            │
            └── Reciprocal Rank Fusion → merged seed set
                        │
                        └── Graph expansion (hops=1 or 2)
                                  │ successors (callees)
                                  │ predecessors (callers)
                                  │
                                  └── Re-rank with boosts:
                                        +0.30  is_xapp_example
                                        +0.20  sm_type match
                                        +0.10  layer == "api"
                                              │
                                              └── top-k results
```

---

## 5. NL Summarization Without API Keys

This is the most common question when deploying on-premise. Here are all options — **none require an API key or internet access after initial model download**.

---

### Option A — Ollama (Recommended)

Run any open-weights model locally via the [Ollama](https://ollama.com) daemon.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (choose based on your hardware)
ollama pull codellama       # best for C code    | 4 GB RAM
ollama pull phi3:mini       # fastest            | 2 GB RAM
ollama pull llama3:8b       # best general       | 5 GB RAM
ollama pull mistral:7b      # balanced           | 4 GB RAM

# Build with Ollama
python pipeline.py build --repo /path/to/flexric --summarizer ollama --model codellama
```

**Model selection guide:**

| Model | RAM needed | Speed | Code quality | Notes |
|---|---|---|---|---|
| `codellama:7b` | 4 GB | medium | ⭐⭐⭐⭐⭐ | Best choice for C/telecom |
| `phi3:mini` | 2 GB | fast | ⭐⭐⭐⭐ | Great quality/size ratio |
| `llama3:8b` | 5 GB | medium | ⭐⭐⭐⭐ | Strong instruction following |
| `mistral:7b` | 4 GB | medium | ⭐⭐⭐⭐ | Good all-rounder |
| `starcoder2:3b` | 2 GB | fast | ⭐⭐⭐⭐ | Pure code focus |

---

### Option B — HuggingFace Local Transformers

Download and run models directly in Python. No daemon, no external process.

```bash
pip install -r requirements-hf.txt

# Small, fast (0.9 GB, runs on CPU):
python pipeline.py build --repo /path/to/flexric \
    --summarizer hf \
    --model Salesforce/codet5p-220m-bimodal

# Larger, better quality (5 GB, GPU recommended):
python pipeline.py build --repo /path/to/flexric \
    --summarizer hf \
    --model microsoft/phi-2
```

**Recommended HuggingFace models:**

| Model | Size | Notes |
|---|---|---|
| `Salesforce/codet5p-220m-bimodal` | 0.9 GB | Code↔NL, fast on CPU |
| `microsoft/phi-2` | 5 GB | Strong reasoning, GPU preferred |
| `bigcode/starcoder2-3b` | 6 GB | Code-specialised, best for C |
| `google/flan-t5-large` | 3 GB | Instruction tuned, seq2seq |

---

### Option C — llama-cpp-python (GGUF, no daemon)

Run quantised models entirely within Python. No Ollama server required.
Best when you can't run background daemons (e.g., inside CI, Docker).

```bash
# CPU build
pip install llama-cpp-python

# GPU build (CUDA)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall

# Download a GGUF from HuggingFace (example)
wget https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf

python pipeline.py build --repo /path/to/flexric \
    --summarizer llamacpp \
    --model /path/to/codellama-7b-instruct.Q4_K_M.gguf
```

---

### Option D — GPT4All

Desktop-friendly runner that auto-downloads models.

```bash
pip install gpt4all

python pipeline.py build --repo /path/to/flexric \
    --summarizer gpt4all \
    --model "Phi-3-mini-4k-instruct.Q4_0.gguf"
```

---

### Option E — TF-IDF (Zero dependencies, instant)

No model download, no GPU, no API — runs in seconds.
Extracts the most statistically distinctive tokens from each function
body and combines them with layer/SM-type metadata.

Quality is lower than LLM summaries, but the NL index still works well
for BM25 retrieval and the code-FAISS index is unaffected.

**Use this when:** bootstrapping quickly, running in CI, or first testing the pipeline.

```bash
# No extra install needed
python pipeline.py build --repo /path/to/flexric --summarizer tfidf
```

---

### Comparison

| Backend | Quality | Speed | RAM | GPU | Install complexity |
|---|---|---|---|---|---|
| Ollama (codellama) | ⭐⭐⭐⭐⭐ | medium | 4 GB | optional | `curl + ollama pull` |
| HuggingFace (phi-2) | ⭐⭐⭐⭐ | slow | 5 GB | recommended | `pip install transformers torch` |
| llama-cpp (Q4 GGUF) | ⭐⭐⭐⭐ | medium | 3 GB | optional | `pip install llama-cpp-python` |
| GPT4All | ⭐⭐⭐ | medium | 3 GB | no | `pip install gpt4all` |
| TF-IDF | ⭐⭐ | instant | <100 MB | no | (included) |
| None | ⭐ | instant | none | no | (included) |

---

## 6. Configuration Reference

```bash
python pipeline.py build \
    --repo       /path/to/flexric   # FlexRIC root directory (required)
    --out        ./flexric_index    # Output index directory
    --summarizer ollama             # ollama | hf | llamacpp | gpt4all | tfidf | none
    --model      codellama          # Model name or path (backend-specific)
    --force                         # Rebuild all stages (ignores cache)
```

```bash
python pipeline.py query \
    --index ./flexric_index   # Index directory (required)
    --q     "your query"      # Query string (required)
    --top   8                 # Number of results (default: 8)
    --hops  1                 # Graph expansion depth (1 or 2)
    --sm    KPM               # Filter by SM type: KPM | RC | MAC | RLC
    --json                    # Output as JSON
```

---

## 7. Querying

### In Python (for LLM integration)

```python
from retrieval.retriever import StructuralRetriever

retriever = StructuralRetriever("./flexric_index/index")

# Get relevant chunks
results = retriever.retrieve(
    query  = "how do I subscribe to KPM measurements from an xApp",
    top_k  = 8,
    hops   = 1,        # include 1-hop callers and callees
    sm_type= "E2SM_KPM",
)

# Build a context block for your LLM
context = retriever.build_context(results, max_tokens=6000)

# Inject into your prompt
prompt = f"""You are writing a FlexRIC xApp in C.
Use only the API shown in the context below.

{context}

Task: Write an xApp that subscribes to KPM E2SM measurements and logs the results.
"""
```

### Query patterns that work well

```
# Intent-based (hits NL index)
"how do I subscribe to KPM measurements"
"send a control message to the RIC"
"register an xApp callback for E2 indications"

# Symbol-based (hits code + BM25 index)
"e2ap_subscribe function"
"kpm_enc_action_def"
"e2sm_kpm_ind_msg_t struct"

# Combined
"how does kpm_subscribe encode the action definition"
"what calls e2ap_setup during xApp initialization"
```

---

## 8. Project Layout

```
flexric_rag/
├── pipeline.py              # CLI entry point — build / query / repl
│
├── core/
│   ├── parser.py            # Stage 1: AST chunk extraction (tree-sitter + regex)
│   └── callgraph.py         # Stage 2: Bidirectional call graph builder
│
├── index/
│   └── builder.py           # Stage 4: Dual embed + FAISS + BM25 + graph
│
├── retrieval/
│   └── retriever.py         # Stage 5: Hybrid retrieve + RRF + graph expand
│
├── summarize/
│   └── summarizer.py        # Stage 3: NL summaries (6 backends, no API needed)
│
├── utils/
│   ├── logger.py
│   └── flexric_tags.py      # FlexRIC-specific layer/SM classification
│
├── requirements.txt                # Core dependencies
├── requirements-ollama.txt         # Ollama backend
├── requirements-hf.txt             # HuggingFace backend
├── requirements-llamacpp.txt       # llama-cpp-python backend
└── requirements-gpt4all.txt        # GPT4All backend
```

### Index directory layout (after build)

```
flexric_index/
├── chunks.jsonl             # All extracted chunks (cached, re-used across builds)
├── callgraph.json           # Adjacency dict + boundary edges
└── index/
    ├── nl_index.faiss       # NL dense index (MiniLM)
    ├── code_index.faiss     # Code dense index (CodeBERT)
    ├── bm25.pkl             # BM25 lexical index
    ├── graph.pkl            # networkx DiGraph
    ├── meta.json            # Chunk metadata (no vectors)
    └── config.json          # Build configuration
```

---

## 9. Extending the Pipeline

### Add a new summarizer backend

Subclass `BaseSummarizer` and add a factory entry:

```python
# summarize/summarizer.py

class MyBackend(BaseSummarizer):
    def summarize_one(self, chunk: dict) -> str:
        # call your local model / service
        return "one-sentence summary here"

# In SummarizerFactory.create():
if b == "mybackend":
    return MyBackend()
```

### Add a new file type

Implement an extractor with `.extract(filepath, repo_root) -> List[Dict]` and
register it in `CodebaseParser.__init__`.

### Swap the dense model

Change `"all-MiniLM-L6-v2"` in `index/builder.py` and `retrieval/retriever.py`
to any sentence-transformers model. Larger models (e.g., `bge-large-en-v1.5`)
improve retrieval quality at the cost of build time.

---

## 10. FlexRIC-Specific Notes

### Critical tags

| Tag | Path pattern | Why it matters |
|---|---|---|
| `is_xapp_example=True` | `examples/xApp/**` | End-to-end working templates; always surface in results |
| `sm_type=E2SM_KPM` | `src/sm/kpm_sm/**` | Subscription & report encoding |
| `sm_type=E2SM_RC` | `src/sm/rc_sm/**` | Control action encoding |
| `layer=e2ap` | `src/e2ap/**` | API compliance — highest hallucination risk |
| `layer=ric` | `src/ric/**` | How the RIC handles xApp registration |

### xApp ↔ E2AP boundary

The call graph explicitly flags edges crossing this boundary (tagged
`edge_type=xapp_to_e2ap`). These are the functions where LLMs hallucinate
the most. Ensure that any LLM prompt context includes at least one
`is_xapp_example=True` chunk from the same `sm_type` as the query.

### Avoiding hallucination

The single most effective constraint is:
> Always include one `is_xapp_example=True` chunk with matching `sm_type`.

The retriever enforces this via the re-ranking boost (+0.30 for xApp examples,
+0.20 for matching SM type). You can make it stricter by filtering in Python:

```python
xapp_results = [r for r in results if r.get("is_xapp_example")]
if not xapp_results:
    # force-include the top xApp example for the detected SM type
    xapp_results = retriever.retrieve(query, sm_type=detected_sm)
    xapp_results = [r for r in xapp_results if r.get("is_xapp_example")][:1]
```

---

## License

Apache 2.0 — see `LICENSE`.
