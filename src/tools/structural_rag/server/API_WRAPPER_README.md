# FlexRIC RAG — LLM API Wrapper

**A FastAPI server that exposes the FlexRIC Structural RAG pipeline as an HTTP tool, callable by any LLM (OpenAI, Anthropic Claude, LangChain, and others).**

---

## Overview

This wrapper turns the `StructuralRetriever` into a network service with two endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /retrieve` | Full retrieval — returns metadata, scores, call-graph info, and a formatted context block |
| `POST /context` | Lightweight — returns only the pre-formatted context string for direct prompt injection |
| `GET /schema` | Returns OpenAI-compatible tool definitions for LLM function-calling |
| `GET /health` | Liveness check |

The `GET /schema` response is designed to be passed directly to the `tools=` parameter of any major LLM SDK.

---

## Quick Start

### 1. Build the index first

```bash
python pipeline.py build \
    --repo /path/to/flexric \
    --out  ./flexric_index \
    --summarizer tfidf        # or ollama, hf, llamacpp
```

### 2. Install server dependencies

```bash
pip install -r requirements-server.txt
```

### 3. Start the server

```bash
# Minimal — no auth
FLEXRIC_INDEX_DIR=./flexric_index/index uvicorn server:app --host 0.0.0.0 --port 8000

# With optional bearer-token auth
FLEXRIC_INDEX_DIR=./flexric_index/index \
FLEXRIC_API_KEY=mysecretkey \
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 4. Verify it's running

```bash
curl http://localhost:8000/health
# {"status":"ok","index":"./flexric_index/index"}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLEXRIC_INDEX_DIR` | `./flexric_index/index` | Path to the built index directory |
| `FLEXRIC_API_KEY` | _(empty)_ | Bearer token for auth; leave empty to disable auth |
| `FLEXRIC_MAX_TOP_K` | `20` | Hard upper limit on `top_k` to prevent runaway requests |

---

## API Reference

### `POST /retrieve`

Returns structured metadata plus a pre-formatted context block.

**Request body**

```json
{
  "query":   "how do I subscribe to KPM measurements from an xApp",
  "top_k":   8,
  "hops":    1,
  "sm_type": "KPM"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | **required** | Natural-language or symbol-based query |
| `top_k` | int | `8` | Number of chunks to return (1–20) |
| `hops` | int | `1` | Call-graph expansion depth (0 = seed only, 1 = callers/callees, 2 = two hops) |
| `sm_type` | string | auto-detect | Filter by SM: `KPM` \| `RC` \| `MAC` \| `RLC` |

**Response**

```json
{
  "query": "how do I subscribe to KPM measurements from an xApp",
  "detected_sm": "E2SM_KPM",
  "total_results": 8,
  "results": [
    {
      "chunk_id":        "examples/xApp/c/kpm_rc/xapp_kpm_rc.c::kpm_subscribe",
      "name":            "kpm_subscribe",
      "type":            "function",
      "file":            "examples/xApp/c/kpm_rc/xapp_kpm_rc.c",
      "layer":           "xapp",
      "sm_type":         "E2SM_KPM",
      "is_xapp_example": true,
      "signature":       "void kpm_subscribe(e2_node_t* node, kpm_sub_t* sub)",
      "nl_summary":      "Subscribes an xApp to KPM E2SM periodic reports from a given E2 node.",
      "calls":           ["e2ap_subscribe", "kpm_enc_action_def"],
      "called_by":       ["main"],
      "_score":          0.7431,
      "_via_graph":      "seed"
    }
  ],
  "llm_context": "### [1] kpm_subscribe  (function · xapp · E2SM_KPM)\n..."
}
```

---

### `POST /context`

Lightweight endpoint — returns only the formatted string.

**Request body**

```json
{
  "query":     "e2ap_subscribe function signature",
  "top_k":     6,
  "hops":      1,
  "max_chars": 8000
}
```

**Response**

```json
{
  "query":       "e2ap_subscribe function signature",
  "context":     "### [1] e2ap_subscribe  (function · e2ap · none)\nFile: ...",
  "chunk_count": 6
}
```

---

### `GET /schema`

Returns an OpenAI-compatible JSON array of tool definitions.

```bash
curl http://localhost:8000/schema
```

This is designed to be passed directly to `tools=` in OpenAI or Anthropic SDK calls — see [LLM Integration](#llm-integration) below.

---

## LLM Integration

### Direct HTTP (no LLM library)

```python
import requests

ctx = requests.post("http://localhost:8000/context", json={
    "query": "how do I subscribe to KPM measurements",
    "top_k": 8,
}).json()["context"]

prompt = f"""You are writing a FlexRIC xApp in C.
Use only the API shown below.

{ctx}

Task: Write an xApp that subscribes to KPM E2SM periodic reports.
"""
```

---

### OpenAI (GPT-4o, GPT-4-turbo)

```python
import json, requests
from openai import OpenAI

client = OpenAI()
tools  = requests.get("http://localhost:8000/schema").json()

messages = [
    {"role": "system", "content": "You are an expert FlexRIC xApp developer. Always call flexric_retrieve before answering code questions."},
    {"role": "user",   "content": "How do I subscribe to KPM measurements?"},
]

response = client.chat.completions.create(
    model="gpt-4o", messages=messages, tools=tools, tool_choice="auto"
)

# Execute any tool calls
msg = response.choices[0].message
messages.append(msg)

for tc in (msg.tool_calls or []):
    args = json.loads(tc.function.arguments)
    result = requests.post(f"http://localhost:8000/{tc.function.name.replace('flexric_','')}", json=args).json()
    content = result.get("llm_context") or result.get("context", "")
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

# Final grounded answer
final = client.chat.completions.create(model="gpt-4o", messages=messages)
print(final.choices[0].message.content)
```

---

### Anthropic (Claude)

```python
import anthropic, requests

client  = anthropic.Anthropic()
schema  = requests.get("http://localhost:8000/schema").json()

# Convert OpenAI format → Anthropic format
tools = [{"name": t["function"]["name"],
           "description": t["function"]["description"],
           "input_schema": t["function"]["parameters"]} for t in schema]

messages = [{"role": "user", "content": "How do I subscribe to KPM measurements?"}]

while True:
    resp = client.messages.create(
        model="claude-opus-4-5", max_tokens=4096,
        system="You are an expert FlexRIC xApp developer. Always call flexric_retrieve before answering.",
        tools=tools, messages=messages,
    )
    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason != "tool_use":
        print(next(b.text for b in resp.content if b.type == "text"))
        break

    results = []
    for b in resp.content:
        if b.type != "tool_use":
            continue
        endpoint = b.name.replace("flexric_", "")
        data = requests.post(f"http://localhost:8000/{endpoint}", json=b.input).json()
        results.append({"type": "tool_result", "tool_use_id": b.id,
                         "content": data.get("llm_context") or data.get("context", "")})
    messages.append({"role": "user", "content": results})
```

---

### LangChain

```python
from llm_integration import make_langchain_tools
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

tools  = make_langchain_tools()
llm    = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert FlexRIC xApp developer. Use flexric_retrieve for all code questions."),
    ("human",  "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent   = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result   = executor.invoke({"input": "How do I subscribe to KPM measurements?"})
print(result["output"])
```

---

## Tool Definitions

The server exposes two LLM tools:

### `flexric_retrieve`

Full retrieval with metadata. Use when the LLM needs to inspect call-graph relationships, scores, or filter by layer/SM type.

```
Input:  query (required), top_k, hops, sm_type
Output: chunk metadata + formatted context block
```

### `flexric_context`

Context-only. Use as a final step before generating code — returns a pre-formatted string ready to paste into a system prompt.

```
Input:  query (required), top_k, hops, sm_type, max_chars
Output: plain text context string
```

**When to use which:**

| Scenario | Tool |
|---|---|
| Initial exploration, need to check SM types or scores | `flexric_retrieve` |
| Single-step code generation, just need context | `flexric_context` |
| Agentic loop checking if an xApp example was returned | `flexric_retrieve` (inspect `is_xapp_example`) |

---

## Query Tips for LLMs

These query patterns work best with the hybrid (BM25 + dense) index:

```
# Intent queries — hits the NL (MiniLM) index
"how do I subscribe to KPM measurements"
"register an xApp callback for E2 indications"
"send a control message to the RIC"

# Symbol queries — hits BM25 + CodeBERT
"e2ap_subscribe function"
"kpm_enc_action_def"
"e2sm_kpm_ind_msg_t struct"

# Combined — best coverage
"how does kpm_subscribe encode the action definition"
"what calls e2ap_setup during xApp initialization"
```

---

## Security

- By default the server has **no authentication**. Set `FLEXRIC_API_KEY` to enable bearer-token auth.
- Do not expose the server to the public internet without auth or a gateway in front.
- `MAX_TOP_K` caps the `top_k` parameter to prevent abuse; adjust with `FLEXRIC_MAX_TOP_K`.

---

## Files

```
flexric_rag/
├── server.py                  ← FastAPI server (this wrapper)
├── llm_integration.py         ← OpenAI / Anthropic / LangChain examples
├── requirements-server.txt    ← Server-only dependencies
└── retrieval/
    └── retriever.py           ← Core retriever (unchanged)
```

---

## Running in Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt -r requirements-server.txt
ENV FLEXRIC_INDEX_DIR=/data/index
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t flexric-rag-api .
docker run -p 8000:8000 \
  -v /path/to/flexric_index:/data \
  -e FLEXRIC_INDEX_DIR=/data/index \
  flexric-rag-api
```

---

## Interactive API Docs

When the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
