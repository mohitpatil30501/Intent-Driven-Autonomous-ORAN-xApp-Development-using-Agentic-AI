"""
LangChain tool wrappers for the FlexRIC Structural RAG API.

The server (src/structural_rag/server.py) must be running before these tools
are used.  Default port: 7070 (set STRUCTURAL_RAG_URL to override).

Start command (from src/structural_rag/):
    FLEXRIC_INDEX_DIR=./flexric_index/index \
    uvicorn server:app --host 0.0.0.0 --port 7070
"""

import os
import requests
from langchain_core.tools import tool

STRUCTURAL_RAG_URL = os.getenv("STRUCTURAL_RAG_URL", "http://localhost:7070")
_DEFAULT_MAX_CHARS = 3500   # hard cap per call; keeps each result under ~900 tokens
_DEFAULT_TOP_K = 6
_DEFAULT_HOPS = 1


def _post(endpoint: str, payload: dict) -> str:
    """Shared HTTP helper; returns the result string or an error message."""
    try:
        res = requests.post(
            f"{STRUCTURAL_RAG_URL}/{endpoint}",
            json=payload,
            timeout=15,
        )
        if res.status_code == 200:
            return res.json()
        return f"Error {res.status_code}: {res.text[:300]}"
    except Exception as e:
        return f"Error connecting to FlexRIC RAG server at {STRUCTURAL_RAG_URL}: {e}"


@tool
def flexric_rag_context(
    query: str,
    sm_type: str = "",
    top_k: int = _DEFAULT_TOP_K,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """
    Retrieve a compact, pre-formatted FlexRIC code context block for use in an LLM prompt.

    Returns function signatures, one-line summaries, and call-graph edges — NOT raw code.
    This is the primary search tool. Use it for:
    - Finding telemetry struct variables (e.g., 'MAC SM indication struct variables')
    - Finding control action structs (e.g., 'MAC SM control message struct')
    - General FlexRIC API lookups

    Args:
        query:     Natural-language or symbol query (e.g. 'MAC SM ue_stats fields').
        sm_type:   Optional filter: 'MAC', 'KPM', 'RLC', or 'RC'. Leave empty to auto-detect.
        top_k:     Number of chunks (1-20, default 6).
        max_chars: Hard character cap on the returned context (default 3500 ≈ 900 tokens).

    Returns a text block with signatures, summaries, and call relationships.
    The paths in the results are informational only — do NOT try to read them with read_file.
    """
    payload = {
        "query": query,
        "top_k": top_k,
        "max_chars": max_chars,
    }
    if sm_type and sm_type.upper() in ("MAC", "KPM", "RLC", "RC"):
        payload["sm_type"] = sm_type.upper()

    result = _post("context", payload)
    if isinstance(result, dict):
        return result.get("context", "No context returned.")
    return str(result)


@tool
def flexric_rag_retrieve(
    query: str,
    sm_type: str = "",
    top_k: int = _DEFAULT_TOP_K,
    hops: int = _DEFAULT_HOPS,
) -> str:
    """
    Retrieve structured metadata from the FlexRIC codebase with call-graph context.

    Use this when you need to check: whether a specific C variable exists in FlexRIC
    (e.g., validating column names), or which SM type a function belongs to.

    For code generation prompts, prefer flexric_rag_context which returns a ready-to-use
    text block. Use flexric_rag_retrieve when you need to inspect result metadata
    (is_xapp_example, sm_type, chunk_id, calls, called_by).

    Args:
        query:   Natural-language or exact C symbol query (e.g. 'wb_cqi', 'dl_aggr_prb').
        sm_type: Optional filter: 'MAC', 'KPM', 'RLC', or 'RC'.
        top_k:   Number of chunks (1-20, default 6).
        hops:    Call-graph expansion depth (0=exact only, 1=callers+callees, default 1).

    Returns JSON-like text with chunk metadata and a pre-formatted llm_context block.
    The file paths in results are informational only — do NOT attempt to read them.
    """
    payload = {
        "query": query,
        "top_k": top_k,
        "hops": hops,
    }
    if sm_type and sm_type.upper() in ("MAC", "KPM", "RLC", "RC"):
        payload["sm_type"] = sm_type.upper()

    result = _post("retrieve", payload)
    if isinstance(result, dict):
        ctx = result.get("llm_context", "")
        total = result.get("total_results", 0)
        detected = result.get("detected_sm", "")
        return (
            f"detected_sm: {detected}  total_results: {total}\n\n"
            f"{ctx[:_DEFAULT_MAX_CHARS]}"
        )
    return str(result)


# Backward-compatible alias so existing imports of exact_keyword_search still work
# (used by src/module_3/profiler.py to validate column names against the FlexRIC codebase)
@tool
def exact_keyword_search(keyword: str, max_results: int = 5) -> str:
    """
    Check whether an exact C identifier or variable name exists in the FlexRIC codebase.

    This is a backward-compatible wrapper around flexric_rag_retrieve.
    Returns a short context block if the identifier is found (FLEXRIC_VALID),
    or 'No matches found.' if it does not appear in any FlexRIC source file.

    Use this to verify that a telemetry column name maps to a real FlexRIC C variable.
    """
    payload = {
        "query": keyword,
        "top_k": max_results,
        "hops": 0,  # exact matches only, no graph expansion
    }
    result = _post("retrieve", payload)
    if isinstance(result, dict):
        total = result.get("total_results", 0)
        if total == 0:
            return "No matches found."
        ctx = result.get("llm_context", "")
        return ctx[:2000]
    # Connection error or unexpected format
    return str(result)


# Convenience list for easy import into agent tool lists
flexric_rag_tools = [flexric_rag_context, flexric_rag_retrieve]
