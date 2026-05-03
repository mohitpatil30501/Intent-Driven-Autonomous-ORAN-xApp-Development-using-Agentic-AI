"""
flexric_rag/server.py

FastAPI wrapper for the FlexRIC Structural RAG pipeline.
Exposes the retriever as an HTTP API that LLMs can call as a tool.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000

Environment variables:
    FLEXRIC_INDEX_DIR   Path to the built index directory (required)
    FLEXRIC_API_KEY     Optional bearer token for auth
    FLEXRIC_MAX_TOP_K   Upper limit on top_k to prevent abuse (default: 20)
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── config ───────────────────────────────────────────────────────────────────
INDEX_DIR   = os.environ.get("FLEXRIC_INDEX_DIR", "./flexric_index/index")
API_KEY     = os.environ.get("FLEXRIC_API_KEY", "")           # empty = no auth
MAX_TOP_K   = int(os.environ.get("FLEXRIC_MAX_TOP_K", "20"))

# ── global retriever (loaded once at startup) ─────────────────────────────────
_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    idx = Path(INDEX_DIR)
    if not idx.exists():
        raise RuntimeError(
            f"FLEXRIC_INDEX_DIR '{INDEX_DIR}' does not exist. "
            "Build the index first: python pipeline.py build ..."
        )
    logger.info(f"Loading FlexRIC index from {idx} ...")
    # Import here so the server can start without all deps installed at module level
    from retrieval.retriever import StructuralRetriever
    _retriever = StructuralRetriever(idx)
    logger.info("Index loaded — server ready.")
    yield
    _retriever = None


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="FlexRIC Structural RAG API",
    description=(
        "Retrieve structurally-aware code context from the FlexRIC O-RAN codebase. "
        "Returns functions, structs, and call-graph neighbours relevant to a natural-language query."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── auth (optional) ──────────────────────────────────────────────────────────
bearer = HTTPBearer(auto_error=False)


def _check_auth(credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer)):
    if not API_KEY:
        return  # auth disabled
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ── Pydantic models ───────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural-language or symbol-based query about the FlexRIC codebase.",
        examples=["how do I subscribe to KPM measurements from an xApp"],
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of chunks to return (1–20).",
    )
    hops: int = Field(
        default=1,
        ge=0,
        le=2,
        description=(
            "Graph expansion depth. "
            "0 = seed chunks only; 1 = include direct callers/callees; "
            "2 = extend one more hop."
        ),
    )
    sm_type: Optional[Literal["KPM", "RC", "MAC", "RLC"]] = Field(
        default=None,
        description=(
            "Filter results to a specific E2 Service Model. "
            "Auto-detected from query keywords if omitted."
        ),
    )


class ChunkResult(BaseModel):
    chunk_id: str
    name: str
    type: str                    # function | struct | class | file_summary
    file: str
    layer: Optional[str]         # xapp | e2ap | ric | sm | api | util
    sm_type: Optional[str]       # E2SM_KPM | E2SM_RC | E2SM_MAC | E2SM_RLC | none
    is_xapp_example: bool
    signature: Optional[str]
    nl_summary: Optional[str]
    calls: List[str]
    called_by: List[str]
    score: float = Field(alias="_score")
    via_graph: Optional[str] = Field(alias="_via_graph")

    model_config = {"populate_by_name": True}


class RetrieveResponse(BaseModel):
    query: str
    detected_sm: Optional[str]
    total_results: int
    results: List[ChunkResult]
    llm_context: str = Field(
        description=(
            "Pre-formatted context block ready to paste into an LLM prompt. "
            "Contains signatures, summaries, and call edges."
        )
    )


class ContextRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=20)
    hops: int = Field(default=1, ge=0, le=2)
    sm_type: Optional[Literal["KPM", "RC", "MAC", "RLC"]] = None
    max_chars: int = Field(
        default=12000,
        ge=500,
        le=40000,
        description="Truncate the context block at this many characters.",
    )


class ContextResponse(BaseModel):
    query: str
    context: str
    chunk_count: int


# ── helpers ───────────────────────────────────────────────────────────────────

def _sm_flag(sm: Optional[str]) -> Optional[str]:
    """Map short SM name → full E2SM tag expected by the retriever."""
    _map = {"KPM": "E2SM_KPM", "RC": "E2SM_RC", "MAC": "E2SM_MAC", "RLC": "E2SM_RLC"}
    return _map.get(sm) if sm else None


def _build_llm_context(results: list, max_chars: int = 12000) -> str:
    """
    Render retrieved chunks into a compact, LLM-readable context block.
    Format is intentionally terse to fit within token budgets.
    """
    lines = []
    total = 0

    for i, r in enumerate(results, 1):
        header = (
            f"### [{i}] {r.get('name','')}  "
            f"({r.get('type','?')} · {r.get('layer','?')} · {r.get('sm_type','?')})"
        )
        parts = [header]

        if r.get("file"):
            parts.append(f"File: {r['file']}")
        if r.get("signature"):
            parts.append(f"Signature: {r['signature']}")
        if r.get("nl_summary"):
            parts.append(f"Summary: {r['nl_summary']}")
        if r.get("calls"):
            parts.append(f"Calls: {', '.join(r['calls'][:6])}")
        if r.get("called_by"):
            parts.append(f"Called by: {', '.join(r['called_by'][:4])}")
        if r.get("is_xapp_example"):
            parts.append("⚑ This is a complete xApp example.")
        if r.get("_via_graph") and r["_via_graph"] != "seed":
            parts.append(f"(included via graph: {r['_via_graph']})")

        block = "\n".join(parts) + "\n"
        if total + len(block) > max_chars:
            lines.append(f"... context truncated at {max_chars} characters ...")
            break
        lines.append(block)
        total += len(block)

    return "\n".join(lines)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    """Liveness check — returns 200 when the index is loaded."""
    return {"status": "ok", "index": INDEX_DIR}


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["retrieval"],
    summary="Retrieve code context",
    description=(
        "Given a natural-language or symbol query, returns the most relevant "
        "FlexRIC code chunks together with their call-graph context and a "
        "pre-formatted `llm_context` block ready to inject into an LLM prompt."
    ),
    dependencies=[Depends(_check_auth)],
)
def retrieve(req: RetrieveRequest):
    if _retriever is None:
        raise HTTPException(503, "Retriever not initialised.")

    clamped_top_k = min(req.top_k, MAX_TOP_K)
    sm_full = _sm_flag(req.sm_type)

    try:
        raw = _retriever.retrieve(
            query=req.query,
            top_k=clamped_top_k,
            hops=req.hops,
            sm_type=sm_full,
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(500, f"Retrieval error: {exc}")

    # Detect SM for informational purposes
    from retrieval.retriever import _detect_sm_type
    detected = sm_full or _detect_sm_type(req.query)

    llm_ctx = _build_llm_context(raw)

    results = [
        ChunkResult(
            chunk_id=r.get("chunk_id", ""),
            name=r.get("name", ""),
            type=r.get("type", ""),
            file=r.get("file", ""),
            layer=r.get("layer"),
            sm_type=r.get("sm_type"),
            is_xapp_example=bool(r.get("is_xapp_example")),
            signature=r.get("signature"),
            nl_summary=r.get("nl_summary"),
            calls=r.get("calls", []),
            called_by=r.get("called_by", []),
            _score=r.get("_score", 0.0),
            _via_graph=r.get("_via_graph"),
        )
        for r in raw
    ]

    return RetrieveResponse(
        query=req.query,
        detected_sm=detected,
        total_results=len(results),
        results=results,
        llm_context=llm_ctx,
    )


@app.post(
    "/context",
    response_model=ContextResponse,
    tags=["retrieval"],
    summary="Get plain context block",
    description=(
        "Lightweight endpoint that returns only the pre-formatted context string "
        "for direct injection into an LLM prompt. Prefer /retrieve for full metadata."
    ),
    dependencies=[Depends(_check_auth)],
)
def context(req: ContextRequest):
    if _retriever is None:
        raise HTTPException(503, "Retriever not initialised.")

    sm_full = _sm_flag(req.sm_type)
    try:
        raw = _retriever.retrieve(
            query=req.query,
            top_k=min(req.top_k, MAX_TOP_K),
            hops=req.hops,
            sm_type=sm_full,
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(500, f"Retrieval error: {exc}")

    ctx = _build_llm_context(raw, max_chars=req.max_chars)
    return ContextResponse(query=req.query, context=ctx, chunk_count=len(raw))


@app.get(
    "/schema",
    tags=["meta"],
    summary="OpenAI-compatible tool definitions",
    description="Returns the function/tool JSON schema for use with OpenAI, Anthropic, and other LLM providers.",
)
def schema():
    """Return the tool schema so LLMs can register these endpoints as tools."""
    return TOOL_DEFINITIONS


# ── OpenAI / Anthropic-compatible tool definitions ────────────────────────────
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "flexric_retrieve",
            "description": (
                "Search the FlexRIC O-RAN codebase for functions, structs, and call-graph context "
                "relevant to the user's question. Use this whenever the user asks about FlexRIC "
                "xApp development, E2 Service Model APIs (KPM, RC, MAC, RLC), subscription flows, "
                "RIC procedures, or any FlexRIC-specific code. Returns signatures, summaries, and "
                "a ready-to-use context block."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language or symbol-based query, e.g. "
                            "'how do I subscribe to KPM measurements' or 'e2ap_subscribe signature'."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of code chunks to retrieve (1–20). Default 8.",
                        "default": 8,
                    },
                    "hops": {
                        "type": "integer",
                        "description": (
                            "Call-graph expansion depth. "
                            "0 = exact matches only; 1 = include direct callers/callees (recommended); "
                            "2 = go one hop further."
                        ),
                        "default": 1,
                    },
                    "sm_type": {
                        "type": "string",
                        "enum": ["KPM", "RC", "MAC", "RLC"],
                        "description": (
                            "Filter to a specific E2 Service Model. "
                            "Omit to let the server auto-detect from the query."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flexric_context",
            "description": (
                "Retrieve a compact, ready-to-inject context block from the FlexRIC codebase. "
                "Use this when you only need the formatted text for your prompt, not the full metadata. "
                "Ideal as a final step before generating xApp code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language or symbol-based query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (1–20). Default 8.",
                        "default": 8,
                    },
                    "hops": {
                        "type": "integer",
                        "description": "Call-graph expansion depth (0–2). Default 1.",
                        "default": 1,
                    },
                    "sm_type": {
                        "type": "string",
                        "enum": ["KPM", "RC", "MAC", "RLC"],
                        "description": "Optional SM type filter.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters in the returned context block. Default 12000.",
                        "default": 12000,
                    },
                },
                "required": ["query"],
            },
        },
    },
]
