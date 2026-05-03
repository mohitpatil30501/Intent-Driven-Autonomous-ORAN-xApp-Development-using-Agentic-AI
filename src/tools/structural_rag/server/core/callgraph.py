import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
flexric_rag/core/callgraph.py

Stage 2: Build a bidirectional call graph from the extracted chunks.

Strategy:
  - For C: scan function bodies for call_expression patterns using tree-sitter
    (or regex fallback).
  - Resolve callee names to chunk_ids where possible.
  - Produce a networkx DiGraph + a serialisable adjacency dict.
  - Annotate each chunk with "calls" and "called_by" lists.

Key FlexRIC edges to capture:
  - xApp → E2AP (subscription API boundary — highest hallucination risk)
  - xApp → SM encoding/decoding functions
  - RIC subscription manager → xApp callback registration
"""

import re
from collections import defaultdict
from typing import Dict, List, Any

import logging

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    logger.warning("networkx not found; graph will be stored as adjacency dict only. "
                   "Install: pip install networkx")

# ─────────────────────────────────────────────────────────────────────────────
# Call extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

# Match C-style function calls: identifier(
_CALL_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")

# Identifiers that look like keywords or are not function calls
_SKIP_NAMES = frozenset({
    "if", "while", "for", "switch", "return", "sizeof", "typeof",
    "assert", "printf", "fprintf", "sprintf", "snprintf", "malloc",
    "calloc", "free", "memset", "memcpy", "memmove", "strlen", "strcmp",
    "strncmp", "strcpy", "strncpy", "NULL", "true", "false",
})


def _extract_call_names_regex(body: str) -> List[str]:
    """Extract potential function call names from a body of C/Python code."""
    calls = []
    for m in _CALL_RE.finditer(body):
        name = m.group(1)
        if name not in _SKIP_NAMES and not name[0].isupper():
            calls.append(name)
    return list(set(calls))


def _extract_call_names_treesitter(body: str, ts_c_parser) -> List[str]:
    """Use tree-sitter for precise call extraction (preferred)."""
    calls = []
    try:
        tree = ts_c_parser.parse(body.encode())

        def walk(node):
            if node.type == "call_expression":
                fn_child = node.children[0] if node.children else None
                if fn_child and fn_child.type == "identifier":
                    name = body[fn_child.start_byte: fn_child.end_byte]
                    if name not in _SKIP_NAMES:
                        calls.append(name)
            for child in node.children:
                walk(child)

        walk(tree.root_node)
    except Exception:
        pass
    return list(set(calls))


# ─────────────────────────────────────────────────────────────────────────────
# FlexRIC boundary classification
# ─────────────────────────────────────────────────────────────────────────────

_XAPP_E2AP_BOUNDARY = {
    "e2ap_subscribe", "e2ap_unsubscribe", "e2ap_control",
    "e2ap_indication_cb", "e2ap_setup",
}

_SM_ENCODE_DECODE = {
    "kpm_enc_action_def", "kpm_dec_ind_msg", "kpm_enc_sub_req",
    "rc_enc_ctrl_req",    "rc_dec_ctrl_resp",
    "mac_enc_action_def", "rlc_enc_action_def",
}


def _edge_type(caller_layer: str, callee_name: str) -> str:
    if callee_name in _XAPP_E2AP_BOUNDARY:
        return "xapp_to_e2ap"
    if callee_name in _SM_ENCODE_DECODE:
        return "sm_codec"
    return "internal"


# ─────────────────────────────────────────────────────────────────────────────
# CallGraphBuilder
# ─────────────────────────────────────────────────────────────────────────────

class CallGraphBuilder:
    """
    Build a bidirectional call graph from a list of chunk dicts.

    The resulting graph is stored as:
      - self.graph  : networkx.DiGraph (if networkx available)
      - self.adj    : dict  chunk_id → {"calls": [...], "called_by": [...]}
    """

    def __init__(self, chunks: List[Dict], ts_c_parser=None):
        self.chunks        = chunks
        self.ts_parser     = ts_c_parser   # optional tree-sitter C parser
        self._name_to_ids  = defaultdict(list)  # function name → [chunk_ids]

        for c in chunks:
            self._name_to_ids[c["name"]].append(c["chunk_id"])

    # ── public ──────────────────────────────────────────────────────────────

    def build(self) -> Dict[str, Any]:
        """
        Returns a serialisable dict:
        {
          "adjacency": { chunk_id: {"calls": [...], "called_by": [...]} },
          "boundary_edges": [ {from, to, type} ],
          "stats": { nodes, edges, xapp_to_e2ap_edges }
        }
        """
        adj: Dict[str, Dict] = {c["chunk_id"]: {"calls": [], "called_by": []}
                                 for c in self.chunks}
        boundary_edges = []
        total_edges    = 0

        if _HAS_NX:
            self.graph = nx.DiGraph()
            for c in self.chunks:
                self.graph.add_node(c["chunk_id"],
                                    name=c["name"], layer=c.get("layer",""),
                                    sm_type=c.get("sm_type",""), type=c["type"],
                                    is_xapp=c.get("is_xapp_example", False))

        for chunk in self.chunks:
            if chunk["type"] not in ("function", "class"):
                continue

            raw_calls = (
                _extract_call_names_treesitter(chunk["body"], self.ts_parser)
                if self.ts_parser
                else _extract_call_names_regex(chunk["body"])
            )

            resolved_ids = []
            for call_name in raw_calls:
                callee_ids = self._name_to_ids.get(call_name, [])
                if not callee_ids:
                    continue

                # prefer callee in same sm_type or same layer
                callee_id = self._pick_best_callee(chunk, callee_ids)
                if callee_id and callee_id != chunk["chunk_id"]:
                    resolved_ids.append(callee_id)
                    adj[callee_id]["called_by"].append(chunk["chunk_id"])
                    total_edges += 1

                    etype = _edge_type(chunk.get("layer", ""), call_name)
                    if etype != "internal":
                        boundary_edges.append({
                            "from":  chunk["chunk_id"],
                            "to":    callee_id,
                            "type":  etype,
                            "name":  call_name,
                        })

                    if _HAS_NX:
                        self.graph.add_edge(chunk["chunk_id"], callee_id,
                                            rel="calls", edge_type=etype)

            adj[chunk["chunk_id"]]["calls"] = list(set(resolved_ids))

        stats = {
            "nodes":              len(self.chunks),
            "edges":              total_edges,
            "boundary_edges":     len(boundary_edges),
            "xapp_to_e2ap_edges": sum(1 for e in boundary_edges
                                      if e["type"] == "xapp_to_e2ap"),
        }
        logger.info(f"Call graph: {stats}")

        return {
            "adjacency":       adj,
            "boundary_edges":  boundary_edges,
            "stats":           stats,
        }

    def annotate_chunks(self, chunks: List[Dict],
                        cg_data: Dict[str, Any]) -> List[Dict]:
        """Write resolved calls/called_by back into each chunk dict."""
        adj = cg_data["adjacency"]
        for c in chunks:
            cid = c["chunk_id"]
            c["calls"]     = adj.get(cid, {}).get("calls",     [])
            c["called_by"] = adj.get(cid, {}).get("called_by", [])
        return chunks


    @staticmethod
    def fix_cross_sm_functions(chunks: List[Dict]) -> List[Dict]:
        """
        If a function is called by callers spanning more than one SM type,
        mark it as cross-cutting API (sm_type='none', layer='api').
        This fixes functions like report_sm_xapp_api that serve all SM types
        but were mis-tagged by path-based classification.
        """
        sm_map = {c["chunk_id"]: c.get("sm_type", "none") for c in chunks}

        for chunk in chunks:
            if not chunk.get("called_by"):
                continue

            real_sm_types = {
                sm_map.get(caller)
                for caller in chunk["called_by"]
                if sm_map.get(caller) not in (None, "none")
            }

            if len(real_sm_types) > 1:
                chunk["sm_type"] = "none"
                chunk["layer"]   = "api"

        return chunks


    # ── private ─────────────────────────────────────────────────────────────

    def _pick_best_callee(self, caller: Dict, callee_ids: List[str]) -> str | None:
        """
        When multiple chunks share the same function name (e.g., static helpers
        in different TUs), prefer the one in the same sm_type or layer.
        """
        if len(callee_ids) == 1:
            return callee_ids[0]

        caller_sm    = caller.get("sm_type", "none")
        caller_layer = caller.get("layer", "")

        for cid in callee_ids:
            chunk = next((c for c in self.chunks if c["chunk_id"] == cid), None)
            if chunk and chunk.get("sm_type") == caller_sm:
                return cid
        for cid in callee_ids:
            chunk = next((c for c in self.chunks if c["chunk_id"] == cid), None)
            if chunk and chunk.get("layer") == caller_layer:
                return cid

        return callee_ids[0]   # default: first match