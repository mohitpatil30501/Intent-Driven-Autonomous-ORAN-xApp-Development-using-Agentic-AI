# flexric_rag/retrieval/retriever.py

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RRF
# ─────────────────────────────────────────────────────────────────────────────

def _rrf(ranked_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# BM25 tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_tokenize(text: str) -> List[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", text) if len(t) > 2]


# ─────────────────────────────────────────────────────────────────────────────
# SM detection
# ─────────────────────────────────────────────────────────────────────────────

_SM_KEYWORDS = {
    "kpm":  "E2SM_KPM",
    "key performance": "E2SM_KPM",
    " rc ": "E2SM_RC",
    "radio control": "E2SM_RC",
    "mac":  "E2SM_MAC",
    "rlc":  "E2SM_RLC",
}

def _detect_sm_type(query: str) -> str | None:
    q = query.lower()
    for kw, sm in _SM_KEYWORDS.items():
        if kw in q:
            return sm
    return None

def _detect_query_intent(query: str) -> str:
    """Returns 'callee' for implementation queries, 'caller' for usage queries."""
    q = query.lower()

    callee_signals = {
        "internal", "implementation", "how does", "what does"
    }
    caller_signals = {
        "what uses", "what calls", "who calls", "used by"
    }

    if any(s in q for s in callee_signals):
        return "callee"
    if any(s in q for s in caller_signals):
        return "caller"
    return "both"


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────

class StructuralRetriever:

    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self._load()

    def _load(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.config = json.loads((self.index_path / "config.json").read_text())

        self.meta = json.loads((self.index_path / "meta.json").read_text())
        self._id_to_idx = {c["chunk_id"]: i for i, c in enumerate(self.meta)}

        self.nl_index   = faiss.read_index(str(self.index_path / "nl_index.faiss"))
        self.code_index = faiss.read_index(str(self.index_path / "code_index.faiss"))

        self.nl_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.code_model = self._try_load_codebert()

        bm25_data = pickle.loads((self.index_path / "bm25.pkl").read_bytes())
        self.bm25 = bm25_data["bm25"]

        graph_data = pickle.loads((self.index_path / "graph.pkl").read_bytes())
        if hasattr(graph_data, "nodes"):
            self.graph = graph_data
            self._graph_mode = "nx"
        else:
            self.graph = graph_data.get("adjacency", {})
            self._graph_mode = "adj"

    def _try_load_codebert(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            model = AutoModel.from_pretrained("microsoft/codebert-base")
            model.eval()
            return (tok, model)
        except ImportError:
            return None

    # ─────────────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 8, hops: int = 1,
                 sm_type: str = None) -> List[Dict]:

        nl_q_vec   = self._encode_nl(query)
        code_q_vec = self._encode_code(query)
        bm25_tokens = _bm25_tokenize(query)

        detected_sm = sm_type or _detect_sm_type(query)
        intent = _detect_query_intent(query)

        n_seed = max(top_k * 2, 20)

        _, nl_hits   = self.nl_index.search(nl_q_vec, n_seed)
        _, code_hits = self.code_index.search(code_q_vec, n_seed)

        bm25_scores = self.bm25.get_scores(bm25_tokens)
        bm25_hits = list(np.argsort(bm25_scores)[::-1][:n_seed])

        # ── Fix C: Exact name-match boost ─────────────────────────────
# retriever.py — after computing bm25_hits

        name_to_idx = {self.meta[i]["name"]: i for i in range(len(self.meta))}

        for name, idx in name_to_idx.items():
            name_tokens = set(_bm25_tokenize(name))
            query_tokens = set(bm25_tokens)
            # Only boost if the name tokens are a superset of ALL query tokens
            # meaning the query is describing this function precisely
            if query_tokens and query_tokens.issubset(name_tokens):
                if idx in bm25_hits:
                    bm25_hits.remove(idx)
                bm25_hits.insert(0, idx)

        rrf_scores = _rrf([
            list(nl_hits[0]),
            list(code_hits[0]),
            bm25_hits,
        ])

        seed_idxs = sorted(rrf_scores, key=lambda i: -rrf_scores[i])[:top_k]

        if detected_sm:
            seed_idxs = [
                i for i in seed_idxs
                if self.meta[i].get("sm_type") in (detected_sm, "none")
            ]

        # ── Graph expansion ───────────────────────────────────────────────
        expanded = {}

        for idx in seed_idxs:
            cid = self.meta[idx]["chunk_id"]

            expanded[idx] = {
                "score": rrf_scores[idx],
                "via": "seed"   # ← explicit
            }

            for nid, direction in self._get_neighbors(cid, hops, intent=intent):
                nidx = self._id_to_idx.get(nid)
                if nidx is not None and nidx not in expanded:
                    expanded[nidx] = {
                        "score": rrf_scores[idx] * 0.4,
                        "via": direction   # caller / callee
                    }

        if detected_sm:
            expanded = {
                idx: data for idx, data in expanded.items()
                if self.meta[idx].get("sm_type") in (detected_sm, "none")
            }

        # ── Re-rank ───────────────────────────────────────────────────────
        def _rank(idx):
            c = self.meta[idx]
            score = expanded[idx]["score"]
            return -(
                score +
                (0.3 if c.get("is_xapp_example") else 0) +
                (0.2 if detected_sm and c.get("sm_type") == detected_sm else 0) +
                (0.1 if c.get("layer") == "api" else 0)
            )

        ranked = sorted(expanded, key=_rank)[:top_k]

        # ── Final output ───────────────────────────────────────────────────
        results = []
        for idx in ranked:
            chunk = dict(self.meta[idx])
            chunk["_score"] = round(expanded[idx]["score"], 4)
            chunk["_via_graph"] = expanded[idx]["via"]   # ← ALWAYS PRESENT
            results.append(chunk)

        return results

    # ─────────────────────────────────────────────────────────────────────────

    def _encode_nl(self, text: str):
        vec = self.nl_model.encode([text], convert_to_numpy=True).astype("float32")
        return vec / (np.linalg.norm(vec) or 1.0)

    def _encode_code(self, text: str):
        if self.code_model:
            tok, model = self.code_model
            import torch
            enc = tok([text], return_tensors="pt", truncation=True)
            with torch.no_grad():
                out = model(**enc)
            vec = out.last_hidden_state[:, 0, :].numpy().astype("float32")
        else:
            vec = self._encode_nl(text)
        return vec / (np.linalg.norm(vec) or 1.0)

    # ─────────────────────────────────────────────────────────────────────────

    def _get_neighbors(self, cid: str, hops: int, intent: str = "both"):
        visited = {cid}
        frontier = [(cid, 0)]
        results = []

        while frontier:
            cid, depth = frontier.pop()
            if depth >= hops:
                continue

            # ── Callee expansion (downstream calls) ──
            if intent in ("callee", "both"):
                for nid in self._successors(cid):
                    if nid not in visited:
                        visited.add(nid)
                        results.append((nid, "callee"))
                        frontier.append((nid, depth + 1))

            # ── Caller expansion (who calls this) ──
            if intent in ("caller", "both"):
                for nid in self._predecessors(cid):
                    if nid not in visited:
                        visited.add(nid)
                        results.append((nid, "caller"))
                        frontier.append((nid, depth + 1))

        return results

    def _successors(self, cid):
        return list(self.graph.successors(cid)) if self._graph_mode == "nx" else self.graph.get(cid, {}).get("calls", [])

    def _predecessors(self, cid):
        return list(self.graph.predecessors(cid)) if self._graph_mode == "nx" else self.graph.get(cid, {}).get("called_by", [])