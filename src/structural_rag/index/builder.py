import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
flexric_rag/index/builder.py

Stage 4: Dual embedding + hybrid index construction.

Stores three complementary indices under index_path/:
  code_index.faiss  — CodeBERT embeddings of raw function bodies
  nl_index.faiss    — sentence-transformer embeddings of NL summaries
  bm25.pkl          — BM25Okapi over tokenized bodies (exact lexical match)
  graph.pkl         — networkx DiGraph (or adjacency dict fallback)
  meta.json         — chunk metadata (no vectors)
  config.json       — index configuration

Design rationale
─────────────────
• CodeBERT  : best when the query contains identifiers / API names
              ("e2ap_subscribe function", "kpm_ind_msg_t struct")
• NL embed  : best when the query is intent-driven
              ("how to subscribe to KPM measurements")
• BM25      : catches exact symbol names that dense models paraphrase away
              ("e2sm_kpm_ind_msg_t", "kpm_enc_action_def")
• Graph     : structural expansion — retrieves callers/callees of seed results
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional heavy dependencies — imported lazily so the module is importable
# even when not all packages are installed.
# ─────────────────────────────────────────────────────────────────────────────

def _load_code_model():
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        tok   = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base")
        model.eval()
        logger.info("CodeBERT loaded for code embedding")
        return tok, model
    except ImportError:
        logger.warning("transformers not available; code embedding will use sentence-transformer as fallback")
        return None, None


def _load_nl_model():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("sentence-transformer (all-MiniLM-L6-v2) loaded")
        return m
    except ImportError:
        logger.error("sentence-transformers not found. Install: pip install sentence-transformers")
        raise


def _load_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        logger.error("faiss not found. Install: pip install faiss-cpu   (or faiss-gpu)")
        raise


def _load_bm25():
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi
    except ImportError:
        logger.error("rank_bm25 not found. Install: pip install rank-bm25")
        raise


def _load_networkx():
    try:
        import networkx as nx
        return nx
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_with_codebert(texts: List[str], tok, model) -> np.ndarray:
    import torch
    vecs = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc   = tok(batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        # CLS token pooling
        cls_vecs = out.last_hidden_state[:, 0, :].numpy()
        vecs.append(cls_vecs)
        if i % 200 == 0 and i > 0:
            logger.info(f"  CodeBERT: {i}/{len(texts)} encoded")
    return np.vstack(vecs).astype("float32")


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer for BM25
# ─────────────────────────────────────────────────────────────────────────────

import re

def _bm25_tokenize(text: str) -> List[str]:
    """
    Split on non-alphanumeric boundaries, lowercase, filter short tokens.
    Keeps camelCase parts split (e.g., kpmIndMsg → kpm, ind, msg).
    """
    # split camelCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # split snake_case and other separators
    return [t.lower() for t in re.split(r"[^a-zA-Z0-9]+", text) if len(t) > 2]


# ─────────────────────────────────────────────────────────────────────────────
# IndexBuilder
# ─────────────────────────────────────────────────────────────────────────────

class IndexBuilder:
    """
    Build and persist the full hybrid index from a list of annotated chunks.
    """

    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

    def build(self, chunks: List[Dict], cg_data: Dict[str, Any]):
        # ── Load models ──────────────────────────────────────────────────────
        faiss_lib = _load_faiss()
        BM25Okapi = _load_bm25()
        nl_model  = _load_nl_model()
        c_tok, c_model = _load_code_model()
        nx_lib    = _load_networkx()

        n = len(chunks)
        logger.info(f"Building index for {n} chunks …")

        # ── NL embeddings ─────────────────────────────────────────────────────
        logger.info("  Encoding NL summaries …")
        nl_texts = [c.get("nl_summary") or c["name"] for c in chunks]
        nl_vecs  = nl_model.encode(nl_texts, batch_size=64,
                                   show_progress_bar=True,
                                   convert_to_numpy=True).astype("float32")
        nl_vecs  = _normalize(nl_vecs)

        # ── Code embeddings ───────────────────────────────────────────────────
        logger.info("  Encoding code bodies …")
        code_texts = [c["body"][:512] for c in chunks]
        if c_tok and c_model:
            code_vecs = _encode_with_codebert(code_texts, c_tok, c_model)
        else:
            # fallback: encode bodies with sentence-transformer
            logger.info("  (using sentence-transformer for code embedding fallback)")
            code_vecs = nl_model.encode(code_texts, batch_size=64,
                                        show_progress_bar=True,
                                        convert_to_numpy=True).astype("float32")
        code_vecs = _normalize(code_vecs)

        # ── FAISS indices ─────────────────────────────────────────────────────
        nl_dim   = nl_vecs.shape[1]
        code_dim = code_vecs.shape[1]

        nl_index   = faiss_lib.IndexFlatIP(nl_dim)
        code_index = faiss_lib.IndexFlatIP(code_dim)
        nl_index.add(nl_vecs)
        code_index.add(code_vecs)

        faiss_lib.write_index(nl_index,   str(self.index_path / "nl_index.faiss"))
        faiss_lib.write_index(code_index, str(self.index_path / "code_index.faiss"))
        logger.info(f"  FAISS: nl_dim={nl_dim}, code_dim={code_dim}")

        # ── BM25 ──────────────────────────────────────────────────────────────
        logger.info("  Building BM25 …")
        tokenized_corpus = [_bm25_tokenize(c["body"]) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(self.index_path / "bm25.pkl", "wb") as f:
            pickle.dump({"bm25": bm25, "tokenized": tokenized_corpus}, f)

        # ── Graph store ───────────────────────────────────────────────────────
        logger.info("  Building graph store …")
        if nx_lib:
            G = nx_lib.DiGraph()
            for i, c in enumerate(chunks):
                G.add_node(c["chunk_id"],
                           idx=i, name=c["name"],
                           layer=c.get("layer", ""),
                           sm_type=c.get("sm_type", ""),
                           is_xapp=c.get("is_xapp_example", False),
                           type=c["type"])
            adj = cg_data.get("adjacency", {})
            for cid, edges in adj.items():
                for callee_id in edges.get("calls", []):
                    G.add_edge(cid, callee_id, rel="calls")
            with open(self.index_path / "graph.pkl", "wb") as f:
                pickle.dump(G, f)
            logger.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        else:
            # store adjacency dict as fallback
            with open(self.index_path / "graph.pkl", "wb") as f:
                pickle.dump({"adjacency": cg_data.get("adjacency", {}), "_type": "adj"}, f)

        # ── Metadata ──────────────────────────────────────────────────────────
        meta = []
        for c in chunks:
            meta.append({
                k: c[k] for k in
                ["chunk_id", "name", "type", "file", "signature",
                 "layer", "sm_type", "is_xapp_example", "module",
                 "calls", "called_by", "nl_summary"]
                if k in c
            })
        with open(self.index_path / "meta.json", "w") as f:
            json.dump(meta, f)

        # ── Config ────────────────────────────────────────────────────────────
        config = {
            "n_chunks":   n,
            "nl_dim":     nl_dim,
            "code_dim":   code_dim,
            "nl_model":   "all-MiniLM-L6-v2",
            "code_model": "microsoft/codebert-base" if (c_tok and c_model)
                          else "all-MiniLM-L6-v2 (fallback)",
            "has_networkx": nx_lib is not None,
        }
        (self.index_path / "config.json").write_text(json.dumps(config, indent=2))

        logger.info(f"Index written to {self.index_path}")
