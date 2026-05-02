"""
flexric_rag/pipeline.py

End-to-end Structural RAG pipeline for the FlexRIC codebase.

Usage:
    # 1. Build the index from a FlexRIC repo
    python pipeline.py build --repo /path/to/flexric --out ./flexric_index

    # 2. Query the index
    python pipeline.py query --index ./flexric_index --q "how do I subscribe to KPM measurements"

    # 3. Interactive REPL
    python pipeline.py repl --index ./flexric_index
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.parser import CodebaseParser
from core.callgraph import CallGraphBuilder
from index.builder import IndexBuilder
from retrieval.retriever import StructuralRetriever
from summarize.summarizer import SummarizerFactory
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s", datefmt="%H:%M:%S", stream=__import__("sys").stderr)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# BUILD
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline(repo_path: str, out_dir: str, summarizer_backend: str = "ollama",
                   summarizer_model: str = "codellama", force: bool = False):
    """
    Full build pipeline:
      parse → call-graph → NL-summarize → dual-embed → index
    """
    repo = Path(repo_path).resolve()
    out  = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    chunks_path = out / "chunks.jsonl"
    graph_path  = out / "callgraph.json"
    index_path  = out / "index"

    # ── Stage 1: AST parsing ─────────────────────────────────────────────────
    if chunks_path.exists() and not force:
        logger.info("Chunks already exist, loading from cache (use --force to rebuild)")
        chunks = [json.loads(l) for l in chunks_path.read_text().splitlines() if l.strip()]
    else:
        logger.info("Stage 1/4 — AST parsing …")
        t0 = time.time()
        parser = CodebaseParser(repo)
        chunks = parser.parse_all()
        logger.info(f"  Extracted {len(chunks)} chunks in {time.time()-t0:.1f}s")
        with open(chunks_path, "w") as f:
            for c in chunks:
                # don't persist ast node objects
                safe = {k: v for k, v in c.items() if k != "_ast_node"}
                f.write(json.dumps(safe) + "\n")

    # ── Stage 2: Call graph ──────────────────────────────────────────────────
    if graph_path.exists() and not force:
        logger.info("Call-graph already exists, loading from cache")
        cg_data = json.loads(graph_path.read_text())
    else:
        logger.info("Stage 2/4 — Building call graph …")
        t0 = time.time()
        builder = CallGraphBuilder(chunks)
        cg_data = builder.build()
        chunks  = builder.annotate_chunks(chunks, cg_data)
        chunks  = CallGraphBuilder.fix_cross_sm_functions(chunks)  # fix cross-SM API functions
        graph_path.write_text(json.dumps(cg_data, indent=2))
        # update chunks file with call annotations
        with open(chunks_path, "w") as f:
            for c in chunks:
                safe = {k: v for k, v in c.items() if k != "_ast_node"}
                f.write(json.dumps(safe) + "\n")
        logger.info(f"  Call graph: {cg_data['stats']} — {time.time()-t0:.1f}s")

    # ── Stage 3: NL summarization ────────────────────────────────────────────
    summaries_exist = all(c.get("nl_summary") for c in chunks[:10])
    if summaries_exist and not force:
        logger.info("NL summaries already present, skipping")
    else:
        logger.info(f"Stage 3/4 — NL summarization via [{summarizer_backend}/{summarizer_model}] …")
        summarizer = SummarizerFactory.create(summarizer_backend, summarizer_model)
        t0 = time.time()
        chunks = summarizer.summarize_chunks(chunks)
        # persist updated chunks
        with open(chunks_path, "w") as f:
            for c in chunks:
                safe = {k: v for k, v in c.items() if k != "_ast_node"}
                f.write(json.dumps(safe) + "\n")
        logger.info(f"  Summarized {len(chunks)} chunks in {time.time()-t0:.1f}s")

    # ── Stage 4: Dual embed + index ──────────────────────────────────────────
    logger.info("Stage 4/4 — Dual embedding and indexing …")
    t0 = time.time()
    ib = IndexBuilder(index_path)
    ib.build(chunks, cg_data)
    logger.info(f"  Index built in {time.time()-t0:.1f}s")

    logger.info(f"\n✓ Pipeline complete. Index written to: {out}")
    _print_stats(chunks, cg_data)


# ──────────────────────────────────────────────────────────────────────────────
# QUERY
# ──────────────────────────────────────────────────────────────────────────────

def query_pipeline(index_dir: str, query: str, top_k: int = 12, hops: int = 1,
                   sm_filter: str = None, output_format: str = "text"):
    retriever = _load_retriever(index_dir)
    results   = retriever.retrieve(query, top_k=top_k, hops=hops, sm_type=sm_filter)

    if output_format == "json":
        print(json.dumps([_chunk_summary(r) for r in results], indent=2))
    else:
        _print_results(query, results)

    return results


def repl_pipeline(index_dir: str):
    retriever = _load_retriever(index_dir)
    print("\n╔══════════════════════════════════════════════╗")
    print("║   FlexRIC Structural RAG — Interactive REPL  ║")
    print("╚══════════════════════════════════════════════╝")
    print("Commands: :sm <KPM|RC|MAC|RLC>  :hops <n>  :top <n>  :quit\n")

    sm_filter = None
    hops = 1
    top_k = 12

    while True:
        try:
            raw = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue
        if raw == ":quit":
            break
        if raw.startswith(":sm "):
            sm_filter = raw.split()[1].upper()
            print(f"  SM filter set to: {sm_filter}")
            continue
        if raw.startswith(":hops "):
            hops = int(raw.split()[1])
            print(f"  Graph hops: {hops}")
            continue
        if raw.startswith(":top "):
            top_k = int(raw.split()[1])
            print(f"  Top-k: {top_k}")
            continue

        results = retriever.retrieve(raw, top_k=top_k, hops=hops, sm_type=sm_filter)
        _print_results(raw, results)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_retriever(index_dir: str) -> StructuralRetriever:
    p = Path(index_dir)
    if not p.exists():
        sys.exit(f"Index directory not found: {index_dir}")
    logger.info(f"Loading index from {p} …")
    return StructuralRetriever(p / "index")


def _print_results(query: str, results: list):
    print(f"\n── Query: '{query}' ── {len(results)} results ──────────────────")
    for i, r in enumerate(results, 1):
        xapp_flag = " ★" if r.get("is_xapp_example") else ""
        sm        = f"[{r['sm_type']}]" if r.get("sm_type") and r["sm_type"] != "none" else ""
        print(f"\n[{i}] {r['name']}{xapp_flag}  {sm}  score={r.get('_score', 0):.3f}")
        print(f"    file : {r['file']}")
        print(f"    layer: {r.get('layer','?')}   type: {r['type']}")
        if r.get("nl_summary"):
            print(f"    desc : {r['nl_summary']}")
        if r.get("calls"):
            print(f"    calls: {', '.join(r['calls'][:5])}")
        if r.get("called_by"):
            print(f"    ←used: {', '.join(r['called_by'][:3])}")
        if r.get("_via_graph"):
            print(f"    via  : graph expansion ({r['_via_graph']})")
    print()


def _chunk_summary(c: dict) -> dict:
    return {k: c[k] for k in
            ["chunk_id", "name", "type", "file", "sm_type", "layer",
             "is_xapp_example", "nl_summary", "calls", "called_by",
             "_score", "_via_graph"]   # ← ADD THIS
            if k in c}


def _print_stats(chunks: list, cg_data: dict):
    from collections import Counter
    types  = Counter(c["type"]  for c in chunks)
    layers = Counter(c.get("layer", "?") for c in chunks)
    sms    = Counter(c.get("sm_type", "none") for c in chunks if c.get("sm_type") != "none")
    xapp   = sum(1 for c in chunks if c.get("is_xapp_example"))

    print("\n── Index stats ────────────────────────────────────")
    print(f"  Total chunks  : {len(chunks)}")
    print(f"  xApp examples : {xapp}")
    print(f"  Types         : {dict(types)}")
    print(f"  Layers        : {dict(layers)}")
    print(f"  SM coverage   : {dict(sms)}")
    print(f"  Call edges    : {cg_data.get('stats', {}).get('edges', '?')}")
    print("───────────────────────────────────────────────────\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FlexRIC Structural RAG — build and query a structural code index"
    )
    sub = ap.add_subparsers(dest="cmd")

    # build
    bp = sub.add_parser("build", help="Parse FlexRIC repo and build the index")
    bp.add_argument("--repo",  required=True, help="Path to FlexRIC root")
    bp.add_argument("--out",   default="./flexric_index", help="Output index directory")
    bp.add_argument("--summarizer", default="ollama",
                    choices=["ollama","hf","tfidf","none"],
                    help="NL summarizer backend (default: ollama)")
    bp.add_argument("--model", default="codellama",
                    help="Model name for summarizer (default: codellama)")
    bp.add_argument("--force", action="store_true", help="Rebuild all stages")

    # query
    qp = sub.add_parser("query", help="Query the built index")
    qp.add_argument("--index", required=True, help="Index directory")
    qp.add_argument("--q",     required=True, help="Query string")
    qp.add_argument("--top",   type=int, default=12)
    qp.add_argument("--hops",  type=int, default=1)
    qp.add_argument("--sm",    default=None, help="Filter by SM type (KPM|RC|MAC|RLC)")
    qp.add_argument("--json",  action="store_true", help="JSON output")

    # repl
    rp = sub.add_parser("repl", help="Interactive query REPL")
    rp.add_argument("--index", required=True)

    args = ap.parse_args()

    if args.cmd == "build":
        build_pipeline(args.repo, args.out, args.summarizer, args.model, args.force)
    elif args.cmd == "query":
        query_pipeline(args.index, args.q, args.top, args.hops, args.sm,
                       "json" if args.json else "text")
    elif args.cmd == "repl":
        repl_pipeline(args.index)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()