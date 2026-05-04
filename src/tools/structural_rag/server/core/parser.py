import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
flexric_rag/core/parser.py

Stage 1: AST-based chunk extraction for the FlexRIC codebase.

Supports:
  - C files  (.c, .h) via tree-sitter-c
  - Python   (.py)    via tree-sitter-python
  - Falls back to regex-based extraction when tree-sitter is unavailable

Each chunk carries the full schema described in the design doc.
"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import logging
from rag_utils.flexric_tags import classify_file

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FlexRIC directory patterns
# ─────────────────────────────────────────────────────────────────────────────

XAPP_DIRS   = {"examples/xApp", "examples/xapp"}
SM_DIRS = {
    "src/sm/kpm_sm", "src/sm/rc_sm",
    "src/sm/mac_sm", "src/sm/rlc_sm",
    "src/sm/pdcp_sm", "src/sm/gtp_sm",  # ← ADD
    "src/sm/slice_sm", "src/sm/tc_sm",  # ← ADD
}
RIC_DIRS    = {"src/ric", "src/near-rt-ric"}
UTIL_DIRS   = {"src/util", "src/lib"}
INCLUDE_DIR = "include"

SM_TYPE_MAP = {
    "kpm": "E2SM_KPM",
    "rc":  "E2SM_RC",
    "mac": "E2SM_MAC",
    "rlc": "E2SM_RLC",
    "slice": "E2SM_SLICE",
    "e2ap": "E2AP",
}

C_EXTENSIONS  = {".c", ".h", ".cc", ".cpp"}
PY_EXTENSIONS = {".py"}

# parser.py

SKIP_DIRS = {
    "build", "cmake", ".git", "CMakeFiles",
    "ci-scripts",
    "alg_ds",
    "sqlite3",
    "emulator",
    "docker", "grafana", "multiRAT", "multiRIC", "openshift",
    "swig",        # ← ADD: auto-generated Python binding wrappers
    "fig",         # ← ADD: images only
}

# Also exclude specific high-noise paths
SKIP_PATH_FRAGMENTS = {
    "src/agent",
    "src/ric/iApps",
    "src/ric/msg_handler_ric",
    "src/ric/near_ric",
    "src/ric/e2ap_ric",
    "src/ric/asio_ric",
    "src/ric/endpoint_ric",
    "src/ric/not_handler_ric",
    "src/ric/e2_node",         # ← ADD
    "src/ric/map_e2_node",     # ← ADD
    "src/ric/generate_setup",  # ← ADD
    "src/ric/act_req",         # ← ADD
    "src/ric/plugin_ric",      # ← ADD: sm_plugin_ric was noise in Test 4
    "src/lib/e2ap/v1_01",
    "src/lib/e2ap/v2_03",
    "src/sm/agent_if",         # ← ADD: free_subscribe_timer noise source
    "rc_sm_agent", "rc_sm_ric",
    "kpm_sm_agent", "kpm_sm_ric",
    "mac_sm_agent", "mac_sm_ric",
    "rlc_sm_agent", "rlc_sm_ric",
    "pdcp_sm_agent", "pdcp_sm_ric",
    "gtp_sm_agent", "gtp_sm_ric",
    "slice_sm_agent", "slice_sm_ric",
    "tc_sm_agent", "tc_sm_ric",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tree-sitter helpers (optional dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_treesitter():
    """Returns (c_parser, py_parser) or (None, None) if unavailable."""
    try:
        import tree_sitter_c      as tsc
        import tree_sitter_python as tspy
        from tree_sitter import Language, Parser

        c_lang  = Language(tsc.language())
        py_lang = Language(tspy.language())

        c_parser  = Parser(c_lang)
        py_parser = Parser(py_lang)
        logger.info("tree-sitter loaded (C + Python grammars)")
        return c_parser, py_parser
    except ImportError:
        logger.warning(
            "tree-sitter not found — falling back to regex parser. "
            "Install: pip install tree-sitter tree-sitter-c tree-sitter-python"
        )
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Chunk ID
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk_id(rel_path: str, name: str) -> str:
    return f"{rel_path}::{name}"


# ─────────────────────────────────────────────────────────────────────────────
# Metadata classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify_chunk(rel_path: str, name: str) -> Dict[str, Any]:
    """Infer layer, sm_type, is_xapp_example from the file path."""
    parts = rel_path.replace("\\", "/").lower()

    is_xapp = any(xd in parts for xd in XAPP_DIRS)
    layer   = "xapp" if is_xapp else "other"

    if any(sd in parts for sd in SM_DIRS):
        layer = "sm"
    elif any(rd in parts for rd in RIC_DIRS):
        layer = "ric"
    elif any(ud in parts for ud in UTIL_DIRS):
        layer = "util"
    elif INCLUDE_DIR in parts:
        layer = "api"

    # Dynamic SM detection: if file is in src/sm/<sm_name>_sm, use <sm_name>
    sm_type = "none"
    if "src/sm" in parts:
        # Extract folder name after src/sm/
        match = re.search(r"src/sm/([^/]+)", parts)
        if match:
            sm_folder = match.group(1)
            # Normalize: e.g. kpm_sm -> KPM, slice_sm -> SLICE
            sm_type = "E2SM_" + sm_folder.replace("_sm", "").upper()
    else:
        # Fallback to keyword matching for files outside src/sm
        path_tokens = set(re.split(r"[^a-zA-Z0-9]", parts))
        name_tokens = set(re.split(r"[^a-zA-Z0-9]", name.lower()))
        for key, val in SM_TYPE_MAP.items():
            if key in path_tokens or key in name_tokens:
                sm_type = val
                break

    return {
        "layer": layer,
        "sm_type": sm_type,
        "is_xapp_example": is_xapp,
        "module": _infer_module(parts),
    }


def _infer_module(parts: str) -> str:
    for marker in ["kpm_rc", "kpm", "rc_sm", "rc", "mac", "rlc", "e2ap", "ric"]:
        if marker in parts:
            return marker
    return "core"


# ─────────────────────────────────────────────────────────────────────────────
# Tree-sitter C extractor
# ─────────────────────────────────────────────────────────────────────────────

class TreeSitterCExtractor:
    def __init__(self, parser):
        self.parser = parser

    def extract(self, filepath: Path, repo_root: Path) -> List[Dict]:
        src_bytes = filepath.read_bytes()
        tree      = self.parser.parse(src_bytes)
        rel_path  = str(filepath.relative_to(repo_root))
        chunks    = []

        self._walk(tree.root_node, src_bytes, rel_path, chunks, depth=0)
        return chunks

    def _walk(self, node, src: bytes, rel_path: str, chunks: list, depth: int):
        """Walk AST; extract function_definitions and struct_specifiers."""
        if depth > 6:          # avoid infinite recursion on deeply nested code
            return

        if node.type == "function_definition":
            chunk = self._extract_function(node, src, rel_path)
            if chunk:
                chunks.append(chunk)
            return  # don't recurse into function bodies for top-level chunks

        if node.type in ("struct_specifier", "enum_specifier", "union_specifier"):
            chunk = self._extract_struct(node, src, rel_path)
            if chunk:
                chunks.append(chunk)

        for child in node.children:
            self._walk(child, src, rel_path, chunks, depth + 1)

    def _extract_function(self, node, src: bytes, rel_path: str) -> Dict | None:
        name      = self._get_function_name(node, src)
        if not name:
            return None
        signature = self._get_signature(node, src)
        body_text = src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        meta      = _classify_chunk(rel_path, name)

        return {
            "chunk_id": _make_chunk_id(rel_path, name),
            "type":      "function",
            "name":      name,
            "signature": signature,
            "body":      body_text,
            "file":      rel_path,
            "calls":     [],          # filled by CallGraphBuilder
            "called_by": [],
            "nl_summary": "",         # filled by Summarizer
            **meta,
        }

    def _extract_struct(self, node, src: bytes, rel_path: str) -> Dict | None:
        # find the type name: struct_specifier > type_identifier
        name = None
        for child in node.children:
            if child.type == "type_identifier":
                name = src[child.start_byte:child.end_byte].decode()
                break
        if not name:
            return None

        body_text = src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        meta      = _classify_chunk(rel_path, name)

        return {
            "chunk_id":   _make_chunk_id(rel_path, name),
            "type":       "struct",
            "name":       name,
            "signature":  f"struct {name}",
            "body":       body_text,
            "file":       rel_path,
            "calls":      [],
            "called_by":  [],
            "nl_summary": "",
            **meta,
        }

    def _get_function_name(self, node, src: bytes) -> str | None:
        """Navigate function_definition → declarator → identifier."""
        for child in node.children:
            if child.type in ("function_declarator", "pointer_declarator"):
                return self._extract_identifier(child, src)
            if child.type == "identifier":
                return src[child.start_byte:child.end_byte].decode()
        return None

    def _extract_identifier(self, node, src: bytes) -> str | None:
        for child in node.children:
            if child.type == "identifier":
                return src[child.start_byte:child.end_byte].decode()
            if child.type in ("function_declarator", "pointer_declarator"):
                return self._extract_identifier(child, src)
        return None

    def _get_signature(self, node, src: bytes) -> str:
        """Return the function signature (everything before the compound_statement)."""
        for child in node.children:
            if child.type == "compound_statement":
                sig = src[node.start_byte:child.start_byte].decode("utf-8", errors="replace")
                return re.sub(r"\s+", " ", sig).strip()
        return src[node.start_byte:node.end_byte][:120].decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Tree-sitter Python extractor
# ─────────────────────────────────────────────────────────────────────────────

class TreeSitterPyExtractor:
    def __init__(self, parser):
        self.parser = parser

    def extract(self, filepath: Path, repo_root: Path) -> List[Dict]:
        src_bytes = filepath.read_bytes()
        tree      = self.parser.parse(src_bytes)
        rel_path  = str(filepath.relative_to(repo_root))
        chunks    = []

        for node in tree.root_node.children:
            if node.type in ("function_definition", "decorated_definition"):
                chunk = self._extract_function(node, src_bytes, rel_path)
                if chunk:
                    chunks.append(chunk)
            elif node.type == "class_definition":
                chunk = self._extract_class(node, src_bytes, rel_path)
                if chunk:
                    chunks.append(chunk)
        return chunks

    def _extract_function(self, node, src: bytes, rel_path: str) -> Dict | None:
        # handle decorated_definition wrapping
        fn_node = node
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "function_definition":
                    fn_node = child
                    break

        name = None
        for child in fn_node.children:
            if child.type == "identifier":
                name = src[child.start_byte:child.end_byte].decode()
                break
        if not name:
            return None

        body_text = src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        meta      = _classify_chunk(rel_path, name)

        return {
            "chunk_id":   _make_chunk_id(rel_path, name),
            "type":       "function",
            "name":       name,
            "signature":  body_text.split("\n")[0].strip(),
            "body":       body_text,
            "file":       rel_path,
            "calls":      [],
            "called_by":  [],
            "nl_summary": "",
            **meta,
        }

    def _extract_class(self, node, src: bytes, rel_path: str) -> Dict | None:
        name = None
        for child in node.children:
            if child.type == "identifier":
                name = src[child.start_byte:child.end_byte].decode()
                break
        if not name:
            return None

        body_text = src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        meta      = _classify_chunk(rel_path, name)

        return {
            "chunk_id":   _make_chunk_id(rel_path, name),
            "type":       "class",
            "name":       name,
            "signature":  body_text.split("\n")[0].strip(),
            "body":       body_text,
            "file":       rel_path,
            "calls":      [],
            "called_by":  [],
            "nl_summary": "",
            **meta,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Regex fallback extractor (C only)
# ─────────────────────────────────────────────────────────────────────────────

# Matches:  return_type function_name(params) {
_C_FUNC_RE = re.compile(
    r"(?:^|\n)"                          # line start
    r"(?:static\s+|inline\s+|extern\s+)*"  # optional qualifiers
    r"(?:[\w\s\*]+?)\s+"                # return type
    r"(\w+)\s*\("                        # function name
    r"([^)]*)\)"                         # params
    r"\s*(?:__attribute__\s*\([^)]*\))?" # optional __attribute__
    r"\s*\{",
    re.MULTILINE,
)


class RegexCExtractor:
    """Fallback when tree-sitter is not installed."""

    def extract(self, filepath: Path, repo_root: Path) -> List[Dict]:
        src      = filepath.read_text(errors="replace")
        rel_path = str(filepath.relative_to(repo_root))
        chunks   = []

        for match in _C_FUNC_RE.finditer(src):
            name   = match.group(1)
            params = match.group(2).strip()
            # crude body extraction: from match start to matching brace
            start  = match.start()
            body   = self._extract_body(src, match.end() - 1)
            meta   = _classify_chunk(rel_path, name)

            chunks.append({
                "chunk_id":   _make_chunk_id(rel_path, name),
                "type":       "function",
                "name":       name,
                "signature":  f"{name}({params})",
                "body":       body,
                "file":       rel_path,
                "calls":      [],
                "called_by":  [],
                "nl_summary": "",
                **meta,
            })

        return chunks

    def _extract_body(self, src: str, open_brace_pos: int) -> str:
        depth = 0
        for i, ch in enumerate(src[open_brace_pos:], start=open_brace_pos):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return src[open_brace_pos: i + 1]
        return src[open_brace_pos: open_brace_pos + 2000]  # truncate if unmatched


# ─────────────────────────────────────────────────────────────────────────────
# File-summary chunk (one per file)
# ─────────────────────────────────────────────────────────────────────────────

def _make_file_summary_chunk(filepath: Path, repo_root: Path,
                              sub_chunks: List[Dict]) -> Dict:
    rel_path = str(filepath.relative_to(repo_root))
    meta     = _classify_chunk(rel_path, "")
    names    = [c["name"] for c in sub_chunks if c["type"] == "function"][:20]

    return {
        "chunk_id":   f"{rel_path}::__file__",
        "type":       "file_summary",
        "name":       filepath.name,
        "signature":  rel_path,
        "body":       f"File: {rel_path}\nFunctions: {', '.join(names)}",
        "file":       rel_path,
        "calls":      [],
        "called_by":  [],
        "nl_summary": "",
        **meta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main parser
# ─────────────────────────────────────────────────────────────────────────────

class CodebaseParser:
    """
    Walk the FlexRIC repo, extract function/struct/class chunks from every
    supported source file, and return a flat list of chunk dicts.
    """

    def __init__(self, repo_root: Path):
        self.repo = Path(repo_root)
        c_ts, py_ts = _try_import_treesitter()
        self.c_extractor  = TreeSitterCExtractor(c_ts) if c_ts  else RegexCExtractor()
        self.py_extractor = TreeSitterPyExtractor(py_ts) if py_ts else None

        mode = "tree-sitter" if c_ts else "regex"
        logger.info(f"Using {mode} C extractor")

    def parse_all(self) -> List[Dict]:
        all_chunks: List[Dict] = []
        file_count = 0

        for filepath in self._iter_source_files():
            ext = filepath.suffix.lower()
            try:
                if ext in C_EXTENSIONS:
                    file_chunks = self.c_extractor.extract(filepath, self.repo)
                elif ext in PY_EXTENSIONS and self.py_extractor:
                    file_chunks = self.py_extractor.extract(filepath, self.repo)
                else:
                    continue

                if file_chunks:
                    # add a file-level summary chunk
                    file_chunks.append(
                        _make_file_summary_chunk(filepath, self.repo, file_chunks)
                    )
                    all_chunks.extend(file_chunks)
                    file_count += 1

            except Exception as exc:
                logger.debug(f"Skipping {filepath}: {exc}")

        logger.info(f"Parsed {file_count} files → {len(all_chunks)} chunks")
        return all_chunks

# Inside CodebaseParser — replace the existing _iter_source_files with:
    def _iter_source_files(self):
        for p in self.repo.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in (C_EXTENSIONS | PY_EXTENSIONS):
                continue
            rel = str(p.relative_to(self.repo)).replace("\\", "/")
            if any(skip in p.parts for skip in SKIP_DIRS):
                continue
            if any(frag in rel for frag in SKIP_PATH_FRAGMENTS):
                continue
            yield p