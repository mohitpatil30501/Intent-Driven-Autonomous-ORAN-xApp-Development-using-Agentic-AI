import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
flexric_rag/summarize/summarizer.py

Stage 3: NL summarization — generating one-sentence summaries for each chunk
so the NL embedding has meaningful signal beyond raw code tokens.

──────────────────────────────────────────────────────────────────────────────
BACKEND OPTIONS (no API key required for any of them)
──────────────────────────────────────────────────────────────────────────────

1. OLLAMA  (recommended)
   - Run a local LLM (CodeLlama, Mistral, Llama-3, Phi-3, …) via Ollama.
   - Zero cost, fully offline, best quality after API-hosted models.
   - Install: https://ollama.com  → `ollama pull codellama`
   - Config : OLLAMA_BASE_URL, OLLAMA_MODEL env vars

2. HUGGINGFACE (local transformers)
   - Use any seq2seq or causal model from HuggingFace Hub locally.
   - Good choices: Salesforce/codet5p-220m-bimodal, Phi-2, starcoder2-3b
   - Needs GPU/CPU RAM; slower than Ollama but fully self-contained.
   - Install: pip install transformers accelerate sentencepiece

3. LLAMA_CPP (llama-cpp-python)
   - Run GGUF quantised models directly in Python; no Ollama daemon needed.
   - Works on CPU (slow) or GPU (fast).
   - Install: pip install llama-cpp-python
   - Download any GGUF from https://huggingface.co/TheBloke

4. GPT4ALL
   - Desktop-friendly local LLM runner; simple Python API.
   - Install: pip install gpt4all

5. TFIDF  (zero-dependency fallback)
   - Extract the most distinctive tokens from each function as a pseudo-summary.
   - No model needed; fast; lower quality but still useful for BM25 indexing.
   - Always available as a fallback.

6. NONE
   - Skip summarization entirely (nl_summary = "").
   - The NL index will still work but will be less useful for intent queries.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import re
from abc import ABC, abstractmethod
from typing import List, Dict

import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are a telecom systems engineer summarizing C/Python code for a FlexRIC O-RAN codebase.
Write exactly ONE sentence (max 25 words) describing what this function does and which O-RAN
component it belongs to (xApp, E2AP, KPM, RC, RIC, etc.).

Function name: {name}
File: {file}
Signature: {signature}

Code (first 600 chars):
{body_snippet}

One-sentence summary:"""


def _make_prompt(chunk: Dict) -> str:
    return _PROMPT_TEMPLATE.format(
        name=chunk["name"],
        file=chunk["file"],
        signature=chunk.get("signature", ""),
        body_snippet=chunk["body"][:600],
    )


def _clean_summary(raw: str) -> str:
    """Strip leading labels like 'Summary:' and trailing whitespace."""
    raw = raw.strip()
    raw = re.sub(r"^(summary|description|one.sentence summary)\s*:?\s*",
                 "", raw, flags=re.IGNORECASE)
    # take only first sentence
    sentences = re.split(r"(?<=[.!?])\s", raw)
    return sentences[0].strip() if sentences else raw[:120]


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseSummarizer(ABC):
    """Summarize a list of code chunks in-place."""

    SKIP_TYPES = {"file_summary"}   # don't summarize file-level chunks

    def summarize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        to_summarize = [c for c in chunks
                        if c["type"] not in self.SKIP_TYPES
                        and not c.get("nl_summary")]
        total = len(to_summarize)
        logger.info(f"Summarizing {total} chunks with {self.__class__.__name__} …")

        for i, chunk in enumerate(to_summarize):
            try:
                summary = self.summarize_one(chunk)
                chunk["nl_summary"] = _clean_summary(summary)
            except Exception as exc:
                logger.debug(f"  Failed on {chunk['name']}: {exc}")
                chunk["nl_summary"] = self._fallback_summary(chunk)

            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{total} done …")

        # file-summary chunks get a simple auto-summary
        for c in chunks:
            if c["type"] == "file_summary" and not c.get("nl_summary"):
                c["nl_summary"] = f"Source file: {c['file']}"

        return chunks

    @abstractmethod
    def summarize_one(self, chunk: Dict) -> str:
        ...

    def _fallback_summary(self, chunk: Dict) -> str:
        parts = []
        sig = chunk.get("signature", "")
        if sig:
            parts.append(sig)
        callers = [c.split("::")[-1] for c in chunk.get("called_by", [])[:2]]
        if callers:
            parts.append(f"called by {', '.join(callers)}")
        callees = [c.split("::")[-1] for c in chunk.get("calls", [])[:3]]
        if callees:
            parts.append(f"calls {', '.join(callees)}")
        return " — ".join(parts) if parts else chunk["name"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ollama backend
# ─────────────────────────────────────────────────────────────────────────────

class OllamaSummarizer(BaseSummarizer):
    """
    Uses a locally running Ollama server (http://localhost:11434 by default).
    Start with: `ollama serve` and `ollama pull codellama` (or any model).

    Recommended models (quality ↑ / speed ↓):
      codellama:7b   — best for C code, 4-8 GB RAM
      phi3:mini      — fast, surprisingly good, ~2 GB RAM
      mistral:7b     — general purpose, good quality
      llama3:8b      — strong reasoning, 5 GB RAM
    """

    def __init__(self, model: str = "codellama", base_url: str = None):
        self.model    = model
        self.base_url = (base_url
                         or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
        import urllib.request
        self._urlrequest = urllib.request

    def summarize_one(self, chunk: Dict) -> str:
        import json, urllib.request
        payload = json.dumps({
            "model":  self.model,
            "prompt": _make_prompt(chunk),
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 60},
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get("response", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. HuggingFace local transformers
# ─────────────────────────────────────────────────────────────────────────────

class HFSummarizer(BaseSummarizer):
    """
    Run any HuggingFace text-generation or seq2seq model locally.

    Good picks for code summarization (no API key):
      - Salesforce/codet5p-220m-bimodal   (seq2seq, fast, ~0.9 GB)
      - microsoft/phi-2                   (causal, very capable, ~5 GB)
      - bigcode/starcoder2-3b             (code-focused, ~6 GB)
      - google/flan-t5-large              (instruction tuned, ~3 GB)

    Install: pip install transformers accelerate sentencepiece bitsandbytes
    """

    def __init__(self, model_name: str = "Salesforce/codet5p-220m-bimodal",
                 device: str = "auto"):
        logger.info(f"Loading HuggingFace model: {model_name} …")
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        self.model_name = model_name
        self.device     = device

        # Auto-detect seq2seq vs causal
        try:
            self.pipe = pipeline(
                "text2text-generation",
                model=model_name,
                device_map=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                max_new_tokens=64,
            )
            self._mode = "seq2seq"
        except Exception:
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                device_map=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                max_new_tokens=64,
                truncation=True,
                max_length=512,
            )
            self._mode = "causal"

        logger.info(f"Model loaded in {self._mode} mode")

    def summarize_one(self, chunk: Dict) -> str:
        prompt = _make_prompt(chunk)
        result = self.pipe(prompt[:1024])[0]

        if self._mode == "seq2seq":
            return result.get("generated_text", "")
        else:
            # strip the prompt from causal output
            full = result.get("generated_text", "")
            return full[len(prompt):].strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3. llama-cpp-python backend
# ─────────────────────────────────────────────────────────────────────────────

class LlamaCppSummarizer(BaseSummarizer):
    """
    Use any GGUF model via llama-cpp-python.
    Install: pip install llama-cpp-python
    Models : https://huggingface.co/TheBloke (e.g. CodeLlama-7B-Instruct.Q4_K_M.gguf)

    Args:
        model_path: Path to the .gguf file on disk.
        n_gpu_layers: Number of layers to offload to GPU (0 = CPU only).
    """

    def __init__(self, model_path: str, n_gpu_layers: int = 0, n_ctx: int = 2048):
        logger.info(f"Loading GGUF model from {model_path} …")
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def summarize_one(self, chunk: Dict) -> str:
        prompt = _make_prompt(chunk)
        output = self.llm(
            prompt,
            max_tokens=64,
            temperature=0.1,
            stop=["Function name:", "\n\n"],
        )
        return output["choices"][0]["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4. GPT4All backend
# ─────────────────────────────────────────────────────────────────────────────

class GPT4AllSummarizer(BaseSummarizer):
    """
    Use GPT4All — a simple desktop-friendly local LLM runner.
    Install: pip install gpt4all
    Models download automatically on first use.

    Recommended: "Phi-3-mini-4k-instruct.Q4_0.gguf" (~2 GB)
    """

    def __init__(self, model_name: str = "Phi-3-mini-4k-instruct.Q4_0.gguf"):
        from gpt4all import GPT4All
        logger.info(f"Loading GPT4All model: {model_name} …")
        self.model = GPT4All(model_name)

    def summarize_one(self, chunk: Dict) -> str:
        prompt = _make_prompt(chunk)
        with self.model.chat_session():
            return self.model.generate(prompt, max_tokens=64, temp=0.1).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5. TF-IDF keyword extractor (zero-dependency fallback)
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFSummarizer(BaseSummarizer):
    """
    Extract distinctive keywords from the function body using TF-IDF and
    combine them with the function's name and layer metadata to produce a
    pseudo-summary.

    No models, no API keys, no GPU — runs instantly on any machine.
    Quality is lower than LLM summaries but still useful for:
      - BM25 retrieval (keyword matching)
      - NL embedding (injects domain terminology)

    Ideal when: bootstrapping quickly, or as a fallback during CI/CD.
    """

    def __init__(self):
        self._idf: Dict[str, float] = {}
        self._corpus_ready = False

    def summarize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        # Build corpus IDF from all function bodies first
        self._build_idf(chunks)
        return super().summarize_chunks(chunks)

    def _build_idf(self, chunks: List[Dict]):
        import math
        from collections import Counter

        docs = [self._tokenize(c["body"]) for c in chunks
                if c["type"] not in self.SKIP_TYPES]
        df: Dict[str, int] = Counter()
        for doc in docs:
            for tok in set(doc):
                df[tok] += 1
        N = max(len(docs), 1)
        self._idf = {t: math.log(N / (1 + f)) for t, f in df.items()}
        self._corpus_ready = True

    def summarize_one(self, chunk: Dict) -> str:
        tokens   = self._tokenize(chunk["body"])
        tf       = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = max(len(tokens), 1)

        scored = {t: (c / total) * self._idf.get(t, 0.0) for t, c in tf.items()}
        top    = sorted(scored, key=lambda t: -scored[t])[:8]

        layer  = chunk.get("layer", "")
        sm     = chunk.get("sm_type", "")
        prefix = f"[{sm}] " if sm and sm != "none" else ""
        xapp   = "xApp" if chunk.get("is_xapp_example") else layer

        keywords = ", ".join(top)
        return f"{prefix}{chunk['name']} ({xapp}): {keywords}"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # split on non-alphanum, lowercase, filter short tokens
        return [t.lower() for t in re.split(r"[^a-zA-Z0-9_]+", text)
                if len(t) > 2 and not t.isdigit()]   # ← match BM25 threshold


# ─────────────────────────────────────────────────────────────────────────────
# 6. No-op summarizer
# ─────────────────────────────────────────────────────────────────────────────

class NullSummarizer(BaseSummarizer):
    """Skip summarization; leave nl_summary empty."""
    def summarize_one(self, chunk: Dict) -> str:
        return f"{chunk['name']} — {chunk.get('layer','')}"


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

class SummarizerFactory:
    """
    Create the right summarizer based on the backend name.

    Usage:
        s = SummarizerFactory.create("ollama", "codellama")
        s = SummarizerFactory.create("hf", "Salesforce/codet5p-220m-bimodal")
        s = SummarizerFactory.create("llamacpp", "/models/codellama.Q4_K_M.gguf")
        s = SummarizerFactory.create("gpt4all", "Phi-3-mini-4k-instruct.Q4_0.gguf")
        s = SummarizerFactory.create("tfidf")
        s = SummarizerFactory.create("none")
    """

    @staticmethod
    def create(backend: str, model: str = None) -> BaseSummarizer:
        b = backend.lower()
        if b == "ollama":
            return OllamaSummarizer(model or "codellama")
        elif b in ("hf", "huggingface"):
            return HFSummarizer(model or "Salesforce/codet5p-220m-bimodal")
        elif b in ("llamacpp", "llama_cpp", "llama-cpp"):
            if not model:
                raise ValueError("llamacpp backend requires --model /path/to/model.gguf")
            return LlamaCppSummarizer(model)
        elif b == "gpt4all":
            return GPT4AllSummarizer(model or "Phi-3-mini-4k-instruct.Q4_0.gguf")
        elif b == "tfidf":
            return TFIDFSummarizer()
        elif b == "none":
            return NullSummarizer()
        else:
            raise ValueError(
                f"Unknown summarizer backend: {backend!r}. "
                f"Choose: ollama | hf | llamacpp | gpt4all | tfidf | none"
            )
