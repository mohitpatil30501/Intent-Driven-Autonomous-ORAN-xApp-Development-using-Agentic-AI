"""
flexric_rag/llm_integration.py

Drop-in examples showing how to wire the FlexRIC RAG API as a tool
for various LLM providers.

Requires: pip install openai anthropic requests
"""

import json
import os
import requests

# ── Base URL for the running server ──────────────────────────────────────────
BASE_URL  = os.environ.get("FLEXRIC_RAG_URL", "http://localhost:8000")
API_KEY   = os.environ.get("FLEXRIC_API_KEY", "")  # empty = no auth
HEADERS   = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


# ─────────────────────────────────────────────────────────────────────────────
# Direct HTTP client (no LLM library required)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 8, hops: int = 1, sm_type: str = None) -> dict:
    """Call /retrieve and return the full response dict."""
    payload = {"query": query, "top_k": top_k, "hops": hops}
    if sm_type:
        payload["sm_type"] = sm_type
    r = requests.post(f"{BASE_URL}/retrieve", json=payload, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get_context(query: str, top_k: int = 8, hops: int = 1,
                sm_type: str = None, max_chars: int = 12000) -> str:
    """Call /context and return the plain context string."""
    payload = {"query": query, "top_k": top_k, "hops": hops, "max_chars": max_chars}
    if sm_type:
        payload["sm_type"] = sm_type
    r = requests.post(f"{BASE_URL}/context", json=payload, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["context"]


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI tool-use integration
# ─────────────────────────────────────────────────────────────────────────────

def run_openai_agent(user_question: str):
    """
    Runs a two-turn OpenAI tool-use loop:
      1. LLM decides to call flexric_retrieve or flexric_context
      2. We execute the real HTTP call and feed the result back
      3. LLM generates the final answer grounded in real code context
    """
    from openai import OpenAI

    client = OpenAI()

    # Fetch tool schema from the server so it's always in sync
    tools = requests.get(f"{BASE_URL}/schema").json()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert FlexRIC O-RAN xApp developer. "
                "When the user asks about FlexRIC code, ALWAYS call flexric_retrieve first "
                "to get accurate API context before answering. "
                "Never invent function signatures — use only what the tool returns."
            ),
        },
        {"role": "user", "content": user_question},
    ]

    # ── Turn 1: let the LLM decide what to look up ────────────────────────
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message
    messages.append(msg)

    # ── Execute tool calls ────────────────────────────────────────────────
    if msg.tool_calls:
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"[tool] {tc.function.name}({args})")

            if tc.function.name == "flexric_retrieve":
                result = retrieve(**args)
                content = result["llm_context"]
            elif tc.function.name == "flexric_context":
                content = get_context(**args)
            else:
                content = f"Unknown tool: {tc.function.name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })

    # ── Turn 2: final answer grounded in retrieved context ─────────────────
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return final.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic (Claude) tool-use integration
# ─────────────────────────────────────────────────────────────────────────────

def _openai_tool_to_anthropic(tool: dict) -> dict:
    """Convert an OpenAI-format tool definition to Anthropic's format."""
    fn = tool["function"]
    return {
        "name": fn["name"],
        "description": fn["description"],
        "input_schema": fn["parameters"],
    }


def run_anthropic_agent(user_question: str):
    """
    Runs a multi-turn Anthropic tool-use loop with Claude.
    """
    import anthropic

    client = anthropic.Anthropic()

    openai_tools = requests.get(f"{BASE_URL}/schema").json()
    tools = [_openai_tool_to_anthropic(t) for t in openai_tools]

    system = (
        "You are an expert FlexRIC O-RAN xApp developer. "
        "When the user asks about FlexRIC code, ALWAYS call flexric_retrieve first "
        "to get accurate API context before answering. "
        "Never invent function signatures — use only what the tool returns."
    )
    messages = [{"role": "user", "content": user_question}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Extract text from the final response
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        # Execute tool calls and build tool_result blocks
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"[tool] {block.name}({block.input})")

            if block.name == "flexric_retrieve":
                result = retrieve(**block.input)
                content = result["llm_context"]
            elif block.name == "flexric_context":
                content = get_context(**block.input)
            else:
                content = f"Unknown tool: {block.name}"

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })

        messages.append({"role": "user", "content": tool_results})


# ─────────────────────────────────────────────────────────────────────────────
# LangChain tool wrapper
# ─────────────────────────────────────────────────────────────────────────────

def make_langchain_tools():
    """
    Returns a list of LangChain Tool objects wrapping the FlexRIC RAG API.
    Usage:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        tools = make_langchain_tools()
        agent = create_tool_calling_agent(llm, tools, prompt)
    """
    from langchain.tools import StructuredTool
    from pydantic import BaseModel, Field as PydanticField
    from typing import Optional

    class RetrieveInput(BaseModel):
        query: str = PydanticField(description="Natural-language or symbol-based query.")
        top_k: int = PydanticField(default=8, description="Number of results (1–20).")
        hops: int = PydanticField(default=1, description="Graph expansion depth (0–2).")
        sm_type: Optional[str] = PydanticField(default=None, description="KPM | RC | MAC | RLC")

    class ContextInput(BaseModel):
        query: str = PydanticField(description="Natural-language or symbol-based query.")
        top_k: int = PydanticField(default=8)
        hops: int = PydanticField(default=1)
        sm_type: Optional[str] = PydanticField(default=None)
        max_chars: int = PydanticField(default=12000)

    def _retrieve_fn(query, top_k=8, hops=1, sm_type=None):
        data = retrieve(query, top_k=top_k, hops=hops, sm_type=sm_type)
        return data["llm_context"]

    def _context_fn(query, top_k=8, hops=1, sm_type=None, max_chars=12000):
        return get_context(query, top_k=top_k, hops=hops, sm_type=sm_type, max_chars=max_chars)

    return [
        StructuredTool(
            name="flexric_retrieve",
            description=(
                "Search the FlexRIC O-RAN codebase. Use for any question about FlexRIC "
                "xApp development, E2SM APIs, subscription flows, or RIC procedures."
            ),
            func=_retrieve_fn,
            args_schema=RetrieveInput,
        ),
        StructuredTool(
            name="flexric_context",
            description="Get a compact context block from FlexRIC for injecting into prompts.",
            func=_context_fn,
            args_schema=ContextInput,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Minimal demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic", "direct"], default="direct")
    ap.add_argument("--q", default="how do I subscribe to KPM measurements from an xApp in C?")
    args = ap.parse_args()

    if args.provider == "direct":
        ctx = get_context(args.q, top_k=6)
        print("=== Context block ===\n")
        print(ctx)

    elif args.provider == "openai":
        answer = run_openai_agent(args.q)
        print("=== OpenAI answer ===\n")
        print(answer)

    elif args.provider == "anthropic":
        answer = run_anthropic_agent(args.q)
        print("=== Anthropic answer ===\n")
        print(answer)
