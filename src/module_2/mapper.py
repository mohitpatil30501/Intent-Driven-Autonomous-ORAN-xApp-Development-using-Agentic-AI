import json
import re
import os
import requests
from typing import Any, Dict
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.semantic_search.semantic_search_tool import (
    semantic_search_summary,
    semantic_search_detailed,
)
from tools.context_utils import limit_tool_messages

MODULE_2_SYSTEM_PROMPT = """You are "Module 2: The O-RAN Technical Mapper" in an automated xApp development pipeline.
Your ONLY job is to map the "requested_Telemetry_NL" and "target_Action_What_NL" into EXACT FlexRIC Service Models and C-struct variables.

CRITICAL RULES — NO HALLUCINATIONS:
1. You MUST use `semantic_search_summary` to look up the FlexRIC codebase. Never guess variable names.
2. TRACE NESTED TYPES: If a field is a struct or a pointer to a struct (e.g., `fr_slice_t* slices`), you MUST find the definition of that struct type to see its inner fields (like `id`).
3. UNIONS & VARIANTS: When you encounter a `union`, you MUST identify the discriminator field (usually an `enum` like `type` or `conf`) that indicates which union member is active. Map the telemetry to the specific union member path (e.g., `ind.msg.slice_conf.dl.slices[i].params.u.nvs.u.rate.u1.mbps_required` for NVS rates).
4. ARRAYS & POINTERS: If a variable is part of an array or a list, use the `[i]` index notation (e.g., `ind.msg.slice_conf.dl.slices[i].id`).
5. ENTRY POINT: Use `ind.msg` as the default root for indication messages unless the codebase shows otherwise (e.g., `ind.msg.tstamp`).
6. NO TEMPLATES: Map to actual C struct/union fields. Do not use generic template placeholders.

--- SEARCH STRATEGY ---
1. CALL 1: Find the main Indication Message struct for the likely Service Model (e.g., SLICE, KPM, RC).
2. CALL 2+: If the main struct contains nested structs, pointers, or UNIONS, search for those specific type definitions (e.g., "typedef struct fr_slice_t" or "typedef union slice_params_u").
3. IDENTIFY DISCRIMINATORS: If a union is involved, find the enum that controls it so you can map specific telemetry to the correct union branch.
4. DO NOT STOP until you have mapped every requested field or confirmed it's absolutely missing.

--- RESPONSE FORMAT ---
Output a strict JSON code block:

```json
{
  "Intent_Blueprint": {
    "... (copy from input, unchanged)"
  },
  "Technical_Mapping": {
    "Reporting_Service_Model": "MAC | KPM | RLC | RC | SLICE",
    "Telemetry_Variables": [
      {
        "NL_name": "human-readable name from Module 1",
        "C_variable": "exact path (e.g. ind.msg.slice_conf.dl.slices[i].id)",
        "data_type": "uint32_t | uint64_t | float | char*"
      }
    ],
    "Control_Service_Model": "MAC | RC | SLICE | ...",
    "Action_Space_Menu": [
      {
        "action_id": "UPDATE_SLICE_PRB",
        "description": "what it does",
        "parameters": { "slice_id": "uint32_t", "prb_ratio": "uint8_t" }
      },
      {
        "action_id": "DO_NOTHING",
        "description": "Take no action",
        "parameters": {}
      }
    ]
  }
}
```
"""

ORIOSEARCH_URL = os.getenv("ORIOSEARCH_URL", "http://localhost:8000")


@tool
def restricted_domain_search(query: str, domain: str = "o-ran-sc.org") -> str:
    """
    Search the web for O-RAN concepts or external documentation.
    Use only if the FlexRIC codebase search is insufficient.
    """
    try:
        res = requests.get(
            f"{ORIOSEARCH_URL}/search",
            params={"q": f"{query} site:{domain}"},
            timeout=10,
        )
        if res.status_code == 200:
            results = res.json().get("results", [])
            if not results:
                return "No results found."
            return "\n\n".join(
                f"Title: {r.get('title')}\nSnippet: {r.get('content')}"
                for r in results[:3]
            )
        return f"Error: Status code {res.status_code}"
    except Exception as e:
        return f"Error connecting to oriosearch: {e}"


tools = [semantic_search_summary, semantic_search_detailed, restricted_domain_search]


def get_llm():
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=ollama_model, base_url=ollama_url)


def module_2_technical_node(state: dict) -> dict:
    """Module 2: Maps NL intent to exact FlexRIC C-variables using Structural RAG."""

    intent_blueprint = state.get("blueprint", {})

    # Include the SM type hint if already determinable from the intent blueprint
    requested_sm = (
        intent_blueprint.get("Intent_Blueprint", {})
        .get("Reporting_Service_Model", "")
    )
    sm_hint = f"\nHint: the likely Service Model is '{requested_sm}'." if requested_sm else ""

    prompt_content = (
        f"Here is the Intent Blueprint from Module 1:\n"
        f"{json.dumps(intent_blueprint, indent=2)}\n\n"
        f"Use semantic_search_summary (2 calls max) to look up the FlexRIC codebase, "
        f"then output the Technical_Mapping JSON.{sm_hint}"
    )

    llm = get_llm()
    mapper_agent = create_react_agent(
        model=llm, 
        tools=tools, 
        prompt=MODULE_2_SYSTEM_PROMPT,
        pre_model_hook=limit_tool_messages
    )

    try:
        result = mapper_agent.invoke(
            {"messages": [HumanMessage(content=prompt_content)]},
            {"recursion_limit": int(os.getenv("MAPPER_RECURSIVE_LIMIT", max(40, int(os.getenv("RECURSIVE_LIMIT", 20)))))},
        )
        final_text = result["messages"][-1].content
    except Exception as e:
        print(f"Module 2 Error: {e}")
        return {"messages": [AIMessage(content="Failed to map variables. Check logs.")]}

    new_blueprint = intent_blueprint
    json_match = re.search(r'```json\s*(.*?)\s*```', final_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if "Technical_Mapping" in parsed:
                new_blueprint = parsed
        except json.JSONDecodeError:
            pass

    return {
        "blueprint": new_blueprint,
        "messages": [AIMessage(content=final_text)],
    }
