import json
import re
import os
import copy
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
from tools.context_utils import limit_context_window

MODULE_2_SYSTEM_PROMPT = """You are "Module 2: The O-RAN Technical Mapper" in an automated xApp development pipeline.
Your ONLY job is to map the "requested_Telemetry_NL" and "target_Action_What_NL" into EXACT FlexRIC Service Models and C-struct variables.

CRITICAL RULES — NO HALLUCINATIONS:
1. You MUST use `semantic_search_summary` to look up the FlexRIC codebase. Never guess variable names.
2. TRACE NESTED TYPES: If a field is a struct or a pointer to a struct (e.g., `fr_slice_t* slices`), you MUST find the definition of that struct type to see its inner fields (like `id`).
3. UNIONS & VARIANTS: When you encounter a `union`, you MUST expose the different attributes and possible variants as nested objects within the JSON structure so the complete object is visible.
4. HIERARCHICAL JSON OUTPUT: The target output for Telemetry_Variables MUST be a hierarchical JSON object representing the complete streaming dataset payload structure, NOT a flat list of separate variables. Represent arrays in C as JSON arrays (`[]`) and structs as JSON objects (`{}`).
5. ENTRY POINT IDENTIFICATION: You MUST identify the exact Indication Message struct (e.g., `kpm_ind_msg_t`, `slice_ind_msg_t`) that serves as the root of the telemetry stream. Search for "Indication Message" or "ind_msg" in the context of the requested Service Model.
6. RECURSIVE TYPE EXPLORATION: For EVERY field in the root struct, if its type is not a primitive (int, float, etc.), you MUST search for its definition until you reach primitive types or well-known types. This is essential for providing "detailed knowledge" of the Service Model objects.
7. NO TEMPLATES: Map to actual C struct/union field types found in the code. Do not use generic template placeholders.

--- SEARCH STRATEGY ---
1. CALL 1: Find the main Indication Message struct for the likely Service Model (e.g., SLICE, KPM, RC, MAC). Use queries like "typedef struct <SM>_ind_msg_s".
2. CALL 2+: For every nested struct type or union encountered (e.g., `fr_slice_t`, `slice_params_u`), call `semantic_search_detailed` or `semantic_search_summary` to find its full definition.
3. IDENTIFY DISCRIMINATORS & VARIANTS: If a union is used, find the enum discriminator (often in the same or parent struct) to understand how the variants are selected.
4. VERIFY HIERARCHY: Ensure the final JSON reflects the exact nesting found in the C code, starting from the Indication Message root.
5. DO NOT STOP until you have mapped every requested field with its full structural detail.

--- RESPONSE FORMAT ---
Output a strict JSON code block:

```json
{
  "Technical_Mapping": {
    "Reporting_Service_Model": "MAC | KPM | RLC | RC | SLICE",
    "Telemetry_Variables": {
      "// description": "A hierarchical JSON object reflecting the streaming dataset structure",
      "slices": [
        {
          "id": "uint32_t",
          "label": "char*",
          "params": {
            "type": "slice_algorithm_e /* enum */",
            "...": "..."
          }
        }
      ]
    },
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

def extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return {}

def _extract_mapper_context(blueprint: dict) -> dict:
    """Return only the intent fields needed for technical mapping."""
    return {
        "Intent_Blueprint": blueprint.get("Intent_Blueprint", {})
    }

def module_2_technical_node(state: dict) -> dict:
    """Module 2: Maps NL intent to exact FlexRIC C-variables using Structural RAG."""

    intent_blueprint = state.get("blueprint", {})

    # Include the SM type hint if already determinable from the intent blueprint
    requested_sm = (
        intent_blueprint.get("Intent_Blueprint", {})
        .get("Reporting_Service_Model", "")
    )
    sm_hint = f"\nHint: the likely Service Model is '{requested_sm}'." if requested_sm else ""

    mapper_context = _extract_mapper_context(intent_blueprint)

    prompt_content = (
        f"Here is the Intent Blueprint from Module 1:\n"
        f"{json.dumps(mapper_context, indent=2)}\n\n"
        f"Use semantic_search_summary (2 calls max) to look up the FlexRIC codebase, "
        f"then output the Technical_Mapping JSON.{sm_hint}"
    )

    llm = get_llm()
    mapper_agent = create_react_agent(
        model=llm, 
        tools=tools, 
        prompt=MODULE_2_SYSTEM_PROMPT,
        pre_model_hook=limit_context_window
    )

    try:
        result = mapper_agent.invoke(
            {"messages": [HumanMessage(content=prompt_content)]},
            {"recursion_limit": int(os.getenv("MAPPER_RECURSIVE_LIMIT", max(60, int(os.getenv("RECURSIVE_LIMIT", 20)))))},
        )
        final_text = result["messages"][-1].content
    except Exception as e:
        print(f"Module 2 Error: {e}")
        return {"messages": [AIMessage(content="Failed to map variables. Check logs.")]}

    new_blueprint = copy.deepcopy(intent_blueprint) if isinstance(intent_blueprint, dict) else {}
    parsed = extract_json(final_text)
    
    if parsed:
        if "Technical_Mapping" in parsed:
            new_blueprint["Technical_Mapping"] = parsed["Technical_Mapping"]
        elif "Telemetry_Variables" in parsed:
            new_blueprint["Technical_Mapping"] = parsed
        else:
            print("Module 2 Warning: JSON found but missing Technical_Mapping/Telemetry_Variables.")
    else:
        print("Module 2 Warning: Failed to parse any JSON from the LLM output.")

    # Return a summarized AIMessage instead of the full talkative output
    # to keep the main graph's context window clean.
    sm_name = parsed.get("Technical_Mapping", {}).get("Reporting_Service_Model", "Unknown")
    summary = f"Module 2: Technical mapping for Service Model '{sm_name}' completed."

    return {
        "blueprint": new_blueprint,
        "messages": [AIMessage(content=summary)],
    }
