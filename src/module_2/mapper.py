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

from tools.deployer.testbed.introspection_tool import inspect_service_model_runtime

MODULE_2_SYSTEM_PROMPT = """You are "Module 2: The O-RAN Technical Mapper".
Your ONLY job is to map requested telemetry to EXACT FlexRIC Service Model variables.

--- MANDATORY VERIFICATION STRATEGY ---
1. IDENTIFY SM: Determine if the request belongs to MAC, KPM, RLC, or SLICE.
2. CALL inspect_service_model_runtime: Call this tool for the identified SM. 
   - Start with `max_depth=3`.
   - If the output contains `"..."` at a level where you expect your target variables to be, you MUST call it again with `max_depth=5` or `max_depth=7`.
3. HANDLE ARRAYS: If the schema shows a list (e.g., `"slices": [...]`), your mapping MUST reflect that it is an array.
4. MAP EXACTLY: Use the EXACT keys and nesting found in the tool's output.
5. NO HALLUCINATIONS: Do not invent attributes. If the runtime tool doesn't show it, search the codebase with `semantic_search_detailed`.

--- RESPONSE FORMAT ---
Output a strict JSON code block:
```json
{
  "Technical_Mapping": {
    "Reporting_Service_Model": "...",
    "Telemetry_Variables": {
       "//": "Use the EXACT hierarchical structure from the inspection tool",
       "some_list": [
         { "field1": "type", "field2": "type" }
       ],
       "nested_struct": {
         "fieldA": "type"
       }
    },
    "Control_Service_Model": "...",
    "Action_Space_Menu": [ ... ]
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
    


tools = [semantic_search_summary, semantic_search_detailed, restricted_domain_search, inspect_service_model_runtime]


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
        f"MANDATORY FIRST STEP: Call `inspect_service_model_runtime` for the likely Service Model ({requested_sm or 'identify it first'}).\n"
        f"If the output is too shallow (contains '...'), call it again with higher `max_depth` (e.g. 5 or 7).\n"
        f"If the schema contains arrays (e.g. lists), you MUST reflect them as such in your mapping.\n"
        f"Use the returned schema as the absolute source of truth for your Technical_Mapping JSON."
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
