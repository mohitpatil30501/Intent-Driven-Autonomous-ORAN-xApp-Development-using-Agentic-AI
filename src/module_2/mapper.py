import json
import re
import os
import requests
from typing import Any, Dict
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

RECURSIVE_LIMIT = 50

MODULE_2_SYSTEM_PROMPT = """You are "Module 2: The O-RAN Technical Mapper" in an automated xApp development pipeline.
You will receive an "Intent_Blueprint" written in plain English by a Business Analyst. 

Your ONLY job is to map the "requested_Telemetry_NL" and "target_Action_What_NL" into EXACT FlexRIC Service Models and C-struct variables. 

CRITICAL RULES - NO HALLUCINATIONS:
1. You MUST use your search tools (`semantic_code_search` and `exact_keyword_search`) to explore the local FlexRIC codebase.
2. NEVER guess a variable name. If the user wants "throughput", use your semantic search to find the MAC or KPM header files, then use exact search to verify variables like `dl_aggr_tbs` or `bytes_rx` exist.
3. Determine the Action Space by looking for control message structs (e.g., `mac_cb_control` or `rc_cb_control`). 
4. If a requested metric or action is NOT currently implemented in the FlexRIC codebase, do not invent it. Map it to the closest available metric, or note that it is unsupported.

--- SEARCH STRATEGY ---
Step 1: Use `semantic_code_search("Service Model MAC indication header")` or similar to find the telemetry struct definitions.
Step 2: Read the results to find exact C variables mapping to the human's NL intent.
Step 3: Use `semantic_code_search("Service Model MAC control header")` to find how to execute the target action.

--- RESPONSE FORMAT ---
First, output a brief summary of what you searched for and what you found. 
Then, output a strict JSON code block containing BOTH the original "Intent_Blueprint" and your new "Technical_Mapping".

```json
{
  "Intent_Blueprint": { 
    "... (Inherited from Module 1)" 
  },
  "Technical_Mapping": {
    "Reporting_Service_Model": "string (e.g., 'MAC', 'KPM', 'RLC' - The SM used to read telemetry)",
    "Telemetry_Variables": [
      {
        "NL_name": "string (From Module 1, e.g., 'per slice throughput')",
        "C_variable": "string (Exact FlexRIC struct variable, e.g., 'dl_aggr_tbs')",
        "data_type": "string (e.g., 'uint32_t')"
      }
    ],
    "Control_Service_Model": "string (e.g., 'MAC', 'RC' - The SM used to send actions)",
    "Action_Space_Menu": [
      {
        "action_id": "string (A clean name, e.g., 'UPDATE_SLICE_PRB')",
        "description": "string (What it does)",
        "parameters": {
          "variable_name": "data_type (e.g., 'slice_id': 'uint32_t', 'dl_prb_ratio': 'uint8_t')"
        }
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

SEARCH_ENGINE_URL = "http://localhost:7080"
ORIOSEARCH_URL = "http://localhost:8000"

@tool
def semantic_code_search(nl_query: str, max_results: int = 3) -> str:
    """
    Use this to find code based on intent or concepts. 
    Example: 'Where are MAC Service Model variables defined?'
    """
    try:
        res = requests.post(
            f"{SEARCH_ENGINE_URL}/semantic_search", 
            json={"query": nl_query, "n_results": max_results}
        )
        if res.status_code == 200:
            return res.json().get("results", "No results found.")
        return f"Error: Status code {res.status_code}"
    except Exception as e:
        return f"Error connecting to search engine: {e}"

@tool
def exact_keyword_search(keyword: str, max_results: int = 5) -> str:
    """
    Use this to find EXACT references to a specific C-struct, variable, or function name.
    Example: 'dl_aggr_tbs' or 'mac_ind_data'
    """
    try:
        res = requests.post(
            f"{SEARCH_ENGINE_URL}/exact_search", 
            json={"query": keyword, "n_results": max_results}
        )
        if res.status_code == 200:
            return res.json().get("results", "No matches found.")
        return f"Error: Status code {res.status_code}"
    except Exception as e:
        return f"Error connecting to search engine: {e}"

@tool
def restricted_domain_search(query: str, domain: str = "o-ran-sc.org") -> str:
    """
    Use this to search the web for O-RAN concepts or external documentation within a restricted domain.
    """
    try:
        res = requests.get(
            f"{ORIOSEARCH_URL}/search", 
            params={"q": f"{query} site:{domain}"}
        )
        if res.status_code == 200:
            data = res.json()
            results = data.get("results", [])
            if not results:
                return "No results found."
            formatted = []
            for r in results[:3]:
                formatted.append(f"Title: {r.get('title')}\nSnippet: {r.get('content')}")
            return "\n\n".join(formatted)
        return f"Error: Status code {res.status_code}"
    except Exception as e:
        return f"Error connecting to oriosearch: {e}"

tools = [semantic_code_search, exact_keyword_search, restricted_domain_search]

def get_llm():
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=ollama_model, base_url=ollama_url)

def module_2_technical_node(state: dict) -> dict:
    """Module 2: Maps NL intent to exact FlexRIC C-variables using Code Search."""
    
    global RECURSIVE_LIMIT
    # Isolate the blueprint from state
    intent_blueprint = state.get("blueprint", {})
    
    prompt_content = f"Here is the Intent Blueprint from Module 1:\n{json.dumps(intent_blueprint, indent=2)}\n\nPlease map this to FlexRIC variables."
    
    llm = get_llm()
    
    # Create a tightly constrained ReAct agent JUST for searching
    mapper_agent = create_react_agent(
        model=llm, 
        tools=tools, 
        prompt=MODULE_2_SYSTEM_PROMPT
    )
    
    # Run the agent with a strict recursion limit (e.g., max 5 tool calls)
    try:
        result = mapper_agent.invoke(
            {"messages": [HumanMessage(content=prompt_content)]},
            {"recursion_limit": RECURSIVE_LIMIT}
        )
        final_text = result["messages"][-1].content
        
    except Exception as e:
        print(f"Module 2 Search Limit Hit or Error: {e}")
        # In a real app, you might route to a human here to manually input the variables
        return {"messages": [AIMessage(content="Failed to map variables. Check logs.")]}

    # Extract the JSON block
    new_blueprint = intent_blueprint
    json_match = re.search(r'```json\s*(.*?)\s*```', final_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if "Technical_Mapping" in parsed:
                new_blueprint = parsed
            elif "Intent_Blueprint" in parsed and "Technical_Mapping" in parsed:
                new_blueprint = parsed
        except json.JSONDecodeError:
            pass

    # Save the enriched blueprint back to state
    return {
        "blueprint": new_blueprint,
        "messages": [AIMessage(content=final_text)]
    }
