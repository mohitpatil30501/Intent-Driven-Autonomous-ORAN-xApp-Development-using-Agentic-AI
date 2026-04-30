import json
import re
import os
import sys
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

# Add the src folder to path so we can import from tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.workspace.workspace_tools import workspace_tools
from module_2.mapper import exact_keyword_search

MODULE_6_SYSTEM_PROMPT = """You are "Module 6: The xApp Integrator" in an automated O-RAN development pipeline.
Your ONLY job is to inject the standalone logic created by Module 5 into a deployable FlexRIC xApp.

CRITICAL RULES:
1. You do NOT write algorithms. You ONLY write the mapping "glue" between FlexRIC's C-structs and the Python dictionary expected by `XAppLogic`.
2. Look at the `Technical_Mapping` in the Blueprint to know which Service Model (SM) to use (e.g., MAC, KPM, RLC) and what the exact C-variables are.
3. You must read `flexric_template.py` from the workspace and replace the placeholders.

--- PLACEHOLDER REPLACEMENT GUIDE ---
1. `{{ SM_CALLBACK_BASE }}` -> E.g., `ric.mac_cb`, `ric.rlc_cb`, `ric.kpm_cb`
2. `{{ TELEMETRY_MAPPING_CODE }}` -> Write python code to extract the C-structs. 
   Example: `row_dict['dl_aggr_tbs'] = ind.mac_stats[0].dl_aggr_tbs`
3. `{{ CONTROL_MAPPING_CODE }}` -> Write the action mapping based on the `Action_Space_Menu`.
   Example:
   ```python
   if decision.get("action_id") == "UPDATE_SLICE_PRB":
       ctrl_msg = ric.mac_cb_ctrl_msg_t()
       ctrl_msg.action = 0 # Example action enum
       ctrl_msg.dl_prb_ratio = decision["parameters"]["prb_ratio"]
       ric.control_mac_sm(self.node_id, ctrl_msg)
   ```
4. `{{ SM_REPORT_FUNCTION }}` -> E.g., `ric.report_mac_sm`
5. `{{ REPORT_INTERVAL }}` -> E.g., `ric.Interval_ms_10`
6. `{{ SM_RM_REPORT_FUNCTION }}` -> E.g., `ric.rm_report_mac_sm`

--- STRICT WORKFLOW INSTRUCTIONS ---

1. Use your `exact_keyword_search` tool to look up the specific struct definitions for the `Technical_Mapping.Reporting_Service_Model` to ensure you access the struct properties correctly (e.g., is it `ind.mac_stats` or `ind.ue_stats`?).
2. Write the final integrated script to `final_xapp.py` in the workspace using your file writing tool.
3. SYNTAX CHECK: Run `python3 -m py_compile final_xapp.py` using your `terminal_command` tool to ensure there are no indentation or syntax errors. (Do NOT execute it directly, as `ric.init()` will fail if a real RIC is not running).

--- RESPONSE FORMAT --- 
Once the syntax check passes, output a final JSON confirming completion.

```json
{
  "Final_Deployment": {
    "status": "SUCCESS",
    "xapp_path": "workspace/final_xapp.py"
  }
}
```
"""

def get_llm():
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=ollama_model, base_url=ollama_url)

def module_6_integrator_node(state: dict) -> dict:
    """Module 6: Wraps the core logic into the FlexRIC Python Template."""
    
    blueprint = state.get("blueprint", {})
    
    prompt_content = (
        f"Here is the complete Blueprint, including Technical Mapping and Logic Artifacts:\n"
        f"{json.dumps(blueprint, indent=2)}\n\n"
        f"Read `flexric_template.py` from the workspace, replace the placeholders, write it to `final_xapp.py` in the workspace, "
        f"run a syntax check using py_compile, and return the Final_Deployment JSON.\n"
        f"IMPORTANT: Create a `log/` directory in the workspace and save the terminal output of your py_compile syntax check to `log/module_6_integrator.log` using your tools."
    )
    
    llm = get_llm()
    
    # Tools for Module 6
    module_6_tools = workspace_tools + [exact_keyword_search]
    
    # Create the ReAct agent
    integrator_agent = create_react_agent(
        model=llm, 
        tools=module_6_tools, 
        prompt=MODULE_6_SYSTEM_PROMPT
    )
    
    try:
        recursion_limit = int(os.getenv("RECURSIVE_LIMIT", 20))
        result = integrator_agent.invoke(
            {"messages": [HumanMessage(content=prompt_content)]},
            {"recursion_limit": recursion_limit}
        )
        final_text = result["messages"][-1].content
        
    except Exception as e:
        print(f"Module 6 Error (Integration): {e}")
        return {"messages": [AIMessage(content=f"xApp Integration failed: {e}")]}

    # Extract the JSON block
    deployment_status = {}
    json_match = re.search(r'```json\s*(.*?)\s*```', final_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            deployment_status = parsed.get("Final_Deployment", {})
            blueprint["Final_Deployment"] = deployment_status
        except json.JSONDecodeError:
            print("Failed to parse JSON from Module 6 output.")
            pass

    return {
        "blueprint": blueprint,
        "messages": [AIMessage(content=final_text)],
        "is_complete": True
    }
