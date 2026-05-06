import json
import re
import os
import sys
import copy
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.workspace.workspace_tools import workspace_tools
from tools.semantic_search.semantic_search_tool import (
    semantic_search_summary,
    semantic_search_detailed,
)
from tools.context_utils import limit_context_window

MODULE_6_SYSTEM_PROMPT = """You are "Module 6: The xApp Integrator" in an automated O-RAN development pipeline.
Your ONLY job is to inject the standalone logic created by Module 5 into a deployable FlexRIC xApp.

CRITICAL RULES:
1. You do NOT write algorithms. You ONLY write the mapping "glue" between FlexRIC's C-structs and the Python dictionary expected by `XAppLogic`.
2. Look at the `Technical_Mapping` in the Blueprint to know which Service Model (SM) to use (e.g., MAC, KPM, RLC) and what the exact C-variables are.
3. You must read `flexric_template.py` AND the logic script (usually `logic/core_logic.py`) from the workspace.
4. You must create a SINGLE SCRIPT xApp. Instead of `from logic.core_logic import XAppLogic`, you must copy the ENTIRE `XAppLogic` class definition and any necessary imports from the logic script into `final_xapp.py`.
5. If the logic script loads an ML model, ensure the model loading logic is preserved, but note that the model file itself must be present in the same directory as the script.
6. The search tools (`semantic_search_summary` and `semantic_search_detailed`) return code snippets directly in their response text. Do NOT use `read_file` or any file tool on paths mentioned in search results — those paths are internal to the search index and NOT accessible from the workspace.

--- PLACEHOLDER REPLACEMENT GUIDE ---
1. `{{ SM_CALLBACK_BASE }}` -> E.g., `ric.mac_cb`, `ric.rlc_cb`, `ric.kpm_cb`
2. `{{ TELEMETRY_MAPPING_CODE }}` -> Write python code to extract the C-structs.
   Example: `row_dict['dl_aggr_tbs'] = ind.mac_stats[0].dl_aggr_tbs`
3. `{{ CONTROL_MAPPING_CODE }}` -> Write the action mapping based on the `Action_Space_Menu`.
   Example:
   ```python
   if decision.get("action_id") == "UPDATE_SLICE_PRB":
       ctrl_msg = ric.mac_cb_ctrl_msg_t()
       ctrl_msg.action = 0
       ctrl_msg.dl_prb_ratio = decision["parameters"]["prb_ratio"]
       ric.control_mac_sm(self.node_id, ctrl_msg)
   ```
4. `{{ SM_REPORT_FUNCTION }}` -> E.g., `ric.report_mac_sm`
5. `{{ REPORT_INTERVAL }}` -> E.g., `ric.Interval_ms_10`
6. `{{ SM_RM_REPORT_FUNCTION }}` -> E.g., `ric.rm_report_mac_sm`

--- VERIFICATION MANDATE (TRUST BUT VERIFY) ---
Your work is incomplete and a FAILURE until you have:
1. CREATED the `log/` directory using your tools.
2. WRITTEN the `final_xapp.py` script with all placeholders replaced and the `XAppLogic` class inlined.
3. EXECUTED a syntax check using `python3 -m py_compile final_xapp.py`.
4. READ the terminal output and SAVED it to `log/module_6_integrator.log`.
5. FIXED any syntax errors or missing imports reported by the compiler.

--- WORKFLOW (5 STEPS, execute in order) ---

STEP 1 — RAG LOOKUP:
  Call `semantic_search_summary(query="<SM> SM indication callback struct fields xApp example")`.
  Read the returned signatures directly from the response text.

STEP 2 — READ LOGIC:
  Call `read_file` with the logic script path from the Blueprint (e.g., `logic/core_logic.py`).

STEP 3 — READ TEMPLATE:
  Call `read_file` with filename `flexric_template.py`.

STEP 4 — WRITE OUTPUT:
  Call `write_file` with filename `final_xapp.py` containing the fully completed xApp. This includes:
    - All imports from the logic script.
    - The inlined `XAppLogic` class replacing the `{{ INLINED_LOGIC_CODE }}` placeholder.
    - All 6 other template placeholders replaced.

STEP 5 — MANDATORY VERIFICATION:
  Call `terminal_command` with:
    `mkdir -p log && python3 -m py_compile final_xapp.py 2>&1 | tee log/module_6_integrator.log`
  If py_compile reports a syntax error, YOU MUST fix the code and re-run this step.
  Only output the Final_Deployment JSON once the code is valid.

--- RESPONSE FORMAT ---
```json
{
  "Final_Deployment": {
    "status": "SUCCESS",
    "xapp_path": "workspace/final_xapp.py"
  }
}
```
"""

_DEFAULT_RECURSION_LIMIT = 120


def _extract_integration_context(blueprint: dict) -> dict:
    """Return only the fields Module 6 needs, keeping the initial prompt small."""
    intent = blueprint.get("Intent_Blueprint", {})
    return {
        "Intent_Blueprint": {
            "xApp_Name": intent.get("xApp_Name", ""),
            "goal": intent.get("goal", ""),
            "cycle_Type": intent.get("cycle_Type", ""),
        },
        "Technical_Mapping": blueprint.get("Technical_Mapping", {}),
        "Logic_Artifacts": blueprint.get("Logic_Artifacts", {}),
    }


def _pre_model_hook(state: dict) -> dict:
    """Trim old messages before each model call to prevent context explosion."""
    return limit_context_window(state, max_messages=14)


def get_llm():
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=ollama_model, base_url=ollama_url)


def module_6_integrator_node(state: dict) -> dict:
    """Module 6: Wraps the core logic into the FlexRIC Python Template."""

    blueprint = copy.deepcopy(state.get("blueprint", {})) if isinstance(state.get("blueprint"), dict) else {}
    integration_context = _extract_integration_context(blueprint)

    prompt_content = (
        f"Integration Context (Technical Mapping + Logic Artifacts):\n"
        f"{json.dumps(integration_context, indent=2)}\n\n"
        f"Execute the 5-step workflow: "
        f"(1) semantic_search_summary lookup, "
        f"(2) read the logic script, "
        f"(3) read flexric_template.py, "
        f"(4) write final_xapp.py with the XAppLogic class inlined and all 6 placeholders replaced, "
        f"(5) run `mkdir -p log && python3 -m py_compile final_xapp.py 2>&1 | tee log/module_6_integrator.log`. "
        f"Then output the Final_Deployment JSON."
    )

    llm = get_llm()
    module_6_tools = workspace_tools + [semantic_search_summary, semantic_search_detailed]

    integrator_agent = create_react_agent(
        model=llm,
        tools=module_6_tools,
        prompt=MODULE_6_SYSTEM_PROMPT,
        pre_model_hook=_pre_model_hook,
    )

    try:
        recursion_limit = int(os.getenv(
            "INTEGRATOR_RECURSIVE_LIMIT",
            max(_DEFAULT_RECURSION_LIMIT, int(os.getenv("RECURSIVE_LIMIT", _DEFAULT_RECURSION_LIMIT)))
        ))
        result = integrator_agent.invoke(
            {"messages": [HumanMessage(content=prompt_content)]},
            {"recursion_limit": recursion_limit}
        )
        final_text = result["messages"][-1].content

    except Exception as e:
        error_msg = str(e)
        print(f"Module 6 Error (Integration): {error_msg}")
        return {"messages": [AIMessage(content=f"xApp Integration failed: {error_msg}")]}

    # Surface the LangGraph step-limit error clearly instead of propagating the opaque message
    if "need more steps" in final_text.lower():
        print(
            f"Module 6: hit recursion limit ({recursion_limit} steps). "
            f"Set INTEGRATOR_RECURSIVE_LIMIT env var to a higher value."
        )
        return {
            "blueprint": blueprint,
            "messages": [AIMessage(
                content=(
                    f"Integration incomplete: step limit of {recursion_limit} reached. "
                    f"Set INTEGRATOR_RECURSIVE_LIMIT > {recursion_limit} and re-run."
                )
            )],
            "is_complete": False,
        }

    deployment_status = {}
    json_match = re.search(r'```json\s*(.*?)\s*```', final_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            deployment_status = parsed.get("Final_Deployment", {})
            blueprint["Final_Deployment"] = deployment_status
        except json.JSONDecodeError:
            print("Failed to parse JSON from Module 6 output.")

    # Return a summary to keep main state clean
    xapp_path = deployment_status.get("xapp_path", "workspace/final_xapp.py")
    summary = f"Module 6: Integration successful. Final xApp saved to {xapp_path}."

    return {
        "blueprint": blueprint,
        "messages": [AIMessage(content=summary)],
        "is_complete": True
    }
