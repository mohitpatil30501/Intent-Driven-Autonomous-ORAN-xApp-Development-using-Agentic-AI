import json
import re
import os
import sys
import copy
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

# Add the src folder to path so we can import from tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.workspace.workspace_tools import workspace_tools
from tools.context_utils import limit_context_window
from tools.deployer.testbed.introspection_tool import inspect_service_model_runtime

MODULE_5_SYSTEM_PROMPT = """You are "Module 5: The Core Programmer" in an automated xApp development pipeline.
Your ONLY job is to write the standalone algorithmic brain of the xApp.

CRITICAL RULES - PURE LOGIC ONLY:
1. DO NOT write any FlexRIC, E2, or networking code.
2. RUNTIME SCHEMA VERIFICATION (NEW): If the `Technical_Mapping` is unclear about attribute names (e.g., `ind.msg.mac_info` vs `ind.mac_info`), you MUST use the `inspect_service_model_runtime` tool. This will run a probe in the testbed to show you the EXACT Python object structure.
3. Every decision your code makes MUST match the `Action_Space_Menu`.

--- VERIFICATION MANDATE ---
1. PREPARE: Create `logic/` and `log/` folders.
2. INSPECT (IF NEEDED): Use `inspect_service_model_runtime` to verify the `ind` structure.
3. WRITE: Create `logic/core_logic.py`.
4. DEPENDENCIES: If you use any external libraries (numpy, pandas, sklearn, etc.), create a `requirements.txt` file in the workspace root listing them.
5. TEST: Run `python3 logic/core_logic.py` with mock data.
6. SAVE: Save logs to `log/module_5_logic.log`.

--- STRICT WORKFLOW INSTRUCTIONS ---
... (rest of the workflow)
"""

def get_llm():
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=ollama_model, base_url=ollama_url)

def _extract_logic_context(blueprint: dict) -> dict:
    """Return only the fields needed for logic programming."""
    return {
        "Intent_Blueprint": blueprint.get("Intent_Blueprint", {}),
        "Technical_Mapping": blueprint.get("Technical_Mapping", {}),
        "Data_Paths": blueprint.get("Data_Paths", {}),
        "ML_Model_Artifacts": blueprint.get("ML_Model_Artifacts", {})
    }

def module_5_logic_dev_node(state: dict) -> dict:
    """Module 5: Writes and tests the independent Python logic class."""
    
    blueprint = copy.deepcopy(state.get("blueprint", {})) if isinstance(state.get("blueprint"), dict) else {}
    logic_context = _extract_logic_context(blueprint)
    prompt_content = (
        f"Here is the complete Blueprint:\n"
        f"{json.dumps(logic_context, indent=2)}\n\n"
        f"Verify the `ind` structure using `inspect_service_model_runtime` if needed, "
        f"then write and test `core_logic.py`."
    )
    
    llm = get_llm()
    
    # Include the new tool
    module_5_tools = workspace_tools + [inspect_service_model_runtime]
    
    logic_agent = create_react_agent(
        model=llm, 
        tools=module_5_tools, 
        prompt=MODULE_5_SYSTEM_PROMPT,
        pre_model_hook=limit_context_window
    )
    
    try:
        # Recursion limit of 10-20 to allow for coding and debugging
        recursion_limit = int(os.getenv("LOGIC_RECURSIVE_LIMIT", max(40, int(os.getenv("RECURSIVE_LIMIT", 20)))))
        result = logic_agent.invoke(
            {"messages": [HumanMessage(content=prompt_content)]},
            {"recursion_limit": recursion_limit}
        )
        final_text = result["messages"][-1].content
        
    except Exception as e:
        print(f"Module 5 Error (Logic Programming): {e}")
        return {"messages": [AIMessage(content=f"Logic programming failed: {e}")]}

    # Extract the JSON block
    logic_artifacts = {}
    json_match = re.search(r'```json\s*(.*?)\s*```', final_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            logic_artifacts = parsed.get("Logic_Artifacts", {})
            blueprint["Logic_Artifacts"] = logic_artifacts
        except json.JSONDecodeError:
            print("Failed to parse JSON from Module 5 output.")
            pass

    # Return a summary to keep main state clean
    script_path = blueprint.get("Logic_Artifacts", {}).get("logic_script_path", "logic/core_logic.py")
    summary = f"Module 5: Core logic generation and verification complete. Script saved to {script_path}."

    return {
        "blueprint": blueprint,
        "messages": [AIMessage(content=summary)]
    }
