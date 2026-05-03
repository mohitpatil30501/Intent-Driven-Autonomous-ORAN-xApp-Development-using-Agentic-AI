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

MODULE_5_SYSTEM_PROMPT = """You are "Module 5: The Core Programmer" in an automated xApp development pipeline.
Your ONLY job is to write the standalone algorithmic brain of the xApp.

CRITICAL RULES - PURE LOGIC ONLY:
1. DO NOT write any FlexRIC, E2, or networking code (no `ric.subscribe()`, no `ric.control()`). 
2. Your code must be 100% independent, stateful, and object-oriented.
3. Every decision your code makes MUST exactly match one of the schemas in `Technical_Mapping.Action_Space_Menu`.

--- STRICT WORKFLOW INSTRUCTIONS ---
You must use your tools to execute the following steps in order:

1. CREATE DIRECTORY: Create a folder named `logic/` inside the workspace.
2. WRITE SCRIPT: Write `logic/core_logic.py`. This script MUST contain:
   - A class named `XAppLogic`.
   - An `__init__(self)` method. ONLY load a `.pkl` model file if `ML_Model_Artifacts` explicitly exists in the blueprint. If `cycle_Type` is `Pure_Logic` or `ML_Model_Artifacts` is absent, the `__init__` must be empty (no model loading, no sklearn imports). IGNORE any `model_acceptance_criteria` field — it is irrelevant for Pure_Logic.
   - A method `process_interval(self, row_dict)` that takes a dictionary (representing one timestep of KPM data).
   - `process_interval` MUST return a dictionary matching an action from the `Action_Space_Menu` (e.g., `{"action_id": "UPDATE_SLICE_PRB", "parameters": {"slice_id": 1, "prb_ratio": 80}}`).
   
3. WRITE TEST LOOP: At the bottom of `logic/core_logic.py`, write an `if __name__ == '__main__':` block that:
   - Loads `Data_Paths.streaming_mock_data_path` using pandas.
   - Instantiates `XAppLogic()`.
   - Iterates through the CSV row by row, converting each row to a dictionary, and passes it to `process_interval()`.
   - Prints the output decisions.

4. EXECUTE SCRIPT: Run `python3 logic/core_logic.py`. If it crashes (e.g., KeyError, model shape mismatch), read the terminal error, fix the script, and run it again.
   - **CRITICAL**: Read the output of your test loop! The logic MUST NOT simply return `DO_NOTHING` for every single row. If it doesn't trigger the expected action for at least some rows (based on anomaly spikes in the data), your logic threshold or logic implementation is wrong. You MUST rewrite the logic to properly achieve the goal defined in the Blueprint and run it again.

--- RESPONSE FORMAT ---
ONLY after the script has executed successfully and the logic works, output a final JSON block updating the blueprint.

```json
{
  "Logic_Artifacts": {
    "logic_script_path": "logic/core_logic.py",
    "class_name": "XAppLogic",
    "entry_function": "process_interval"
  }
}
```
"""

def get_llm():
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=ollama_model, base_url=ollama_url)

def module_5_logic_dev_node(state: dict) -> dict:
    """Module 5: Writes and tests the independent Python logic class."""
    
    blueprint = state.get("blueprint", {})
    prompt_content = (
        f"Here is the complete Blueprint, including Technical Mapping, Data Paths, and ML Artifacts:\n"
        f"{json.dumps(blueprint, indent=2)}\n\n"
        f"Create the `logic/` directory in the workspace, write `core_logic.py`, write the testing loop, "
        f"and execute it by giving the streaming mock dataset from the data directory as input to ensure it processes the mock data without errors. Finally, return the Logic_Artifacts JSON.\n"
        f"IMPORTANT: Create a `log/` directory in the workspace and save the terminal output of your script execution (the test loop) to `log/module_5_logic.log` using your tools."
    )
    
    llm = get_llm()
    
    # Create the ReAct agent
    logic_agent = create_react_agent(
        model=llm, 
        tools=workspace_tools, 
        prompt=MODULE_5_SYSTEM_PROMPT
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

    return {
        "blueprint": blueprint,
        "messages": [AIMessage(content=final_text)]
    }
