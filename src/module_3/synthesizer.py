import json
import re
import os
import sys
from typing import Any, Dict
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

# Add the src folder to path so we can import from tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.workspace.workspace_tools import workspace_tools
from tools.context_utils import limit_tool_messages

MODULE_3_SYSTEM_PROMPT = """You are "Module 3: The O-RAN Data Engineer" in an automated xApp development pipeline.
Your job is to generate synthetic datasets based on the provided Blueprint and Technical Mapping.

--- STRICT WORKFLOW INSTRUCTIONS ---
You must use your provided tools to execute the following steps in order:

1. CREATE DIRECTORY: Create a folder named `data/` inside the workspace.
2. WRITE SCRIPT: Write a Python script (e.g., `data/generate_data.py`) using `json` and python dictionaries. You MUST use the `write_file` tool to create this script on disk. Do NOT try to run inline python scripts or heredocs (`python -c` or `python - << 'EOF'`) using the `terminal_command` tool.
   - Look at `Technical_Mapping.Telemetry_Variables`. Your generated streaming JSON objects MUST EXACTLY match this hierarchical structure.
   - Each item in the JSON array must be an object containing a `timestamp` and the telemetry object schema from Module 2.
   - Look at `Intent_Blueprint.data_Requirements` for the math/logic needed to simulate anomalies or traffic spikes.
   - If ML is used, the script MUST generate a separate `data/test_data.csv` for model evaluation.
   - If labels are needed for evaluation, use a column named `label` where 0 means normal/no-action behavior and 1 means anomaly/positive-action behavior.
3. EXECUTE SCRIPT: Run the script using your terminal_command tool (e.g., `python3 data/generate_data.py`).
4. CROSS-CHECK (MANDATORY): You MUST verify the data. Run commands or Python one-liners to view the first 2 JSON items.
   - Verify that the JSON structure exactly matches the required hierarchical schema.
   - For ML datasets, verify that `data/test_data.csv` exists and contains representative evaluation rows.
   - For labeled evaluation, verify that the `label` column contains at least two classes when possible.
   - Verify that the data values make mathematical sense based on the requirements.
   - If the data is wrong, rewrite the script and run it again.

--- WHAT TO GENERATE ---
Check `Intent_Blueprint.cycle_Type`. 
If it is "Pure_Logic":
- Generate ONLY `data/streaming_mock_data.json` (approx 100-500 items). It must be a JSON array: `[ {"timestamp": 1600000000, "data": { <Telemetry_Variables structure> }}, ... ]`.
- Do NOT generate model training or test data. Return null for ML-only paths.

If it is "Supervised_ML" or "Unsupervised_ML":
- Generate `data/streaming_mock_data.json` (100-500 items) formatted the same way.
- AND Generate `data/historical_training_data.csv` (EXACTLY 5000 rows) based on the `historical_data_description_NL`. Use numpy to generate all 5000 rows in one vectorized call — do NOT use a Python loop. Include a 'label' column if Supervised_ML. Balance classes: ~50% label=0, ~50% label=1 for Supervised_ML.
- AND Generate `data/test_data.csv` (EXACTLY 1000 rows) for model evaluation. Include 'label' for Supervised_ML.
- IMPORTANT: Use numpy vectorized generation (np.random.randint, np.random.uniform, np.where) for ALL rows in a single script — NOT row-by-row loops. The script must complete in under 10 seconds.
- For Unsupervised_ML anomaly detection: training data may be normal-heavy or unlabeled, but test data SHOULD include a `label` column so Module 4 can compute anomaly F1.
- For autoencoder-style requirements: generate normal-only historical training data and mixed normal/anomaly test data with `label`.
- Choose and report one of these `training_data_profile` values:
  - `supervised_labeled`
  - `unsupervised_mixed`
  - `autoencoder_normal_train_anomaly_test`

--- RESPONSE FORMAT ---
ONLY after you have successfully verified the first 5 rows, output a final JSON block updating the blueprint with the paths to the generated data.
For verification, use oneliner python script to read csv's first 5 row and its headers.
 
```json
{
  "Data_Paths": {
    "streaming_mock_data_path": "data/streaming_mock_data.json",
    "historical_training_data_path": "data/historical_training_data.csv", // or null if Pure_Logic
    "test_data_path": "data/test_data.csv", // or null if Pure_Logic
    "test_label_column": "label", // or null if no labels exist
    "training_data_profile": "supervised_labeled | unsupervised_mixed | autoencoder_normal_train_anomaly_test | pure_logic"
  }
}
```
"""

def get_llm():
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
    return ChatOllama(model=ollama_model, base_url=ollama_url)

def _finalize_data_paths(data_paths: Dict[str, Any], blueprint: Dict[str, Any]) -> Dict[str, Any]:
    cycle_type = blueprint.get("Intent_Blueprint", {}).get("cycle_Type", "Pure_Logic")
    data_paths.setdefault("streaming_mock_data_path", "data/streaming_mock_data.json")

    if cycle_type in ["Supervised_ML", "Unsupervised_ML"]:
        data_paths.setdefault("historical_training_data_path", "data/historical_training_data.csv")
        data_paths.setdefault("test_data_path", "data/test_data.csv")
        data_paths.setdefault("test_label_column", "label")
        if cycle_type == "Supervised_ML":
            data_paths.setdefault("training_data_profile", "supervised_labeled")
        else:
            data_paths.setdefault("training_data_profile", "unsupervised_mixed")
    else:
        data_paths["historical_training_data_path"] = None
        data_paths["test_data_path"] = None
        data_paths["test_label_column"] = None
        data_paths["training_data_profile"] = "pure_logic"
        # Null out ML acceptance criteria so Modules 5/6 LLMs don't misinterpret
        # threshold values as instructions to load or evaluate an ML model.
        intent = blueprint.get("Intent_Blueprint", {})
        intent["model_acceptance_criteria"] = {
            "threshold": None,
            "metric_policy": None,
            "metric_description_NL": None,
        }
        blueprint["Intent_Blueprint"] = intent

    return data_paths

def module_3_data_node(state: dict) -> dict:
    """Module 3: Synthesizes and verifies CSV mock data."""
    
    blueprint = state.get("blueprint", {})
    
    prompt_content = (
        f"Here is the complete Blueprint and Technical Mapping:\n"
        f"{json.dumps(blueprint, indent=2)}\n\n"
        f"Create the `data/` directory, write the python script inside it, generate the CSVs, "
        f"verify the first 5 rows and headers for every generated CSV using the terminal, "
        f"and return the extended Data_Paths JSON.\n"
        f"IMPORTANT: Create a `log/` directory in the workspace and save the terminal output of your script execution to `log/module_3_data.log` using your tools."
    )
    
    llm = get_llm()
    
    # Create the ReAct agent
    data_agent = create_react_agent(
        model=llm, 
        tools=workspace_tools, 
        prompt=MODULE_3_SYSTEM_PROMPT,
        pre_model_hook=limit_tool_messages
    )
    
    try:
        # Recursion limit increased to 8 to account for:
        # 1. mkdir -> 2. write file -> 3. run python -> 4. run head -n 6 -> 5. output JSON
        # Plus a few extra steps if it makes a syntax error and needs to retry.
        result = data_agent.invoke(
            {"messages": [HumanMessage(content=prompt_content)]},
            {"recursion_limit": int(os.getenv("RECURSIVE_LIMIT", 20))}
        )
        final_text = result["messages"][-1].content
        
    except Exception as e:
        print(f"Module 3 Error (Data Gen/Verify): {e}")
        return {"messages": [AIMessage(content=f"Data generation failed: {e}")]}

    # Extract the JSON block
    data_paths = {}
    json_match = re.search(r'```json\s*(.*?)\s*```', final_text, re.DOTALL)
    if json_match:
        try:
            data_paths = _finalize_data_paths(json.loads(json_match.group(1)).get("Data_Paths", {}), blueprint)
            blueprint["Data_Paths"] = data_paths
        except json.JSONDecodeError:
            print("Failed to parse JSON from Module 3 output.")
            pass

    return {
        "blueprint": blueprint,
        "messages": [AIMessage(content=final_text)]
    }
