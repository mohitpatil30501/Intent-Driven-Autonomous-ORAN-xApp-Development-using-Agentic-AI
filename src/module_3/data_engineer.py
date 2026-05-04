import json
import re
import os
import sys
import copy
from typing import Any, Dict
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.workspace.workspace_tools import workspace_tools
from tools.semantic_search.semantic_search_tool import semantic_search_summary
from tools.context_utils import limit_context_window

DATA_ENGINEER_SYSTEM_PROMPT = """You are "Module 3: The O-RAN Data Engineer" in an automated xApp development pipeline.
Your job is to provide the necessary datasets based on the provided Blueprint, Technical Mapping, and the User's Data Availability.

The user may want to use an existing dataset for all data, synthesize all data, or mix them (e.g., use an existing dataset for ML training/testing, but synthesize streaming data).

CRITICAL RULE — RAN-REPORTABLE COLUMNS ONLY:
An xApp deployed on a real RAN can ONLY receive metrics that FlexRIC can actually report.
- REQUIRED columns (from the leaf nodes of `Technical_Mapping.Telemetry_Variables`) are already FlexRIC-validated.
- ADDITIONAL columns from an existing dataset must be verified using `semantic_search_summary` before inclusion. A column whose name appears in the FlexRIC codebase (.h or .c file) is FLEXRIC_VALID. All others are EXCLUDED.
- Administrative columns (IPs, MACs, timestamps, user IDs, URLs) are NEVER included.

CRITICAL RULE — MINIMAL TOOL USAGE:
Do NOT use tools unnecessarily. Rely entirely on the Blueprint and Technical Mapping provided in your prompt. Use tools ONLY when absolutely necessary (e.g., reading user-provided dataset directories or writing output files). If the user requests purely synthetic data, skip dataset discovery and just write the generation script.

CRITICAL RULE — DATA SYNTHESIS BLUEPRINT (NO RANDOM NOISE):
Your synthesized data MUST be mathematically and logically consistent. Avoid `np.random.rand()` for primary signals.
1. TEMPORAL CONTINUITY: Streaming and ML data are time-series. Each row must depend on the previous row. Use random walks (`np.cumsum(np.random.normal(0, scale, size))`) or Sine waves with noise to ensure stability.
2. CROSS-FEATURE CORRELATION: Features must relate to each other. Define a 'master' variable (e.g., `num_ues`) and derive others: `throughput = num_ues * base_kbps * (1 - congestion_factor)`. If `latency` is a variable, it should rise when `throughput` drops or `buffer_occupancy` rises.
3. GOAL-DRIVEN EVENTS: If the `Intent_Blueprint.goal` involves anomaly detection, inject a specific, recognizable anomaly pattern (e.g., a 5-minute spike/drop) at a specific timestamp.
4. VECTORIZED LOGIC: Use `numpy` to generate arrays of signals first, then combine them into a DataFrame.
   Example Logic:
   ```python
   t = np.linspace(0, 10, num_rows)
   base_load = 50 + 20 * np.sin(t) # Diurnal cycle
   noise = np.random.normal(0, 2, num_rows)
   load = base_load + noise
   latency = 5 + (load ** 1.5) / 100 + np.random.normal(0, 0.5, num_rows)
   ```
5. STRUCTURAL INTEGRITY: The streaming JSON must exactly map these correlated values into the hierarchical `Technical_Mapping` schema.

--- VERIFICATION MANDATE (TRUST BUT VERIFY) ---
Your work is incomplete and a FAILURE until you have:
1. CREATED the `data/` and `log/` directories using your tools.
2. WRITTEN the `data/build_streaming_datasets.py` script.
3. EXECUTED that script using `python3 data/build_streaming_datasets.py`.
4. VERIFIED the output files exist and contain the correct columns/data using `ls` and `head`.
5. READ the terminal output for any errors. If it fails (e.g., "Directory does not exist"), YOU MUST FIX IT and re-run.

--- STRICT WORKFLOW INSTRUCTIONS ---
Execute the following steps IN ORDER using your tools.

STEP 1: ANALYZE & PREPARE
- Read the "User Data Availability" string to understand what data needs to be synthetic and what is provided as paths.
- Note `Intent_Blueprint.cycle_Type`. If "Pure_Logic", ONLY `data/streaming_mock_data.json` is needed. If "Supervised_ML" or "Unsupervised_ML", you ALSO need `data/historical_training_data.csv` and `data/test_data.csv`.
- **MANDATORY**: Create the `data/` and `log/` directories immediately if they do not exist.

STEP 2: DISCOVER & PROFILE PROVIDED DATASETS (If any user paths are provided)
- Discover files in the provided path(s).
- Load headers only.
- Write and execute a python script (`data/pre_filter.py`) to pre-filter columns (drop object dtypes, admin/timestamp columns, zero-variance columns).
- Match the surviving columns to the required leaf node names extracted from `Technical_Mapping.Telemetry_Variables`.
- For ML datasets, use `semantic_search_summary` to FlexRIC-validate any remaining unmatched columns.

STEP 3: GENERATE / MERGE DATAFRAMES
- Write a python script `data/build_streaming_datasets.py` (using pandas, numpy, and json). This script MUST:
  a. Load provided datasets (if any) and select ONLY the matched required columns and FLEXRIC_VALID columns. Rename them to their exact leaf node names.
  b. Synthesize any missing required leaf nodes using the **Data Synthesis Blueprint** above.
  c. **CRITICAL**: If you are synthesizing both the ML datasets and the streaming JSON, they MUST share the same underlying mathematical distribution and correlation logic. The streaming data should ideally be a continuation or a representative slice of the same signal used for training.
  d. For SYNTHETIC streaming data, generate a JSON file `data/streaming_mock_data.json` with 100-500 items. It MUST be a JSON array `[ {"timestamp": 1600000000, "data": { <Module 2 Telemetry_Variables schema> }}, ... ]`.
  e. If ML is required and the user wants SYNTHETIC ML data, generate 5000 rows for `data/historical_training_data.csv` and 1000 rows for `data/test_data.csv`.
  f. Save the required files to `data/`.
- **MANDATORY EXECUTION**: Run `python3 data/build_streaming_datasets.py` using your tools. If it fails, fix the code and re-run.

STEP 4: CROSS-VALIDATE EVERY OUTPUT FILE
- **MANDATORY**: Use `ls data/` to confirm files are created.
- For the CSVs, run a Python one-liner to print shape and head(3).
- For the streaming JSON, run a one-liner to print its length and first 2 items.
- Mandatory checks:
  - All required leaf node columns are present in the datasets (prior to JSON restructuring).
  - No column is all-zero or constant.
  - If Supervised_ML, ensure a label column exists in both training and test CSVs.

STEP 5: DETERMINE training_data_profile
- Set to one of:
  - "pure_logic" (if Pure_Logic)
  - "supervised_labeled"
  - "supervised_synthesized_labels"
  - "unsupervised_mixed"
  - "unsupervised_unlabeled_test"

--- RESPONSE FORMAT ---
ONLY after successfully verifying EVERY output CSV, output a final JSON block:
```json
{
  "Data_Paths": {
    "streaming_mock_data_path": "data/streaming_mock_data.json",
    "historical_training_data_path": "data/historical_training_data.csv", // or null if Pure_Logic
    "test_data_path": "data/test_data.csv", // or null if Pure_Logic
    "test_label_column": "label", // or null
    "training_data_profile": "<value from Step 5>",
    "profiler_notes": "<Describe data sources used, synthesized columns, row counts>"
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


def _extract_data_context(blueprint: dict) -> dict:
    """Return only the fields needed for data engineering."""
    return {
        "Intent_Blueprint": blueprint.get("Intent_Blueprint", {}),
        "Technical_Mapping": blueprint.get("Technical_Mapping", {})
    }

def module_3_data_node(state: dict) -> dict:
    """Module 3: Unified Data Engineer handling synthetic, profiled, and mixed datasets."""
    blueprint = copy.deepcopy(state.get("blueprint", {})) if isinstance(state.get("blueprint"), dict) else {}
    user_input = state.get("user_dataset_input", "")

    data_context = _extract_data_context(blueprint)

    prompt_content = (
        f"Here is the complete Blueprint and Technical Mapping:\n"
        f"{json.dumps(data_context, indent=2)}\n\n"
        f"User Data Availability:\n"
        f"`{user_input}`\n\n"
        f"Execute all steps: analyze, discover/profile (if paths provided), generate/merge, cross-validate, "
        f"and return the Data_Paths JSON block.\n"
        f"IMPORTANT: Create a `log/` directory in the workspace and save your output "
        f"to `log/module_3_data_engineer.log` using your tools."
    )

    llm = get_llm()
    data_tools = workspace_tools + [semantic_search_summary]

    data_agent = create_react_agent(
        model=llm,
        tools=data_tools,
        prompt=DATA_ENGINEER_SYSTEM_PROMPT,
        pre_model_hook=limit_context_window
    )

    try:
        recursion_limit = int(os.getenv("RECURSIVE_LIMIT", 40))
        result = data_agent.invoke(
            {"messages": [HumanMessage(content=prompt_content)]},
            {"recursion_limit": recursion_limit},
        )
        final_text = result["messages"][-1].content
    except Exception as e:
        print(f"Module 3 Error (Data Engineer): {e}")
        return {"messages": [AIMessage(content=f"Data engineering failed: {e}")]}

    # Extract the JSON block
    data_paths = {}
    json_match = re.search(r"```json\s*(.*?)\s*```", final_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            data_paths = _finalize_data_paths(parsed.get("Data_Paths", {}), blueprint)
            blueprint["Data_Paths"] = data_paths
        except json.JSONDecodeError:
            print("Failed to parse JSON from Data Engineer output.")

    # Return a summary to keep main state clean
    profile = data_paths.get("training_data_profile", "unknown")
    summary = f"Module 3: Data engineering complete. Profile: {profile}. Paths saved to blueprint."

    return {
        "blueprint": blueprint,
        "messages": [AIMessage(content=summary)],
    }
