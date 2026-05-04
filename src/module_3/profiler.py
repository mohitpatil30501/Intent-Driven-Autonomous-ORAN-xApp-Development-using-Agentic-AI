import json
import re
import os
import sys
from typing import Any, Dict
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.workspace.workspace_tools import workspace_tools
from tools.semantic_search.semantic_search_tool import semantic_search_summary
from tools.context_utils import limit_tool_messages
from module_3.synthesizer import _finalize_data_paths, get_llm

DATASET_PROFILER_SYSTEM_PROMPT = """You are "Module 3b: The O-RAN Dataset Profiler" in an automated xApp development pipeline.
The user has provided an EXISTING dataset. Your job is to discover, pre-filter, FlexRIC-validate,
map, merge, and ingest it so that downstream modules receive the exact same Data_Paths structure
as if the data had been synthesized by Module 3.

CRITICAL RULE — RAN-REPORTABLE COLUMNS ONLY:
An xApp deployed on a real RAN (srsRAN via FlexRIC service models) can ONLY receive metrics that
FlexRIC can actually report. Training an ML model on columns that FlexRIC cannot report will cause
the deployed xApp to fail at inference time. Therefore:
  - REQUIRED columns (from Technical_Mapping.Telemetry_Variables[*].C_variable) are already
    FlexRIC-validated by Module 2. Always include these.
  - ADDITIONAL columns (for ML enrichment, Supervised_ML or Unsupervised_ML only) must be
    verified using semantic_search_summary before inclusion. A column whose name appears in the
    FlexRIC codebase (.h or .c file) is FLEXRIC_VALID. All others are EXCLUDED.
  - Administrative columns (IPs, MACs, timestamps, user IDs, sequence numbers, URLs) are NEVER
    included regardless of FlexRIC verification.

--- STRICT WORKFLOW INSTRUCTIONS ---
Execute the following 9 steps IN ORDER using your tools.

STEP 1 — DISCOVER FILES
  a. Verify the user path exists and determine its type:
       python3 -c "import os; p='<USER_PATH>'; print('file' if os.path.isfile(p) else 'dir' if os.path.isdir(p) else 'missing')"
  b. If directory, find all tabular files recursively:
       find <USER_PATH> \\( -name "*.csv" -o -name "*.tsv" -o -name "*.parquet" -o -name "*.xlsx" \\) -type f | sort
  c. If the path is missing or contains no tabular files:
     Output the Data_Paths JSON with all paths set to null and
     profiler_notes="USER_PATH_NOT_FOUND: <path>". STOP immediately.

STEP 2 — LOAD HEADERS ONLY (zero-row read — fast even for 280-column datasets)
  For each discovered file, read ONLY the header row:
    python3 -c "import pandas as pd; df=pd.read_csv('<FILE>', nrows=0); print(list(df.columns))"
  (Use sep='\\t' for .tsv | pd.read_parquet('<FILE>').columns for .parquet | pd.read_excel('<FILE>', nrows=0) for .xlsx)
  Record: {file_path: [column_list]}. Print this map.

STEP 3 — PYTHON PRE-FILTER (critical for datasets with 100+ columns — do NOT skip)
  Write and execute `data/pre_filter.py`. This script must:
  a. Load a SAMPLE of each file (nrows=500) to detect actual dtypes.
  b. Collect all column names across all files into a single CANDIDATE list.
  c. Drop from the CANDIDATE list any column where:
       - dtype is object or string (non-numeric after sample load)
       - name matches admin/infrastructure patterns (case-insensitive regex):
           r'\\b(ip|mac|addr|url|host|email|uri|src|dst|source|destination)\\b'
       - name matches pure timestamp/ID patterns:
           r'^(time|ts|timestamp|date|datetime|id|uid|uuid|index|row|num|count)$'
       - all 500 sampled values are identical (zero-variance)
       - name contains spaces or characters that are not valid C identifiers (not a-z, A-Z, 0-9, _)
  d. Print: "CANDIDATE COLUMNS after pre-filter: <count> of <total>" and the reduced list.
  Execute: python3 data/pre_filter.py
  Fix and re-run if it crashes.
  The output of this step is the CANDIDATE COLUMN LIST. LLM reasoning (Steps 4-5) only applies to this list.

STEP 4 — MATCH REQUIRED COLUMNS (Technical_Mapping → CANDIDATE COLUMN LIST)
  Required variables: Technical_Mapping.Telemetry_Variables[*].C_variable
  For EACH required C_variable, search the CANDIDATE COLUMN LIST in priority order:
    Priority 1 — Exact match (case-sensitive string equality)
    Priority 2 — Case-insensitive match (lower() comparison)
    Priority 3 — Normalized match: replace hyphens, spaces, dots with underscores, then compare lowercase
    Priority 4 — Semantic match: use your LLM reasoning to identify the same metric under a different name.
                 Only apply if confidence > 90%. Example: "DL_ThroughputBytes" → "dl_aggr_tbs".
                 Document your reasoning explicitly.
    Priority 5 — MISSING: not found in the CANDIDATE list. Record as missing; will be synthesized in Step 7.
  Print a MATCH REPORT dict:
    {c_variable: {status: "exact|case|normalized|semantic|missing", source_col: "...", file: "...", confidence: "high|N/A"}}

STEP 5 — FLEXRIC-VALIDATE ADDITIONAL COLUMNS (ML only — skip this entire step for Pure_Logic)
  If cycle_Type is Supervised_ML or Unsupervised_ML:
  Take all CANDIDATE COLUMN LIST columns that are NOT already matched as required columns (Step 4).
  For each additional candidate, call: semantic_search_summary(query=column_name, n_results=3)
    - If the result string contains a reference to a .h or .c file → mark as FLEXRIC_VALID.
    - If the result is "No matches found" or references only non-C/H files → mark as EXCLUDED.
  Print a VALIDATION REPORT: {column_name: "FLEXRIC_VALID" | "EXCLUDED"}

  IMPORTANT: Do NOT call semantic_search_summary more than once per unique column name.
  If the CANDIDATE list has many columns, process them in batches using a Python loop script
  that writes the results to a file, then read that file. This avoids hitting the recursion limit.

STEP 6 — COUNT ROWS and DETERMINE SPLIT STRATEGY
  For each file that contributes matched required or FLEXRIC_VALID additional columns, count rows:
    python3 -c "import pandas as pd; print('<file>', pd.read_csv('<file>').shape)"

  Split strategy based on Intent_Blueprint.cycle_Type and TOTAL rows across contributing files:

  Pure_Logic → only data/streaming_mock_data.csv is needed (100–500 rows)
    Use the file with the most matched required columns. If >500 rows, randomly sample 300 rows.

  Supervised_ML / Unsupervised_ML → three output files needed:
    Total rows < 1 000  → 70% training / 20% test / 10% streaming (cap streaming at 500)
    Total 1K – 9 999   → 70% training / 20% test / 10% streaming (cap streaming at 500)
    Total ≥ 10 000     → 80% training / 15% test / 5% streaming  (cap streaming at 500)

  Label column detection (Supervised_ML / Unsupervised_ML):
    Look for a column named: label, class, target, y, anomaly, or fault (case-insensitive).
    Found → treat as the label column; preserve it in both training and test splits.
    Not found in Supervised_ML → will be synthesized in Step 7.
    Not found in Unsupervised_ML → note absence; test split will have no label.

STEP 7 — BUILD MERGED DATAFRAME
  Write `data/profile_and_merge.py`. This script MUST:
  i.   Load ONLY the matched + FLEXRIC_VALID columns from source file(s) using usecols= parameter
       to avoid loading the full 280 columns:
         pd.read_csv('<file>', usecols=[list_of_needed_source_columns])
  ii.  Rename matched source columns to their exact C_variable names:
         df.rename(columns={"source_col_name": "c_variable_name"}, inplace=True)
  iii. For EACH required column with status "missing" (from Step 4), synthesize realistic values:
         uint64_t byte counters   → np.random.randint(1_000_000, 100_000_000, size=n)
         uint32_t (PRB / ratios)  → np.random.randint(0, 100, size=n)
         float                    → np.random.uniform(0.0, 1.0, size=n)
       Print: "WARNING: Synthesizing column '<col>' — not found in user dataset."
  iv.  Include all FLEXRIC_VALID additional columns alongside the required ones (ML only).
  v.   Drop ALL other columns. Final DataFrame = required C_variables + FLEXRIC_VALID + label (if present).
  vi.  Split into output DataFrames per Step 6 strategy using pandas sample() with random_state=42.
  vii. Write splits to workspace (all paths relative to workspace root):
         data/streaming_mock_data.csv        (always)
         data/historical_training_data.csv   (ML only)
         data/test_data.csv                  (ML only)
  viii.Print shape and head(3) for every output file.
  Execute: python3 data/profile_and_merge.py
  Fix and re-run if it crashes.

STEP 8 — CROSS-VALIDATE EVERY OUTPUT CSV
  For every CSV written in Step 7, run:
    python3 -c "import pandas as pd; df=pd.read_csv('data/<FILE>'); print(df.shape, df.columns.tolist(), df.head(3))"
  Mandatory checks:
    - All required C_variable columns are present and contain no entirely-NaN column.
    - No column is all-zero or constant.
    - Supervised_ML: label column exists in BOTH training and test CSVs.
    - Unsupervised_ML: if source had a label column, it exists in test CSV; otherwise note absence.
    - Row counts match the intended split within 5% tolerance.
  Fix profile_and_merge.py and re-run if any check fails.

STEP 9 — DETERMINE training_data_profile
  Set training_data_profile to ONE of these values based on actual output:
    "pure_logic"                          if cycle_Type == Pure_Logic
    "supervised_labeled"                  if Supervised_ML AND label column found in source data
    "supervised_synthesized_labels"       if Supervised_ML AND label column was synthesized
    "unsupervised_mixed"                  if Unsupervised_ML AND test split has a label column
    "unsupervised_unlabeled_test"         if Unsupervised_ML AND no label column available

--- RESPONSE FORMAT ---
ONLY after successfully verifying EVERY output CSV in Step 8, output a final JSON block:

```json
{
  "Data_Paths": {
    "streaming_mock_data_path": "data/streaming_mock_data.csv",
    "historical_training_data_path": "data/historical_training_data.csv",
    "test_data_path": "data/test_data.csv",
    "test_label_column": "label",
    "training_data_profile": "<one of the 5 values from Step 9>",
    "profiler_notes": "<REQUIRED: describe column match quality, any synthesized columns, FlexRIC-validated extras, row counts, split ratios used, and any warnings>"
  }
}
```

For Pure_Logic: set historical_training_data_path, test_data_path, and test_label_column to null.
"""


def dataset_profiler_node(state: dict) -> dict:
    """Module 3b: Profiles a user-provided dataset and maps it to the required telemetry variables."""

    blueprint = state.get("blueprint", {})
    user_path = state.get("user_dataset_path", "")

    prompt_content = (
        f"Here is the complete Blueprint and Technical Mapping:\n"
        f"{json.dumps(blueprint, indent=2)}\n\n"
        f"The user has provided the following dataset path: `{user_path}`\n\n"
        f"Execute all 9 steps: discover files, pre-filter columns, FlexRIC-validate additional "
        f"columns, match required telemetry variables, build and validate the merged CSVs in the "
        f"workspace data/ directory, and return the Data_Paths JSON block.\n"
        f"IMPORTANT: Create a `log/` directory in the workspace and save your profiling output "
        f"to `log/module_3_profiler.log` using your tools."
    )

    llm = get_llm()

    # semantic_search_summary is needed to validate additional columns against the FlexRIC codebase
    profiler_tools = workspace_tools + [semantic_search_summary]

    profiler_agent = create_react_agent(
        model=llm,
        tools=profiler_tools,
        prompt=DATASET_PROFILER_SYSTEM_PROMPT,
        pre_model_hook=limit_tool_messages
    )

    try:
        recursion_limit = int(os.getenv("RECURSIVE_LIMIT", 25))
        result = profiler_agent.invoke(
            {"messages": [HumanMessage(content=prompt_content)]},
            {"recursion_limit": recursion_limit},
        )
        final_text = result["messages"][-1].content

    except Exception as e:
        print(f"Module 3b Error (Dataset Profiler): {e}")
        return {"messages": [AIMessage(content=f"Dataset profiling failed: {e}")]}

    # Extract the JSON block and finalize paths using the shared helper from synthesizer
    data_paths = {}
    json_match = re.search(r"```json\s*(.*?)\s*```", final_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            data_paths = _finalize_data_paths(parsed.get("Data_Paths", {}), blueprint)
            blueprint["Data_Paths"] = data_paths
        except json.JSONDecodeError:
            print("Failed to parse JSON from Dataset Profiler output.")

    return {
        "blueprint": blueprint,
        "messages": [AIMessage(content=final_text)],
    }
