# Intent-Driven Autonomous O-RAN xApp Development using Agentic AI

This repository implements a LangGraph-based agent pipeline that converts a high-level natural-language network intent into a deployable FlexRIC-style xApp, and optionally deploys it on a local testbed. The system is organized as seven sequential modules, each acting as a specialized agent role: intent analyst, O-RAN mapper, data engineer, ML developer, core logic programmer, FlexRIC integrator, and testbed deployer.

At runtime, the modules collaborate through a shared `blueprint` dictionary. Each stage reads the current blueprint, enriches it with new technical details or generated artifacts, and returns the updated state to the graph.

## High-Level Flow

```text
Human intent
   |
   v
Module 1: Intent Decomposer
   |  Produces Intent_Blueprint
   v
Human review / confirmation  [INTERRUPT]
   |
   v
Module 2: Technical Mapper
   |  Queries Semantic Search (port 7080) for FlexRIC C-structs
   |  Produces hierarchical Telemetry_Variables JSON + Action_Space_Menu
   v
Dataset question  [INTERRUPT]
   |  "Do you have an existing dataset, or should one be synthesized?"
   |  Free-form natural language reply (path, 'no', or a mix)
   v
Module 3: Data Engineer (unified)
   |  Discovers user paths if any, RAN-validates additional columns,
   |  synthesizes missing data with correlated numpy signals,
   |  writes streaming_mock_data.json + ML CSVs
   v
Conditional branch
   |-- Supervised_ML / Unsupervised_ML --> Module 4: ML Developer
   |                                      Runs the pre-written two-stage
   |                                      auto-training script and writes
   |                                      ml/evaluation_report.json
   v
Module 5: Core Programmer
   |  Creates and tests standalone XAppLogic against streaming JSON
   v
Module 6: xApp Integrator
   |  Queries Semantic Search for SM callback patterns
   |  Injects logic into the FlexRIC Python template
   v
Deployment question  [INTERRUPT]
   |  "Proceed with deploying to the testbed?"
   v
Conditional branch
   |-- "Proceed"  --> Module 7: Deployer
   |                  Copies artifacts to nearrtric/xapps,
   |                  rebuilds the Docker image, runs the xApp for 20s,
   |                  captures container logs
   |-- otherwise  --> END
```

The main graph is defined in `src/agent.py`. It uses LangGraph's `StateGraph` and a small set of conditional routing functions to decide when to pause for the human, when to proceed to technical mapping, whether ML training is required, and whether to deploy.

## Repository Layout

```text
.
|-- requirements.txt
|-- QUICKSTART.md
|-- README.md
|-- src
|   |-- agent.py
|   |-- langgraph.json
|   |-- module_1                # Intent Decomposer
|   |-- module_2                # Technical Mapper
|   |-- module_3                # Data Engineer (unified synth + profile)
|   |-- module_4                # ML Developer (auto_train + registry)
|   |-- module_5                # Core Programmer
|   |-- module_6                # xApp Integrator (FlexRIC template)
|   |-- module_7                # Testbed Deployer
|   |-- tools
|   |   |-- semantic_search     # Dockerized FlexRIC code search (Chroma + HF)
|   |   |-- oriosearch          # O-RAN web search (Docker)
|   |   |-- workspace           # File + terminal tools for ReAct agents
|   |   `-- context_utils.py    # Sliding-window message trimmer
|   `-- workspace               # All generated artifacts land here
`-- test
```

The `src/workspace` directory is the controlled working area used by tool-enabled agents. Generated data, scripts, logs, logic code, ML artifacts, and the final xApp file are created there.

## Runtime State

The graph state is declared in `src/agent.py` as:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    blueprint: Dict[str, Any]
    is_complete: bool
    user_dataset_input: str   # Free-form NL describing data availability
```

| Field | Purpose |
|---|---|
| `messages` | Human/assistant conversation accumulated by LangGraph |
| `blueprint` | Structured contract passed between modules |
| `is_complete` | Whether the intent blueprint has enough information to proceed |
| `user_dataset_input` | Set by `receive_dataset` after the dataset interrupt. Can be `'no'`, an absolute path, or a free-form mix such as "Use /path/to/ml_data for training/testing, but synthesize streaming data" |

The pipeline relies on JSON extraction from LLM responses. Most modules ask the LLM/ReAct agent to emit a strict JSON block, parse it, then merge the parsed values back into `blueprint`.

## Orchestration: `src/agent.py`

`src/agent.py` is the conductor. It imports all seven module node functions and wires them into a LangGraph workflow.

Key nodes:

- `intent_decomposer`: runs Module 1 and produces the initial intent blueprint.
- `ask_human`: a placeholder node used as an interrupt point for human-in-the-loop review.
- `technical_mapper`: runs Module 2 after the human confirms the blueprint.
- `ask_dataset`: appends an AIMessage that shows the FlexRIC-validated telemetry schema and asks the user how to source the data (path / `no` / mix).
- `receive_dataset`: reads the user's reply and stores it verbatim in `user_dataset_input`.
- `data_engineer`: runs Module 3 — handles synthesis, profiling of user-provided datasets, and any mix of the two.
- `ml_dev`: conditionally runs Module 4 for ML-based xApps.
- `logic_dev`: runs Module 5 and writes the standalone decision engine.
- `integrator`: runs Module 6 and creates the final FlexRIC-integrated xApp.
- `ask_to_deploy`: appends an AIMessage asking whether to deploy to the testbed.
- `receive_deploy_decision`: interrupt node that pauses for the user's deploy decision.
- `deployer`: runs Module 7 only when the user types `Proceed`.

Important routing functions:

- `should_continue(state)`: always routes from Module 1 to `ask_human`, forcing review before continuing.
- `check_confirmation(state)`: sends the flow to Module 2 only when the blueprint is complete and the latest human message contains `confirm`.
- `should_run_ml(state)`: checks `Intent_Blueprint.cycle_Type`; routes to Module 4 for `Supervised_ML` or `Unsupervised_ML`, otherwise skips directly to Module 5. Module 3 is responsible for nulling-out ML acceptance criteria when the cycle is `Pure_Logic` so this gate sees a clean state.
- `should_deploy(state)`: routes to `deployer` only when the human reply contains `proceed`; otherwise terminates the graph.

The graph is compiled with:

```python
graph = builder.compile(interrupt_before=["ask_human", "receive_dataset", "receive_deploy_decision"])
```

There are three human-in-the-loop interrupt points: blueprint confirmation, the dataset question after Module 2, and the deploy decision after Module 6.

## Module 1: Intent Decomposer

Location: `src/module_1/decomposer.py`

Role: business analyst.

Module 1 interviews the user and converts a high-level requirement into an `Intent_Blueprint`. It is intentionally not allowed to invent FlexRIC implementation details. Its prompt explicitly says not to hallucinate O-RAN service models, C structs, or API variables.

It must capture:

- metrics to monitor,
- action to take,
- objective or reason,
- cycle type: `Pure_Logic`, `Supervised_ML`, or `Unsupervised_ML`,
- Service Model hint (e.g. SLICE, MAC, KPM, RC),
- data behavior, especially for ML scenarios,
- model acceptance criteria for ML scenarios, defaulting to threshold `0.85` with metric policy `task_aware` when the user does not specify one. For `Pure_Logic` workflows all three criteria fields (`threshold`, `metric_policy`, `metric_description_NL`) are set to `null` so downstream modules are not confused into loading a non-existent model.

Main functions:

- `get_llm()`: creates a `ChatOllama` client using `OLLAMA_URL` and `OLLAMA_MODEL`.
- `extract_json(text)`: extracts the JSON code block from the model response.
- `decomposer_node(state, config)`: invokes the LLM, parses the blueprint, and sets `is_complete`. Uses `limit_context_window` to keep the conversation under 14 messages.

Expected output shape:

```json
{
  "Intent_Blueprint": {
    "validation": {
      "isComplete": false,
      "missingFields": [],
      "questionsForHuman": []
    },
    "xApp_Name": "string",
    "code_language": "C++ or Python3",
    "objective_Why": "string",
    "target_Action_What_NL": "string",
    "cycle_Type": "Pure_Logic | Supervised_ML | Unsupervised_ML | TBD",
    "Service Models": ["e.g. SLICE SM", "MAC SM"],
    "requested_Telemetry_NL": [],
    "data_Requirements": {
      "needs_historical_training_data": false,
      "historical_data_description_NL": null,
      "streaming_mock_data_description_NL": "string"
    },
    "model_acceptance_criteria": {
      "threshold": 0.85,
      "metric_policy": "task_aware",
      "metric_description_NL": "..."
    }
  }
}
```

If fields are missing, the graph loops back to this module after the human answers the clarifying questions.

## Module 2: Technical Mapper

Location: `src/module_2/mapper.py`

Role: O-RAN engineer.

Module 2 translates the plain-English blueprint into concrete FlexRIC technical mappings using the **Semantic Search** service (Dockerized Chroma + HuggingFace embeddings, port 7080). It searches for the Indication Message struct that anchors the requested Service Model, then recursively explores every nested struct, union, and enum until all types are primitives.

**Key responsibilities — no hallucinations:**

1. Identify the exact Indication Message struct (`kpm_ind_msg_t`, `slice_ind_msg_t`, etc.) for the requested Service Model.
2. Trace nested types: when a field is a `struct*` or `union`, look up its definition.
3. Expose union variants and enum discriminators so the streaming payload structure is fully visible.
4. Produce a hierarchical JSON object reflecting the actual C nesting — not a flat list.
5. Map each action in `Action_Space_Menu` to actual C struct/union field types.

Available tools:

- `semantic_search_summary(query, n_results=5)` — primary tool; calls `POST /semantic_search` on the Semantic Search server.
- `semantic_search_detailed(query, n_results=2)` — same endpoint but returns full code bodies; reserved for digging into a specific struct definition once located.
- `restricted_domain_search(query, domain="o-ran-sc.org")` — web search via OrioSearch; used as a fallback when codebase search is insufficient.

Main function:

- `module_2_technical_node(state)`: builds a Semantic-Search-grounded ReAct agent, passes it the Intent Blueprint only (other blueprint fields are stripped to keep the prompt small), and parses a `Technical_Mapping` JSON block.

Expected blueprint addition (hierarchical, not flat):

```json
{
  "Technical_Mapping": {
    "Reporting_Service_Model": "SLICE",
    "Telemetry_Variables": {
      "slices": [
        {
          "id": "uint32_t",
          "label": "char*",
          "params": {
            "type": "slice_algorithm_e /* enum */",
            "u": {
              "STATIC": { "pos_low": "uint32_t", "pos_high": "uint32_t" },
              "NVS":    { "conf": "nvs_slice_conf_e", "u": { "RATE": { "u_rate": {"mb_rsvd": "float", "mb_max": "float"} } } }
            }
          }
        }
      ]
    },
    "Control_Service_Model": "SLICE",
    "Action_Space_Menu": [
      {
        "action_id": "UPDATE_SLICE_PRB",
        "description": "Modify PRB allocation",
        "parameters": {
          "slice_id": "uint32_t",
          "dl_prb_ratio": "uint8_t"
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

This module is the safety barrier against invented FlexRIC symbols. The prompt requires search before mapping and tells the agent to trace every non-primitive type recursively rather than emit template placeholders.

## Module 3: Data Engineer (unified)

Location: `src/module_3/data_engineer.py`

Role: data engineer.

After Module 2 completes, the graph pauses and shows the FlexRIC-validated telemetry schema, then asks:

> "Please specify your data availability. Examples:
> • 'Generate all synthetic data' (or 'no')
> • 'Use /path/to/data/ for all data'
> • 'Use /path/to/ml_data/ for training/testing, but synthesize streaming data'"

A single unified node — `module_3_data_node` — handles **all three modes** (full synthesis, full profile, or any mix). The user's free-form reply is stored in `state["user_dataset_input"]` and passed straight to the agent.

**Critical rules enforced by the prompt:**

- *RAN-reportable columns only.* Required columns come from the leaf nodes of `Technical_Mapping.Telemetry_Variables` (already FlexRIC-validated by Module 2). Additional columns from a user dataset must be verified with `semantic_search_summary` against the FlexRIC codebase before inclusion. Admin/infrastructure columns (IPs, MACs, timestamps, user IDs, URLs) are always excluded.
- *No random noise for primary signals.* Synthesis must use temporally continuous random walks or sine + noise, derive features from a master variable (e.g. `throughput = num_ues * base_kbps * (1 - congestion_factor)`), inject goal-driven anomaly patterns when the intent calls for them, and use vectorized numpy.
- *Structural integrity.* The streaming JSON must exactly map the correlated values into the hierarchical `Technical_Mapping.Telemetry_Variables` schema produced by Module 2.

**5-step workflow:**

1. Analyze the user's data availability string and `cycle_Type`. Create `data/` and `log/` directories.
2. Discover and pre-filter user-provided datasets (if any paths were given) using a generated `data/pre_filter.py` script.
3. Generate / merge dataframes — write `data/build_streaming_datasets.py` to disk (no inline `python -c` heredocs allowed), then execute it. Synthetic streaming and ML data must share the same underlying signal.
4. Cross-validate every output file with `ls`, `head`, and shape/length checks.
5. Determine `training_data_profile`.

Main function:

- `module_3_data_node(state)`: creates a ReAct agent over `workspace_tools + [semantic_search_summary]`, passes it `{Intent_Blueprint, Technical_Mapping}` plus the user data string, parses the `Data_Paths` JSON block, and runs `_finalize_data_paths` to fill in defaults and null-out ML criteria for `Pure_Logic`.

Row-count targets (synthesis):

| File | Target |
|---|---|
| `streaming_mock_data.json` | 100–500 items |
| `historical_training_data.csv` | 5 000 rows |
| `test_data.csv` | 1 000 rows |

When the user provides ML data, it is split 80% / 20% into training and test instead of synthesized.

Generated artifacts:

- `src/workspace/data/build_streaming_datasets.py`
- `src/workspace/data/pre_filter.py` (only when user paths are profiled)
- `src/workspace/data/streaming_mock_data.json` (hierarchical, matches Telemetry_Variables)
- `src/workspace/data/historical_training_data.csv` (ML workflows)
- `src/workspace/data/test_data.csv` (ML workflows)
- `src/workspace/log/module_3_data_engineer.log`

Shared blueprint addition:

```json
{
  "Data_Paths": {
    "streaming_mock_data_path": "data/streaming_mock_data.json",
    "historical_training_data_path": "data/historical_training_data.csv",
    "test_data_path": "data/test_data.csv",
    "test_label_column": "label",
    "training_data_profile": "supervised_labeled | supervised_synthesized_labels | unsupervised_mixed | unsupervised_unlabeled_test | pure_logic",
    "profiler_notes": "string describing data sources used, synthesized columns, row counts"
  }
}
```

For `Pure_Logic`, the ML-only paths and label column are `null`, `training_data_profile` is `pure_logic`, and the agent additionally clears `Intent_Blueprint.model_acceptance_criteria` so Modules 5/6 do not misinterpret a leftover threshold value as an instruction to load a model.

## Module 4: ML Developer

Location: `src/module_4/ml_dev.py` (orchestrator) and `src/module_4/auto_train.py`, `src/module_4/registry.py` (training engine).

Role: data scientist.

Module 4 runs only when `cycle_Type` is `Supervised_ML` or `Unsupervised_ML`. The training script is **pre-written deterministically** by the orchestrator, not by the LLM — the agent's only job is to execute it and verify the report.

**Two-stage auto-training pipeline:**

1. **Spot-check stage** — `module_4/registry.py` defines a curated list of sklearn algorithms (classifiers, regressors, anomaly detectors). Each runs once with default parameters on the training data.
2. **Tuning stage** — `RandomizedSearchCV` tunes the top performers from stage 1 and re-evaluates them on the held-out test set.

`module_4/auto_train.py` exposes `run_auto_training(...)` which:

- Reads `Data_Paths.historical_training_data_path` and `test_data_path`.
- Selects the metric based on `metric_policy` and the dataset profile (accuracy, F1, anomaly F1, neg_rmse, autoencoder reconstruction separation).
- Writes `ml/evaluation_report.json` and `ml/saved_model.pkl`.

The orchestrator (`ml_dev.py`) writes a thin wrapper at `ml/train.py` that calls `run_auto_training` with the threshold, cycle type, and label hint already substituted in. The ReAct agent then:

1. Ensures `ml/` exists.
2. Runs `python3 ml/train.py`.
3. Verifies `ml/evaluation_report.json` parses.
4. Saves terminal output to `log/module_4_ml.log`.
5. Emits the final JSON block.

**Reliability design:** if the agent crashes or hits its recursion limit, `_recover_artifacts_from_workspace` reads `ml/evaluation_report.json` directly from disk. If no report exists at all, `_write_fallback_report` writes a minimal report with `status: AGENT_FAILED` and `threshold_met: false` so the operator always gets a result.

Main functions:

- `module_4_ml_dev_node(state)`: pre-writes `ml/train.py`, invokes the ReAct agent with `workspace_tools`, parses `ML_Model_Artifacts`, and falls back to workspace recovery if the agent's output is unusable.
- `_write_train_script(...)`: emits `ml/train.py` from a string template, baking in the train/test paths, threshold, cycle type, metric policy, and label hint.
- `ensure_model_acceptance_defaults(blueprint)`: backfills threshold `0.85` and `metric_policy="task_aware"` if older blueprints are missing them.
- `_write_fallback_report(blueprint, error_msg)` and `_recover_artifacts_from_workspace(blueprint)`.

Generated artifacts:

- `src/workspace/ml/train.py` (deterministic wrapper)
- `src/workspace/ml/saved_model.pkl`
- `src/workspace/ml/evaluation_report.json`
- `src/workspace/log/module_4_ml.log`

Expected blueprint addition:

```json
{
  "ML_Model_Artifacts": {
    "model_path": "ml/saved_model.pkl",
    "technique_used": "RandomForestClassifier for Supervised_ML",
    "expected_input_features": ["cqi", "buffer_occupancy"],
    "best_metric_name": "accuracy | f1 | f1_macro | neg_rmse | anomaly_f1 | anomaly_separation_advisory",
    "best_metric_value": 0.91,
    "threshold": 0.85,
    "threshold_met": true,
    "evaluation_report_path": "ml/evaluation_report.json"
  }
}
```

If the threshold is not met after the allowed attempts, the best candidate is still saved and the miss is recorded in `ml/evaluation_report.json`.

## Module 5: Core Programmer

Location: `src/module_5/core_programmer.py`

Role: core logic developer.

Module 5 writes the standalone decision engine used by the final xApp. It must remain independent of FlexRIC and networking APIs. This separation keeps the business logic testable before integration.

Main function:

- `module_5_logic_dev_node(state)`: writes `logic/core_logic.py`, exercises it against `streaming_mock_data.json`, saves the run output to `log/module_5_logic.log`, and returns the logic artifact metadata.

Generated artifacts:

- `src/workspace/logic/core_logic.py`
- `src/workspace/log/module_5_logic.log`

Required class contract:

```python
class XAppLogic:
    def __init__(self):
        ...   # loads ML model only if ML_Model_Artifacts is present; empty for Pure_Logic

    def process_interval(self, row_dict):
        ...   # returns a dict matching one Action_Space_Menu entry
```

The bottom of `core_logic.py` must contain an `if __name__ == '__main__':` block that loads `Data_Paths.streaming_mock_data_path` with `json.load`, instantiates `XAppLogic`, iterates through the JSON array, and calls `process_interval(item["data"])` for each entry. The prompt also enforces that the logic must trigger the expected action for at least some rows — returning `DO_NOTHING` for every row is treated as a failed implementation.

`process_interval(row_dict)` must return a dictionary matching one of the actions in `Technical_Mapping.Action_Space_Menu`, for example:

```json
{
  "action_id": "UPDATE_SLICE_PRB",
  "parameters": {
    "slice_id": 1,
    "prb_ratio": 80
  }
}
```

For `Pure_Logic` workflows, the class uses thresholds or optimization rules; `__init__` is empty. For ML workflows, `__init__` loads the saved model from Module 4 and `process_interval` performs inference.

Expected blueprint addition:

```json
{
  "Logic_Artifacts": {
    "logic_script_path": "logic/core_logic.py",
    "class_name": "XAppLogic",
    "entry_function": "process_interval"
  }
}
```

## Module 6: xApp Integrator

Location: `src/module_6/integrator.py`

Role: FlexRIC integrator.

Module 6 injects the tested `XAppLogic` class into a FlexRIC Python SDK template. It uses the Semantic Search service with one targeted call per Service Model to resolve how to access the correct C-struct fields (e.g. `ind.ue_stats[0].wb_cqi` vs `ind.mac_stats[0].dl_aggr_tbs`).

**Context-budget design:** Only the fields Module 6 actually needs are extracted from the full blueprint before being passed to the agent. Specifically `Intent_Blueprint.{xApp_Name, goal, cycle_Type}`, `Technical_Mapping`, and `Logic_Artifacts`. `Data_Paths`, `ML_Model_Artifacts`, and Module 3–5 log content are excluded from the prompt.

**Message-window trimming:** A `pre_model_hook` runs `limit_context_window(state, max_messages=14)` before each model call. If the internal message list exceeds the cap, it keeps the first message (task + context) and the most recent N − 1, dropping the middle. This prevents context-window overflow on OSS models with 128 K limits.

**Strict 4-step workflow** enforced by the system prompt:

1. `semantic_search_summary` — one RAG call for SM callback patterns. (`semantic_search_detailed` is allowed if the summary is insufficient.) The agent reads code snippets directly from the response and is explicitly told not to call `read_file` on internal index paths.
2. `read_file flexric_template.py` — read the template once.
3. `write_file final_xapp.py` — write the completed xApp with all six placeholders replaced.
4. `terminal_command "mkdir -p log && python3 -m py_compile final_xapp.py 2>&1 | tee log/module_6_integrator.log"` — syntax check and log in one combined command.

**Step-limit error handling:** If the recursion limit is reached, the node detects the LangGraph `"need more steps"` message and returns a clear, actionable error (`"Set INTEGRATOR_RECURSIVE_LIMIT > N and re-run"`) instead of silently propagating the opaque message to the UI.

Main functions:

- `module_6_integrator_node(state)`: extracts the integration context, calls the ReAct agent with message trimming, and parses the `Final_Deployment` JSON.
- `_extract_integration_context(blueprint)`: returns only the three blueprint sections Module 6 needs.
- `_pre_model_hook(state)`: trims the agent's internal message list before each model call.

Template locations:

- `src/workspace/flexric_template.py`
- `src/module_6/flexric_template.py` (committed master copy)

Important placeholders:

- `{{ SM_CALLBACK_BASE }}`
- `{{ TELEMETRY_MAPPING_CODE }}`
- `{{ CONTROL_MAPPING_CODE }}`
- `{{ SM_REPORT_FUNCTION }}`
- `{{ REPORT_INTERVAL }}`
- `{{ SM_RM_REPORT_FUNCTION }}`

Generated artifacts:

- `src/workspace/final_xapp.py`
- `src/workspace/log/module_6_integrator.log`

Expected blueprint addition:

```json
{
  "Final_Deployment": {
    "status": "SUCCESS",
    "xapp_path": "workspace/final_xapp.py"
  }
}
```

The module does not execute the final xApp itself — it only performs a syntax check with `python3 -m py_compile`. Live execution against the testbed is the responsibility of Module 7.

## Module 7: Testbed Deployer

Location: `src/module_7/deployer.py`

Role: deployment engineer.

Module 7 runs only when the user types `Proceed` at the deploy prompt. It is a deterministic node (no LLM) that orchestrates the testbed Docker stack.

**Workflow:**

1. Copy `workspace/final_xapp.py` to `workspace/testbed/nearrtric/xapps/xapp.py`.
2. Copy `workspace/{data, ml, logic}/` into `workspace/testbed/nearrtric/xapps/` so the running xApp container can read its model and logic at runtime.
3. Patch `nearrtric/Dockerfile.xapp` so the entire `./xapps/` directory is copied into `/flexric/build/examples/xApp/python3/` (replaces the single-file `COPY` line if present).
4. `docker compose down` to stop any previous run.
5. `docker compose up -d --build` to rebuild and launch the nearrtric stack.
6. Sleep 20 seconds while the xApp executes against the testbed.
7. `docker logs --tail 50 xapp` to capture container output.
8. `docker compose down` to stop the stack.
9. Append the last 1 000 characters of logs to the conversation as a deployment summary.

Main function:

- `module_7_deployer_node(state)`: runs the steps above synchronously and returns `is_complete=True` plus an `AIMessage` summarizing the captured logs.

The testbed layout (`src/workspace/testbed/`) is expected to already be present and to contain a `nearrtric/` Docker Compose stack. Module 7 does not create it from scratch.

## Tooling

### Semantic Search service

Location: `src/tools/semantic_search`

The Semantic Search service replaces the previous FlexRIC Structural RAG. It is a Docker stack consisting of:

- `chroma-db` — ChromaDB vector store (host port `7000`, container port `8000`).
- `semantic-api` — FastAPI app (host port `7080`) that on startup clones every repo listed in `repos.yml` (currently the FlexRIC `dev` branch), chunks `.c/.h/.cpp/.hpp/.py` files with LangChain's `LanguageParser` + `RecursiveCharacterTextSplitter`, embeds them with `all-MiniLM-L6-v2`, and exposes:
  - `POST /semantic_search` — vector retrieval, optional `truncate_chars` and `return_full_text`.
  - `POST /exact_search` — `ripgrep`-backed keyword search across cloned repos.
  - `GET /status` — ingestion progress (cloning + embedding takes a few minutes on first start).

LangChain wrappers in `src/tools/semantic_search/semantic_search_tool.py`:

- `semantic_search_summary(query, n_results=5)` — primary tool used by Modules 2, 3, and 6. Returns full code snippets (`return_full_text=True`).
- `semantic_search_detailed(query, n_results=2)` — same endpoint but reserved for digging into a specific definition.

Both target `SEMANTIC_SEARCH_URL` (default `http://localhost:7080`).

### OrioSearch

Location: `src/tools/oriosearch`

Used by Module 2 through `restricted_domain_search`. Provides web search for O-RAN concepts, restricted to domains such as `o-ran-sc.org`. Runs on port 8000.

### Workspace tools

Location: `src/tools/workspace/workspace_tools.py`

The workspace tool layer gives ReAct modules controlled file and terminal access.

- `WORKSPACE_DIR` resolves to `src/workspace` and is exported for direct use by the Module 4 fallback reporter.
- File tools come from LangChain's `FileManagementToolkit` (read, write, list, copy, move, search).
- `terminal_command(command)` executes shell commands with `cwd=src/workspace`. The wrapper blocks simple attempts to `cd` outside the workspace and times out after 120 seconds.

Modules 3, 4, 5, and 6 depend on these tools to create generated artifacts.

### Context utilities

Location: `src/tools/context_utils.py`

- `limit_tool_messages(state)` — truncates already-consumed `ToolMessage` content to `[Output truncated to maintain context window.]` and caps active tool outputs at 10 000 characters.
- `limit_context_window(state, max_messages=14)` — sliding window that keeps the first message (system prompt or task) plus the most recent N − 1 messages. Used as a `pre_model_hook` by every ReAct agent.

### Convenience scripts

- `src/tools/start_tools.sh` — `cd` into each tool directory and run `docker compose up -d` (oriosearch and semantic_search).
- `src/tools/stop_tools.sh` — `docker compose stop` for both stacks.

## Generated Artifacts

All generated files are created under `src/workspace`:

```text
src/workspace
|-- data
|   |-- build_streaming_datasets.py
|   |-- pre_filter.py                  # only when profiling user paths
|   |-- streaming_mock_data.json       # hierarchical, matches Telemetry_Variables
|   |-- historical_training_data.csv   # ML workflows
|   `-- test_data.csv                  # ML workflows
|-- ml
|   |-- train.py                       # pre-written wrapper around auto_train
|   |-- saved_model.pkl
|   `-- evaluation_report.json         # always present, even on failure
|-- logic
|   `-- core_logic.py
|-- log
|   |-- module_3_data_engineer.log
|   |-- module_4_ml.log
|   |-- module_5_logic.log
|   `-- module_6_integrator.log
|-- testbed                            # populated by Module 7
|   `-- nearrtric/...
|-- flexric_template.py
`-- final_xapp.py
```

Not every run creates every artifact. `ml/` is skipped for `Pure_Logic` workflows. `data/pre_filter.py` only appears when at least one user-provided dataset path was profiled. `testbed/` is only touched if the user opted in to deployment.

## Environment Configuration

The graph is configured through `src/langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "env": ".env"
}
```

### Core variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | LLM inference endpoint |
| `OLLAMA_MODEL` | `llama3.1` | Model name |
| `SEMANTIC_SEARCH_URL` | `http://localhost:7080` | Semantic Search API (Modules 2, 3, 6) |
| `ORIOSEARCH_URL` | `http://localhost:8000` | Web search API (Module 2 fallback) |

### Per-module recursion limits

Each module has its own recursion limit env var. The global `RECURSIVE_LIMIT` acts as a floor — all per-module defaults are at least as large.

| Variable | Default | Module |
|---|---|---|
| `MAPPER_RECURSIVE_LIMIT` | 60 | Module 2 — technical mapper |
| `LOGIC_RECURSIVE_LIMIT` | 40 | Module 5 — core programmer |
| `INTEGRATOR_RECURSIVE_LIMIT` | 120 | Module 6 — xApp integrator |
| `ML_RECURSIVE_LIMIT` | 80 | Module 4 — ML developer |
| `RECURSIVE_LIMIT` | 40 | Global floor used by Module 3 and as fallback |

### Running the LangGraph Agent

From the `src` directory:

```bash
langgraph dev --no-reload
```

A typical interaction is:

1. User provides an xApp intent.
2. Module 1 asks targeted questions until the blueprint is complete.
3. User types `CONFIRM`.
4. Module 2 runs (semantic search calls against the FlexRIC index).
5. The agent shows the validated telemetry schema and asks about data availability; user replies with a path, `no`, or a mix.
6. Modules 3–6 run automatically, producing files under `src/workspace`.
7. The agent asks whether to deploy. User types `Proceed` to launch Module 7, or anything else to stop.

## Testing

`test/test_ml_contract_updates.py` contains contract tests covering:

- `TestMLContractUpdates`: Module 1 acceptance criteria in decomposer prompt; Module 3 test dataset contract; Module 4 evaluation loop contract.
- `TestModule4Reliability`: `ML_RECURSIVE_LIMIT` env var; fallback report writer; workspace artifact recovery; `WORKSPACE_DIR` import; per-attempt report writing; synthesizer row-count target.
- `TestUserDatasetFeature`: dataset interrupt wiring; profiler imports; system prompt content (FlexRIC-only constraint, pre-filter, column matching tiers, profiler notes); Module 3 init export.

Run with:

```bash
python3 test/test_ml_contract_updates.py -v
```

Because Module 1 tests call the configured LLM, they require a working Ollama setup:

```bash
python -m unittest discover -s test
```

## Design Principles

The codebase follows a few important separation-of-concerns rules:

- Module 1 captures intent only; it does not invent O-RAN technical details.
- Module 2 grounds technical mappings in Semantic Search results; it does not hallucinate C variables, and it produces a hierarchical schema rather than a flat list.
- Module 3 creates reproducible, mathematically consistent data — no random noise for primary signals — and validates every additional column against the FlexRIC codebase before letting it through.
- Module 4 runs an offline two-stage auto-training pipeline; the script is pre-written deterministically so the LLM cannot drift, and `evaluation_report.json` is always written so the result is visible to the operator.
- Module 5 writes independent decision logic with no FlexRIC dependencies and tests against the streaming JSON before integration.
- Module 6 handles FlexRIC integration glue only; one Semantic Search call resolves struct field names, and the workflow is hard-capped at 4 tool steps.
- Module 7 deploys deterministically without involving the LLM, capturing real container logs as proof of execution.

This division keeps the generated xApp easier to inspect, test, and debug. The final integration is only attempted after the intent, technical mappings, data, optional model, and core decision logic have been separately produced.

## Known Operational Assumptions

- Ollama is expected for the main LLM calls unless environment variables point elsewhere.
- Modules 2, 3, and 6 expect the Semantic Search server on `http://localhost:7080`. The first run takes several minutes while the FlexRIC repository is cloned and embedded into Chroma — `GET /status` reports progress.
- Module 2's restricted web search expects OrioSearch on `http://localhost:8000`.
- Generated scripts run inside `src/workspace`, so artifact paths in blueprints are relative to that directory.
- Module 6's syntax check does not prove runtime success against a live RIC; it only verifies Python syntax after template integration.
- `ml/evaluation_report.json` is always created by Module 4, even if the agent crashes, so the operator can always see whether the model met the acceptance threshold.
- Module 7 expects a pre-existing `src/workspace/testbed/nearrtric/` Docker Compose stack with a `Dockerfile.xapp`. It does not provision this from scratch.
