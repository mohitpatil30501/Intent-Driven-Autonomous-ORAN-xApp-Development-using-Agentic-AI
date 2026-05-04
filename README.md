# Intent-Driven Autonomous O-RAN xApp Development using Agentic AI

This repository implements a LangGraph-based agent pipeline that converts a high-level natural-language network intent into a deployable FlexRIC-style xApp scaffold. The system is organized as six sequential modules, each acting as a specialized agent role: intent analyst, O-RAN mapper, data engineer, ML developer, core logic programmer, and FlexRIC integrator.

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
   |  Queries Semantic Search (port 7080) — 2 calls max
   |  Adds FlexRIC service-model and variable mappings
   v
Dataset question  [INTERRUPT]
   |  "Do you have an existing dataset, or should one be synthesized?"
   v
Conditional branch
   |-- User provides a path --> Module 3b: Dataset Profiler
   |                            Discovers, filters (FlexRIC-only columns),
   |                            maps, and ingests the user's dataset
   |-- "no"               --> Module 3: Data Synthesizer
   |                            Generates 5 000-row training data and
   |                            streaming mock data using numpy vectorized calls
   v
Conditional branch
   |-- Supervised_ML / Unsupervised_ML --> Module 4: ML Developer
   |                                      Trains model, writes evaluation_report.json
   |                                      after every attempt, recovers artifacts
   |                                      from workspace if agent is interrupted
   v
Module 5: Core Programmer
   |  Creates and tests standalone XAppLogic
   v
Module 6: xApp Integrator
   |  Queries Semantic Search (1 call) for struct field names
   |  Injects logic into FlexRIC template
   v
Final xApp artifact
```

The main graph is defined in `src/agent.py`. It uses LangGraph's `StateGraph` and a small set of conditional routing functions to decide when to pause for the human, when to proceed to technical mapping, and whether ML training is required.

## Repository Layout

```text
.
|-- requirements.txt
|-- src
|   |-- agent.py
|   |-- langgraph.json
|   |-- module_1
|   |-- module_2
|   |-- module_3
|   |-- module_4
|   |-- module_5
|   |-- module_6
|   |-- tools
|   |   |-- semantic_search     # Semantic Search Docker + Tool API wrappers
|   |   |-- workspace           # File + terminal tools for ReAct agents
|   |   `-- oriosearch          # O-RAN web search (Docker)
|   `-- workspace               # All generated artifacts land here
`-- test
```

The `src/workspace` directory is the controlled working area used by tool-enabled agents. Generated data, scripts, logs, logic code, ML artifacts, and final xApp files are created there.

## Runtime State

The graph state is declared in `src/agent.py` as:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    blueprint: Dict[str, Any]
    is_complete: bool
    user_dataset_path: Optional[str]
```

| Field | Purpose |
|---|---|
| `messages` | Human/assistant conversation accumulated by LangGraph |
| `blueprint` | Structured contract passed between modules |
| `is_complete` | Whether the intent blueprint has enough information to proceed |
| `user_dataset_path` | Set by `receive_dataset` after the dataset interrupt. `None` = auto-synthesize; non-empty string = absolute path to the user's dataset |

The pipeline relies on JSON extraction from LLM responses. Most modules ask the LLM/ReAct agent to emit a strict JSON block, parse that block, then merge the parsed values back into `blueprint`.

## Orchestration: `src/agent.py`

`src/agent.py` is the conductor. It imports all six module node functions and wires them into a LangGraph workflow.

Key nodes:

- `intent_decomposer`: runs Module 1 and produces the initial intent blueprint.
- `ask_human`: a placeholder node used as an interrupt point for human-in-the-loop review.
- `technical_mapper`: runs Module 2 after the human confirms the blueprint.
- `ask_dataset`: appends an AIMessage listing the required FlexRIC telemetry columns and asking whether the user has an existing dataset.
- `receive_dataset`: reads the user's reply and writes it to `user_dataset_path` in state (`None` if "no").
- `data_synthesizer`: runs Module 3 and generates synthetic CSV data files.
- `dataset_profiler`: runs Module 3b when the user supplies a dataset path; discovers files, filters to FlexRIC-reportable columns only, maps them to the required telemetry variables, and outputs the same `Data_Paths` structure as `data_synthesizer`.
- `ml_dev`: conditionally runs Module 4 for ML-based xApps.
- `logic_dev`: runs Module 5 and writes the standalone decision engine.
- `integrator`: runs Module 6 and creates the final FlexRIC-integrated xApp.

Important routing functions:

- `should_continue(state)`: always routes from Module 1 to `ask_human`, forcing review before continuing.
- `check_confirmation(state)`: sends the flow to Module 2 only when the blueprint is complete and the latest human message contains `confirm`.
- `check_dataset_input(state)`: routes to `dataset_profiler` when `user_dataset_path` is set, otherwise to `data_synthesizer`.
- `should_run_ml(state)`: checks `Intent_Blueprint.cycle_Type`; routes to Module 4 for `Supervised_ML` or `Unsupervised_ML`, otherwise skips directly to Module 5. Both `data_synthesizer` and `dataset_profiler` feed into this gate.

The graph is compiled with:

```python
graph = builder.compile(interrupt_before=["ask_human", "receive_dataset"])
```

There are two human-in-the-loop interrupt points: the first pauses for blueprint confirmation, the second pauses for the dataset question after Module 2 completes.

## Module 1: Intent Decomposer

Location: `src/module_1/decomposer.py`

Role: business analyst.

Module 1 interviews the user and converts a high-level requirement into an `Intent_Blueprint`. It is intentionally not allowed to invent FlexRIC implementation details. Its prompt explicitly says not to hallucinate O-RAN service models, C structs, or API variables.

It must capture:

- metrics to monitor,
- action to take,
- objective or reason,
- cycle type: `Pure_Logic`, `Supervised_ML`, or `Unsupervised_ML`,
- data behavior, especially for ML scenarios,
- model acceptance criteria for ML scenarios, defaulting to threshold `0.85` with metric policy `task_aware` when the user does not specify one. For `Pure_Logic` workflows all three criteria fields (`threshold`, `metric_policy`, `metric_description_NL`) are set to `null` so downstream modules are not confused into loading a non-existent model.

Main functions:

- `get_llm()`: creates a `ChatOllama` client using `OLLAMA_URL` and `OLLAMA_MODEL`.
- `extract_json(text)`: extracts the JSON code block from the model response.
- `decomposer_node(state, config)`: invokes the LLM, parses the blueprint, and sets `is_complete`.

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

Module 2 translates the plain-English blueprint into concrete FlexRIC technical mappings using **Semantic Search**.

**Search strategy — 2 calls maximum:**

1. `semantic_search_summary(query="<SM> SM indication struct telemetry variables")` — finds the exact C variable names for the requested metrics.
2. `semantic_search_summary(query="<SM> SM control message struct action")` — finds the control struct fields and action types.

Available tools:

- `semantic_search_summary`: Primary search tool that returns broad context and code snippets.
- `semantic_search_detailed`: Only used when full function bodies are needed.
- `restricted_domain_search`: web search via OrioSearch (port 8000), restricted to domains such as `o-ran-sc.org`. Used only as a fallback.

Main function:

- `module_2_technical_node(state)`: builds a Structural-RAG-grounded ReAct agent (2-call strategy), passes it the current blueprint, and parses a `Technical_Mapping` JSON block.

Expected blueprint addition:

```json
{
  "Technical_Mapping": {
    "Reporting_Service_Model": "MAC | KPM | RLC | RC",
    "Telemetry_Variables": [
      {
        "NL_name": "per slice throughput",
        "C_variable": "dl_aggr_tbs",
        "data_type": "uint32_t"
      }
    ],
    "Control_Service_Model": "MAC | RC",
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

This module is the safety barrier against invented FlexRIC symbols. The prompt requires search before mapping and tells the agent to mark unsupported capabilities rather than fabricate variables.

## Module 3: Data Preparation (Synthesizer or Profiler)

After Module 2 completes, the graph pauses and asks:

> "Do you have an existing dataset? Type 'no' to auto-generate synthetic data, or provide an absolute path."

The answer routes the graph to one of two data-preparation nodes. Both produce the identical `Data_Paths` blueprint addition so all downstream modules are unaffected by which path was taken.

### Module 3a: Data Synthesizer

Location: `src/module_3/synthesizer.py`

Role: data engineer (synthetic path).

Activated when the user types `no` at the dataset prompt.

Main function:

- `module_3_data_node(state)`: creates a ReAct agent, instructs it to write and execute a data-generation script using numpy vectorized calls, verify the first rows, and return data paths.

Row-count targets:

| File | Target |
|---|---|
| `streaming_mock_data.csv` | 100–500 rows |
| `historical_training_data.csv` | **Exactly 5 000 rows** (numpy vectorized, no Python loops) |
| `test_data.csv` | **Exactly 1 000 rows** |

Generated artifacts:

- `src/workspace/data/generate_data.py`
- `src/workspace/data/streaming_mock_data.csv`
- `src/workspace/data/historical_training_data.csv` for ML workflows
- `src/workspace/data/test_data.csv` for ML evaluation workflows
- `src/workspace/log/module_3_data.log`

Behavior by cycle type:

- `Pure_Logic`: only streaming mock data is generated.
- `Supervised_ML`: all three files. Training and test data both include a `label` column balanced ~50/50.
- `Unsupervised_ML`: all three files. Historical training data may be unlabeled or normal-heavy; test data includes labels when possible so Module 4 can compute anomaly F1.
- Autoencoder-style anomaly workflows: normal-only historical training data plus mixed normal/anomaly test data with labels.

### Module 3b: Dataset Profiler

Location: `src/module_3/profiler.py`

Role: data engineer (user-provided dataset path).

Activated when the user provides a file or directory path at the dataset prompt.

**Design principle — RAN-reportable columns only:** An xApp deployed on a real RAN (srsRAN via FlexRIC service models) can only receive metrics that FlexRIC can actually report. The profiler enforces this by:

1. Accepting only columns from `Technical_Mapping.Telemetry_Variables[*].C_variable` as required (these are already FlexRIC-validated by Module 2).
2. Validating any additional columns for ML enrichment against the FlexRIC codebase using `semantic_search_summary`. Columns with no FlexRIC match are excluded.

**Handling large and multi-file datasets:** The profiler uses a two-phase approach:

- **Phase 1 — Python pre-filter (no LLM):** A generated script loads only column headers (zero-row read), detects dtypes from a 500-row sample, and drops non-numeric, admin/infrastructure, and zero-variance columns. This reduces a 280-column dataset to a manageable candidate list before any LLM reasoning begins.
- **Phase 2 — LLM reasoning on the reduced set:** The agent matches the candidate columns to required telemetry variables (exact → case-insensitive → normalized → semantic) and validates additional columns with `semantic_search_summary`.

Main function:

- `dataset_profiler_node(state)`: runs a 9-step ReAct agent workflow: discover files → load headers → pre-filter → match required columns → FlexRIC-validate additional columns → determine split strategy → build merged dataframe → cross-validate → set `training_data_profile`.

Generated artifacts:

- `src/workspace/data/pre_filter.py`
- `src/workspace/data/profile_and_merge.py`
- `src/workspace/data/streaming_mock_data.csv`
- `src/workspace/data/historical_training_data.csv` for ML workflows
- `src/workspace/data/test_data.csv` for ML evaluation workflows
- `src/workspace/log/module_3_profiler.log`

### Shared Blueprint Addition

Both paths produce the same `Data_Paths` structure:

```json
{
  "Data_Paths": {
    "streaming_mock_data_path": "data/streaming_mock_data.csv",
    "historical_training_data_path": "data/historical_training_data.csv",
    "test_data_path": "data/test_data.csv",
    "test_label_column": "label",
    "training_data_profile": "supervised_labeled | supervised_synthesized_labels | unsupervised_mixed | unsupervised_unlabeled_test | autoencoder_normal_train_anomaly_test | pure_logic",
    "profiler_notes": "string (profiler path only)"
  }
}
```

For `Pure_Logic`, the ML-only paths and label column are `null` and `training_data_profile` is `pure_logic`.

The two additional `training_data_profile` values produced by the profiler:

- `supervised_synthesized_labels`: Supervised ML where the source dataset lacked a label column and labels were synthesized.
- `unsupervised_unlabeled_test`: Unsupervised ML where the source dataset contained no label column for the test split.

## Module 4: ML Developer

Location: `src/module_4/ml_dev.py`

Role: data scientist.

Module 4 runs only when `cycle_Type` is `Supervised_ML` or `Unsupervised_ML`. It trains candidate models from historical data, evaluates them on the separate test dataset, and saves only the best model artifact.

**Reliability design:** `ml/train.py` is instructed to overwrite `ml/evaluation_report.json` after every training attempt, not just at the end. If the agent exhausts its recursion limit, the node reads the report directly from the workspace filesystem. If no report exists at all (agent crashed before writing), a fallback report is written with `threshold_met: false` and an explanation so the operator always gets a result.

Main function:

- `module_4_ml_dev_node(state)`: backfills ML acceptance defaults, explores train/test CSVs in a single combined one-liner, writes `ml/train.py` (which writes the evaluation report after each attempt), runs it, and parses `ML_Model_Artifacts`.
- `_write_fallback_report(blueprint, error_msg)`: writes a minimal `ml/evaluation_report.json` if the agent crashes or hits the recursion limit before creating one.
- `_recover_artifacts_from_workspace(blueprint)`: reads `ml/evaluation_report.json` from disk after the agent finishes, regardless of whether the agent emitted valid JSON in its final message.

Generated artifacts:

- `src/workspace/ml/train.py`
- `src/workspace/ml/saved_model.pkl`
- `src/workspace/ml/evaluation_report.json`
- `src/workspace/log/module_4_ml.log`

Evaluation behavior:

- Reads `Intent_Blueprint.model_acceptance_criteria.threshold`, defaulting to `0.85`.
- Reads `Intent_Blueprint.model_acceptance_criteria.metric_policy`, defaulting to `task_aware`.
- Reads `MAX_TRAINING_ATTEMPTS` from the environment, defaulting to `5`. The generated `train.py` also reads this env var.
- For supervised labels, uses accuracy by default and F1 when class imbalance makes it more appropriate.
- For unsupervised anomaly detection with labeled test data, uses anomaly F1 as the primary metric.
- For autoencoder-style workflows, trains on normal data, scores reconstruction error on mixed test data, and evaluates anomaly F1 when labels exist.

Expected blueprint addition:

```json
{
  "ML_Model_Artifacts": {
    "model_path": "ml/saved_model.pkl",
    "technique_used": "RandomForestClassifier for Supervised_ML",
    "expected_input_features": ["cqi", "buffer_occupancy"],
    "best_metric_name": "accuracy | f1 | anomaly_f1 | other",
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

- `module_5_logic_dev_node(state)`: writes `logic/core_logic.py`, tests it against streaming mock data, and returns the logic artifact metadata.

Generated artifacts:

- `src/workspace/logic/core_logic.py`
- `src/workspace/log/module_5_logic.log`

Required class contract:

```python
class XAppLogic:
    def __init__(self):
        ...   # loads ML model if ML_Model_Artifacts present; empty for Pure_Logic

    def process_interval(self, row_dict):
        ...   # returns a dict matching one Action_Space_Menu entry
```

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

Module 6 injects the tested `XAppLogic` class into a FlexRIC Python SDK template. It uses semantic search with a single targeted call to resolve how to access the correct C-struct fields for the detected service model (e.g. `ind.ue_stats[0].wb_cqi` vs `ind.mac_stats[0].dl_aggr_tbs`).

**Context-budget design:** Only the fields Module 6 actually needs are extracted from the full blueprint before being passed to the agent. This significantly reduces the initial prompt size compared to dumping the entire blueprint (which includes ML artifacts, data paths, synthesizer metadata, and Module 5 terminal logs that are irrelevant at this stage).

Fields passed to the agent:

| Blueprint field | Why included |
|---|---|
| `Intent_Blueprint.xApp_Name`, `.goal`, `.cycle_Type` | Orientation context |
| `Technical_Mapping` | SM type, telemetry C-variables, action space |
| `Logic_Artifacts` | Path to the Module 5 script |

Fields excluded: `Data_Paths`, `ML_Model_Artifacts`, all synthesizer/profiler metadata, Module 3–5 log content.

**Message-window trimming:** A `pre_model_hook` runs before each model call inside the ReAct loop. If the internal message list exceeds 12 entries, it keeps only the first message (the task + context) and the 10 most recent messages, dropping the middle. This caps the effective context sent to the model at roughly 10–15 K tokens regardless of how many tool-call iterations have accumulated, preventing context-window overflow on OSS models with 128 K limits.

**Strict 4-step workflow:** The system prompt enforces exactly 4 tool calls in sequence, eliminating the open-ended loop that caused the agent to burn through its recursion budget:

1. `semantic_search_summary` — one search call for struct field patterns
2. `read_file flexric_template.py` — read the template once
3. `write_file final_xapp.py` — write the completed xApp once
4. `terminal_command "mkdir -p log && python3 -m py_compile final_xapp.py 2>&1 | tee log/module_6_integrator.log"` — syntax check and log in one combined command

The recursion limit default was reduced from 40 to 20, which is still ample for the 4-step workflow and prevents runaway looping.

**Step-limit error handling:** If the recursion limit is reached, the node detects the LangGraph `"need more steps"` message and returns a clear, actionable error (`"Set INTEGRATOR_RECURSIVE_LIMIT > N and re-run"`) instead of silently propagating the opaque message to the UI.

Main functions:

- `module_6_integrator_node(state)`: extracts the integration context, calls the ReAct agent with message trimming, and parses the `Final_Deployment` JSON.
- `_extract_integration_context(blueprint)`: returns only the three blueprint sections Module 6 needs.
- `_pre_model_hook(state)`: trims the agent's internal message list before each model call.

Template locations:

- `src/workspace/flexric_template.py`
- `src/module_6/flexric_template.py`

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

The module does not execute the final xApp because a live RIC/FlexRIC environment may not be available. It only performs a syntax check with:

```bash
python3 -m py_compile final_xapp.py
```

## Semantic Search

Location: `src/tools/semantic_search/`
Tool wrappers: `src/tools/semantic_search/semantic_search_tool.py`

The Semantic Search tool provides codebase exploration capabilities for the agent.

Start command:

```bash
cd src/tools/semantic_search
docker compose up -d --build
```

## Workspace Tools

Location: `src/tools/workspace/workspace_tools.py`

The workspace tool layer gives ReAct modules controlled file and terminal access.

Key details:

- `WORKSPACE_DIR` resolves to `src/workspace` and is exported for direct use by the Module 4 fallback reporter.
- File tools come from LangChain's `FileManagementToolkit`.
- Available file operations include read, write, list, copy, move, and search.
- `terminal_command(command)` executes shell commands with `cwd=src/workspace`.
- The terminal wrapper blocks simple attempts to `cd` outside the workspace.
- Commands time out after 120 seconds.

Modules 3, 4, 5, and 6 depend on these tools to create generated artifacts.

## OrioSearch Tooling

Location: `src/tools/oriosearch`

Used by Module 2 through `restricted_domain_search`. Provides web/document search for O-RAN concepts and documentation, restricting searches to domains such as `o-ran-sc.org`.

Run it with:

```bash
cd src/tools/oriosearch
docker compose up -d
```

## Generated Artifacts

All generated files are created under `src/workspace`:

```text
src/workspace
|-- data
|   |-- generate_data.py          # synthesizer path only
|   |-- pre_filter.py             # profiler path only
|   |-- profile_and_merge.py      # profiler path only
|   |-- streaming_mock_data.csv
|   |-- historical_training_data.csv
|   `-- test_data.csv
|-- ml
|   |-- train.py
|   |-- saved_model.pkl
|   `-- evaluation_report.json    # written after every training attempt; always present
|-- logic
|   `-- core_logic.py
|-- log
|   |-- module_3_data.log         # synthesizer path only
|   |-- module_3_profiler.log     # profiler path only
|   |-- module_4_ml.log
|   |-- module_5_logic.log
|   `-- module_6_integrator.log
|-- flexric_template.py
`-- final_xapp.py
```

Not every run creates every artifact. `ml/` is skipped for `Pure_Logic` workflows. The `data/` scripts differ depending on whether the synthesizer or profiler path was taken.

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
| `OLLAMA_MODEL` | `GPT-OSS-120B` | Model name |
| `SEMANTIC_SEARCH_URL` | `http://localhost:7080` | Semantic Search API (Modules 2, 6, 3b) |
| `ORIOSEARCH_URL` | `http://localhost:8000` | Web search API (Module 2 fallback) |

### Per-module recursion limits

Each module has its own recursion limit env var. The global `RECURSIVE_LIMIT` acts as a floor — all per-module defaults are at least as large.

| Variable | Default | Module |
|---|---|---|
| `MAPPER_RECURSIVE_LIMIT` | 40 | Module 2 — technical mapper |
| `LOGIC_RECURSIVE_LIMIT` | 40 | Module 5 — core programmer |
| `INTEGRATOR_RECURSIVE_LIMIT` | **20** | Module 6 — xApp integrator (4-step workflow; raise only if the model needs extra repair steps) |
| `ML_RECURSIVE_LIMIT` | **80** | Module 4 — ML developer (more steps: explore → write → run → debug → verify) |
| `RECURSIVE_LIMIT` | 20 | Global floor used by Modules 3, 3b, and as fallback |

### ML training variables

| Variable | Default | Description |
|---|---|---|
| `MAX_TRAINING_ATTEMPTS` | `5` | Module 4 candidate training/evaluation retries |



## Running the LangGraph Agent

From the `src` directory:

```bash
langgraph dev --no-reload
```

A typical interaction is:

1. User provides an xApp intent.
2. Module 1 asks targeted questions until the blueprint is complete.
3. User types `CONFIRM`.
4. Module 2 runs (2 semantic search calls).
5. The agent asks about an existing dataset; user types `no` or provides a path.
6. Modules 3–6 run automatically, producing files under `src/workspace`.

## Testing

`test/test_ml_contract_updates.py` contains contract tests covering:

- `TestMLContractUpdates`: Module 1 acceptance criteria in decomposer prompt; Module 3 test dataset contract; Module 4 evaluation loop contract.
- `TestModule4Reliability`: `ML_RECURSIVE_LIMIT` env var (default 80); fallback report writer; workspace artifact recovery; `WORKSPACE_DIR` import; per-attempt report writing; 5 000-row synthesizer target.
- `TestUserDatasetFeature`: `user_dataset_path` in `AgentState`; new graph nodes and interrupt; profiler imports; system prompt content (FlexRIC-only constraint, pre-filter, column matching tiers, profiler notes); `module_3/__init__.py` export.

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
- Module 2 grounds technical mappings in semantic search results; it does not hallucinate C variables.
- Module 3 creates reproducible data for testing, using numpy vectorized generation.
- Module 4 trains offline models only; it writes `evaluation_report.json` after every attempt so the result is always visible to the operator.
- Module 5 writes independent decision logic with no FlexRIC dependencies.
- Module 6 handles FlexRIC integration glue only; one semantic search call resolves struct field names.

This division keeps the generated xApp easier to inspect, test, and debug. The final integration is only attempted after the intent, technical mappings, data, optional model, and core decision logic have been separately produced.

## Known Operational Assumptions

- Ollama is expected for the main LLM calls unless environment variables point elsewhere.
- Module 2 and Module 6 expect the Semantic Search server on `http://localhost:7080` (set `SEMANTIC_SEARCH_URL` to override).
- Module 2's restricted web search expects OrioSearch on `http://localhost:8000`.
- Generated scripts run inside `src/workspace`, so artifact paths in blueprints are relative to that directory.
- The final xApp syntax check does not prove runtime success against a live RIC; it only verifies Python syntax after template integration.
- `ml/evaluation_report.json` is always created by Module 4, even if the agent crashes, so the operator can always see whether the model met the acceptance threshold.
