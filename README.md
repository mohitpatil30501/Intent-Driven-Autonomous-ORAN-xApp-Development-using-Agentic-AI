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
Human review / confirmation
   |
   v
Module 2: Technical Mapper
   |  Adds FlexRIC service-model and variable mappings
   v
Module 3: Data Synthesizer
   |  Generates streaming mock data and optional training data
   v
Conditional branch
   |-- Supervised_ML / Unsupervised_ML --> Module 4: ML Developer
   |                                      Adds trained model artifacts
   v
Module 5: Core Programmer
   |  Creates and tests standalone XAppLogic
   v
Module 6: xApp Integrator
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
|   |-- structural_rag
|   |-- tools
|   `-- workspace
`-- test
    `-- module_1
```

The `src/workspace` directory is the controlled working area used by tool-enabled agents. Generated data, scripts, logs, logic code, ML artifacts, and final xApp files are expected to be created there.

## Runtime State

The graph state is declared in `src/agent.py` as:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    blueprint: Dict[str, Any]
    is_complete: bool
```

The fields have clear responsibilities:

- `messages`: the human/assistant conversation accumulated by LangGraph.
- `blueprint`: the structured contract passed between modules.
- `is_complete`: whether the intent blueprint has enough information to proceed.

The pipeline relies on JSON extraction from LLM responses. Most modules ask the LLM/ReAct agent to emit a strict JSON block, parse that block, then merge the parsed values back into `blueprint`.

## Orchestration: `src/agent.py`

`src/agent.py` is the conductor. It imports all six module node functions and wires them into a LangGraph workflow.

Key nodes:

- `intent_decomposer`: runs Module 1 and produces the initial intent blueprint.
- `ask_human`: a placeholder node used as an interrupt point for human-in-the-loop review.
- `technical_mapper`: runs Module 2 after the human confirms the blueprint.
- `data_synthesizer`: runs Module 3 and produces mock data files.
- `ml_dev`: conditionally runs Module 4 for ML-based xApps.
- `logic_dev`: runs Module 5 and writes the standalone decision engine.
- `integrator`: runs Module 6 and creates the final FlexRIC-integrated xApp.

Important routing functions:

- `should_continue(state)`: always routes from Module 1 to `ask_human`, forcing review before continuing.
- `check_confirmation(state)`: sends the flow to Module 2 only when the blueprint is complete and the latest human message contains `confirm`.
- `should_run_ml(state)`: checks `Intent_Blueprint.cycle_Type`; routes to Module 4 for `Supervised_ML` or `Unsupervised_ML`, otherwise skips directly to Module 5.

The graph is compiled with:

```python
graph = builder.compile(interrupt_before=["ask_human"])
```

This allows LangGraph Studio or a LangGraph client to pause before `ask_human` and let the user clarify or confirm the blueprint.

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
- model acceptance criteria for ML scenarios, defaulting to threshold `0.85` with metric policy `task_aware` when the user does not specify one.

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
      "metric_description_NL": "Task-aware model quality: supervised accuracy/F1, anomaly F1 for labeled anomaly tests, or reconstruction/separation quality for autoencoder-style workflows."
    }
  }
}
```

If fields are missing, the graph loops back to this module after the human answers the clarifying questions.

## Module 2: Technical Mapper

Location: `src/module_2/mapper.py`

Role: O-RAN engineer.

Module 2 translates the plain-English blueprint into concrete FlexRIC technical mappings. It uses a ReAct agent with search tools and is instructed to verify service models, telemetry variables, and action structures before producing a mapping.

Available tools:

- `semantic_code_search`: calls the semantic search service at `http://localhost:7080/semantic_search`.
- `exact_keyword_search`: calls `http://localhost:7080/exact_search`.
- `restricted_domain_search`: calls the OrioSearch service at `http://localhost:8000/search`, restricted by domain.

Main function:

- `module_2_technical_node(state)`: builds a search-grounded ReAct agent, passes it the current blueprint, and parses a `Technical_Mapping` JSON block.

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

## Module 3: Data Synthesizer

Location: `src/module_3/synthesizer.py`

Role: data engineer.

Module 3 creates synthetic data in the workspace based on the intent and technical mapping. It uses `workspace_tools`, which provide file operations and a restricted terminal rooted at `src/workspace`.

Main function:

- `module_3_data_node(state)`: creates a ReAct agent, instructs it to write a data-generation script, execute it, verify the first rows, and return data paths.

Generated artifacts:

- `src/workspace/data/generate_data.py`
- `src/workspace/data/streaming_mock_data.csv`
- `src/workspace/data/historical_training_data.csv` for ML workflows
- `src/workspace/data/test_data.csv` for ML evaluation workflows
- `src/workspace/log/module_3_data.log`

Behavior by cycle type:

- `Pure_Logic`: only streaming mock data is generated.
- `Supervised_ML`: streaming mock data, historical training data, and separate test data. Both training and test data should include a `label` column where `0` means normal/no-action behavior and `1` means anomaly or positive-action behavior.
- `Unsupervised_ML`: streaming mock data, historical training data, and separate test data. Historical training data may be unlabeled or normal-heavy, while test data should include labels when possible so Module 4 can compute anomaly F1.
- Autoencoder-style anomaly workflows: normal-only historical training data plus mixed normal/anomaly test data with labels.

Module 3 also classifies the generated ML data with one of these `training_data_profile` values:

- `supervised_labeled`
- `unsupervised_mixed`
- `autoencoder_normal_train_anomaly_test`
- `pure_logic`

Expected blueprint addition:

```json
{
  "Data_Paths": {
    "streaming_mock_data_path": "data/streaming_mock_data.csv",
    "historical_training_data_path": "data/historical_training_data.csv",
    "test_data_path": "data/test_data.csv",
    "test_label_column": "label",
    "training_data_profile": "supervised_labeled | unsupervised_mixed | autoencoder_normal_train_anomaly_test | pure_logic"
  }
}
```

For `Pure_Logic`, the ML-only paths and label column are returned as `null`, and `training_data_profile` is `pure_logic`. The module is required to verify CSV headers and first rows for every generated CSV so downstream modules can rely on exact column names matching `Technical_Mapping.Telemetry_Variables[*].C_variable`.

## Module 4: ML Developer

Location: `src/module_4/ml_dev.py`

Role: data scientist.

Module 4 runs only when `cycle_Type` is `Supervised_ML` or `Unsupervised_ML`. It trains candidate models from historical data, evaluates them on Module 3's separate test dataset, retries adjusted configurations until the acceptance threshold is met or attempts are exhausted, and saves only the best model artifact.

Main function:

- `module_4_ml_dev_node(state)`: backfills ML acceptance defaults when needed, explores train/test CSVs, chooses candidate algorithms, writes `ml/train.py`, runs it, and parses `ML_Model_Artifacts`.

Generated artifacts:

- `src/workspace/ml/train.py`
- `src/workspace/ml/saved_model.pkl`
- `src/workspace/ml/evaluation_report.json`
- `src/workspace/log/module_4_ml.log`

Evaluation behavior:

- Reads `Intent_Blueprint.model_acceptance_criteria.threshold`, defaulting to `0.85`.
- Reads `Intent_Blueprint.model_acceptance_criteria.metric_policy`, defaulting to `task_aware`.
- Reads `MAX_TRAINING_ATTEMPTS` from the environment, defaulting to `5`.
- For supervised labels, uses accuracy by default and F1 when class imbalance makes it more appropriate.
- For unsupervised anomaly detection with labeled test data, uses anomaly F1 as the primary metric.
- For autoencoder-style workflows, trains on normal data, scores reconstruction error on mixed test data, and evaluates anomaly F1 when labels exist.
- If labels are unavailable for an unsupervised test set, writes an advisory unsupervised quality metric instead of fabricating supervised accuracy.

Expected blueprint addition:

```json
{
  "ML_Model_Artifacts": {
    "model_path": "ml/saved_model.pkl",
    "technique_used": "IsolationForest for Unsupervised Anomaly Detection",
    "expected_input_features": ["dl_aggr_tbs", "buffer_occupancy"],
    "best_metric_name": "accuracy | f1 | anomaly_f1 | reconstruction_separation_score | other",
    "best_metric_value": 0.91,
    "threshold": 0.85,
    "threshold_met": true,
    "evaluation_report_path": "ml/evaluation_report.json"
  }
}
```

The module is intentionally scoped to offline ML development. It does not write FlexRIC networking code, subscribe to RIC callbacks, or implement real-time control loops. If the threshold is not met after the allowed attempts, it still preserves the best candidate and records the miss in `ml/evaluation_report.json`.

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
        ...

    def process_interval(self, row_dict):
        ...
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

For pure logic workflows, the class usually uses thresholds or optimization rules. For ML workflows, `__init__` loads the saved model from Module 4 and `process_interval` performs inference.

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

The prompt requires a test loop at the bottom of the generated script and checks that the logic does not return `DO_NOTHING` for every row when the mock data contains trigger conditions.

## Module 6: xApp Integrator

Location: `src/module_6/integrator.py`

Role: FlexRIC integrator.

Module 6 injects the tested `XAppLogic` class into a FlexRIC Python SDK template. It is responsible for glue code only: extracting telemetry from FlexRIC indication structs, passing a flat dictionary to `XAppLogic`, and translating returned decisions into FlexRIC control messages.

Main function:

- `module_6_integrator_node(state)`: reads the template, replaces placeholders, writes the final xApp, and runs `py_compile`.

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

## Workspace Tools

Location: `src/tools/workspace/workspace_tools.py`

The workspace tool layer gives ReAct modules controlled file and terminal access.

Key details:

- `WORKSPACE_DIR` resolves to `src/workspace`.
- File tools come from LangChain's `FileManagementToolkit`.
- Available file operations include read, write, list, copy, move, and search.
- `terminal_command(command)` executes shell commands with `cwd=src/workspace`.
- The terminal wrapper blocks simple attempts to `cd` outside the workspace.
- Commands time out after 120 seconds.

Modules 3, 4, 5, and 6 depend on these tools to create generated artifacts.

## Structural RAG Subsystem

Location: `src/structural_rag`

The Structural RAG subsystem is a separate code-indexing and retrieval pipeline for FlexRIC source code. It is designed to help agents ground xApp generation in real FlexRIC functions, structs, service models, and call graph context.

It has four build stages:

1. AST parsing: `core/parser.py`
2. Call graph building: `core/callgraph.py`
3. Natural-language summarization: `summarize/summarizer.py`
4. Hybrid indexing: `index/builder.py`

### `core/parser.py`

Parses C, C++, header, and Python files into chunks. It prefers tree-sitter when available and falls back to regex extraction for C.

Chunk types include:

- `function`
- `struct`
- `class`
- `file_summary`

Each chunk carries metadata such as:

- `chunk_id`
- `name`
- `signature`
- `body`
- `file`
- `layer`
- `sm_type`
- `is_xapp_example`
- `calls`
- `called_by`
- `nl_summary`

The parser skips high-noise folders such as build directories, test folders, generated wrappers, emulator code, and selected FlexRIC internals.

### `core/callgraph.py`

Builds a bidirectional call graph from parsed chunks. It extracts call names, resolves them to known chunk IDs, and annotates chunks with:

- `calls`
- `called_by`

It also identifies important boundary edges such as xApp-to-E2AP and service-model codec calls. When `networkx` is installed, it stores a `DiGraph`; otherwise it keeps an adjacency dictionary.

### `summarize/summarizer.py`

Adds one-sentence summaries to chunks so natural-language retrieval has stronger signal.

Supported summarizer backends:

- Ollama
- HuggingFace transformers
- llama.cpp
- GPT4All
- TF-IDF fallback
- no-op summarizer

The factory is exposed through `SummarizerFactory.create(...)`.

### `index/builder.py`

Builds a hybrid retrieval index under the output directory:

- `code_index.faiss`: CodeBERT or fallback code embeddings.
- `nl_index.faiss`: sentence-transformer embeddings of summaries.
- `bm25.pkl`: lexical BM25 index.
- `graph.pkl`: graph store.
- `meta.json`: chunk metadata.
- `config.json`: index configuration.

The hybrid index helps handle both intent-style queries such as "how do I subscribe to KPM measurements" and exact symbol queries such as `dl_aggr_tbs`.

### `retrieval/retriever.py`

Loads the saved index and retrieves relevant chunks using:

- natural-language FAISS search,
- code FAISS search,
- BM25 exact search,
- reciprocal rank fusion,
- service-model filtering,
- graph expansion over callers and callees,
- reranking boosts for xApp examples, matching SM type, and API-layer chunks.

### `server.py`

Wraps the retriever in a FastAPI service.

Endpoints:

- `GET /health`
- `POST /retrieve`
- `POST /context`
- `GET /schema`

The `/schema` endpoint returns OpenAI/Anthropic-compatible tool definitions, allowing external LLM agents to call the retrieval system.

### `llm_integration.py`

Provides examples for wiring the Structural RAG API into:

- direct HTTP calls,
- OpenAI tool use,
- Anthropic tool use,
- LangChain `StructuredTool` wrappers.

## Semantic Search Tooling

Location: `src/tools/semantic_search`

This is a Dockerized code search service used directly by Module 2. It clones repositories listed in `repos.yml`, chunks source files, embeds them into ChromaDB, and exposes both semantic and exact search endpoints.

Default configured repository:

```yaml
repositories:
  - name: "flexric"
    url: "https://gitlab.eurecom.fr/mosaic5g/flexric.git"
    branch: "dev"
```

Important endpoints:

- `GET /status`: reports ingestion status and logs.
- `POST /semantic_search`: semantic code search.
- `POST /exact_search`: exact keyword search using `ripgrep`.

Run it with:

```bash
cd src/tools/semantic_search
docker compose up -d --build
```

The first startup can take several minutes because the service clones and embeds the target codebases.

## OrioSearch Tooling

Location: `src/tools/oriosearch`

OrioSearch is used by Module 2 through `restricted_domain_search`. It provides web/document search for O-RAN concepts and documentation while allowing the prompt to restrict searches to domains such as `o-ran-sc.org`.

Run it with:

```bash
cd src/tools/oriosearch
docker compose up -d
```

## Generated Artifacts

Most generated files are created under `src/workspace`:

```text
src/workspace
|-- data
|   |-- generate_data.py
|   |-- streaming_mock_data.csv
|   |-- historical_training_data.csv
|   `-- test_data.csv
|-- ml
|   |-- train.py
|   |-- saved_model.pkl
|   `-- evaluation_report.json
|-- logic
|   `-- core_logic.py
|-- log
|   |-- module_3_data.log
|   |-- module_4_ml.log
|   |-- module_5_logic.log
|   `-- module_6_integrator.log
|-- flexric_template.py
`-- final_xapp.py
```

Not every run creates every artifact. For example, `ml/` is skipped for `Pure_Logic` workflows.

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

Useful environment variables:

- `OLLAMA_URL`: defaults to `http://localhost:11434`.
- `OLLAMA_MODEL`: defaults to `llama3.1`.
- `RECURSIVE_LIMIT`: defaults to `20` in modules that run ReAct agents.
- `MAX_TRAINING_ATTEMPTS`: defaults to `5` for Module 4 candidate training/evaluation retries.
- `FLEXRIC_INDEX_DIR`: used by the Structural RAG FastAPI server.
- `FLEXRIC_API_KEY`: optional auth token for the Structural RAG API.
- `FLEXRIC_MAX_TOP_K`: caps Structural RAG result count.

The existing `src/README.md` also documents LangGraph frontend/server variables such as `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_ASSISTANT_ID`, and `LANGSMITH_API_KEY`.

## Running the LangGraph Agent

From the `src` directory:

```bash
langgraph dev --no-reload
```

The graph entry point is:

```text
agent: ./agent.py:graph
```

A typical interaction is:

1. User provides an xApp intent.
2. Module 1 asks targeted questions until the blueprint is complete.
3. User types `CONFIRM`.
4. The graph runs Modules 2 through 6.
5. Generated artifacts appear in `src/workspace`.

## Testing

Current tests live under `test/module_1`.

`test/module_1/test_module_1.py` exercises the Module 1 intent-decomposition loop with several sample intents. It uses:

- the LangGraph `graph`,
- a simulated user prompt,
- an LLM-as-judge prompt,
- a score from 0 to 5 for blueprint quality.

Because the tests call the configured LLM, they require a working Ollama setup or compatible environment.

Run with:

```bash
python -m unittest discover -s test
```

## Design Principles

The codebase follows a few important separation-of-concerns rules:

- Module 1 captures intent only; it does not invent O-RAN technical details.
- Module 2 grounds technical mappings in search results.
- Module 3 creates reproducible data for testing.
- Module 4 trains offline models only.
- Module 5 writes independent decision logic with no FlexRIC dependencies.
- Module 6 handles FlexRIC integration glue only.

This division keeps the generated xApp easier to inspect, test, and debug. The final integration is only attempted after the intent, technical mappings, data, optional model, and core decision logic have been separately produced.

## Known Operational Assumptions

- Ollama is expected for the main LLM calls unless environment variables point elsewhere.
- Module 2 expects the semantic search service on `http://localhost:7080`.
- Module 2's restricted web search expects OrioSearch on `http://localhost:8000`.
- Generated scripts run inside `src/workspace`, so artifact paths in blueprints are relative to that directory.
- The final xApp syntax check does not prove runtime success against a live RIC; it only verifies Python syntax after template integration.
