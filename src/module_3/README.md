Module 3: Data Preparation (Synthesizer or Profiler)

After Module 2 completes the technical mapping, the graph pauses and asks whether you have an
existing dataset. Your answer routes to one of two nodes. Both produce the identical Data_Paths
blueprint structure so all downstream modules (4, 5, 6) are unaffected by which path ran.

---

Module 3a: Data Synthesizer — src/module_3/synthesizer.py

Activated when the user types "no" at the dataset prompt.

Generates synthetic data in the workspace based on the blueprint and technical mapping.
Uses workspace_tools for file operations and a restricted terminal rooted at src/workspace.

What it generates (by cycle type):

  Pure_Logic:
    - data/streaming_mock_data.csv (100–500 rows)
    - training_data_profile: "pure_logic"

  Supervised_ML:
    - data/streaming_mock_data.csv
    - data/historical_training_data.csv (5 000+ rows, includes "label" column)
    - data/test_data.csv (1 000+ rows, includes "label" column)
    - training_data_profile: "supervised_labeled"

  Unsupervised_ML:
    - data/streaming_mock_data.csv
    - data/historical_training_data.csv (normal-heavy or unlabeled)
    - data/test_data.csv (includes "label" column when possible for anomaly F1)
    - training_data_profile: "unsupervised_mixed"

  Autoencoder-style:
    - data/historical_training_data.csv (normal-only)
    - data/test_data.csv (mixed normal/anomaly with labels)
    - training_data_profile: "autoencoder_normal_train_anomaly_test"

Generated workspace files:
  data/generate_data.py
  data/streaming_mock_data.csv
  data/historical_training_data.csv  (ML only)
  data/test_data.csv                 (ML only)
  log/module_3_data.log

---

Module 3b: Dataset Profiler — src/module_3/profiler.py

Activated when the user provides an absolute path to their own dataset at the dataset prompt.
Supports single files, multi-file directories, and nested folder structures.

Design principle — RAN-reportable columns only:
  An xApp deployed on a real RAN (srsRAN via FlexRIC) can only receive metrics that FlexRIC can
  actually report. Training on non-FlexRIC columns will break the deployed xApp at inference time.
  The profiler enforces this:
    Required columns: Technical_Mapping.Telemetry_Variables[*].C_variable (FlexRIC-validated by Module 2)
    Additional columns (ML only): verified against the indexed FlexRIC codebase via exact_keyword_search
    Admin/infrastructure columns (IPs, MACs, timestamps, IDs): always excluded

Handling large datasets (100+ columns):
  Phase 1 — Python pre-filter (no LLM): loads only headers (nrows=0), checks dtypes on a 500-row
  sample, and drops non-numeric, admin, and zero-variance columns. A 280-column dataset typically
  reduces to 30–60 candidates before any LLM reasoning starts.
  Phase 2 — LLM reasoning on the reduced set: matches candidates to required columns (exact →
  case-insensitive → normalized → semantic) and runs exact_keyword_search for additional columns.

9-step workflow:
  1. Discover all .csv/.tsv/.parquet/.xlsx files at the user path
  2. Load headers only (zero-row read)
  3. Python pre-filter script — reduce to numeric, valid-identifier candidates
  4. Match required C_variable columns from Technical Mapping
  5. FlexRIC-validate additional candidate columns (ML only)
  6. Count rows and determine train/stream/test split ratios
  7. Build merged dataframe — rename matched columns, synthesize missing ones
  8. Cross-validate every output CSV
  9. Set training_data_profile

Split strategy (Supervised_ML / Unsupervised_ML):
  < 1 000 rows:   70% training / 20% test / 10% streaming (cap at 500)
  1K – 9 999:     70% training / 20% test / 10% streaming (cap at 500)
  ≥ 10 000:       80% training / 15% test / 5% streaming  (cap at 500)

training_data_profile values (profiler path adds two new values):
  "supervised_labeled"              — Supervised ML, label found in source
  "supervised_synthesized_labels"   — Supervised ML, label synthesized (not in source)
  "unsupervised_mixed"              — Unsupervised ML, test split has labels
  "unsupervised_unlabeled_test"     — Unsupervised ML, no label column found
  "pure_logic"                      — Pure Logic, only streaming data produced

Generated workspace files:
  data/pre_filter.py
  data/profile_and_merge.py
  data/streaming_mock_data.csv
  data/historical_training_data.csv  (ML only)
  data/test_data.csv                 (ML only)
  log/module_3_profiler.log

---

Shared blueprint output (Data_Paths):

  {
    "Data_Paths": {
      "streaming_mock_data_path": "data/streaming_mock_data.csv",
      "historical_training_data_path": "data/historical_training_data.csv",  // null for Pure_Logic
      "test_data_path": "data/test_data.csv",                                // null for Pure_Logic
      "test_label_column": "label",                                          // null for Pure_Logic
      "training_data_profile": "<see values above>",
      "profiler_notes": "<profiler path only>"
    }
  }
