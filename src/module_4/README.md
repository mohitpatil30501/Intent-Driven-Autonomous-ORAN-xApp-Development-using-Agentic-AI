Module 4: ML Developer (The Data Scientist) [CONDITIONAL]
What it does: If the template says Supervised_ML or Unsupervised_ML, this module wakes up. It writes a Python script that trains candidate models using Module 3's historical training data, evaluates them with Module 3's test data, and saves only the best model.
Crucial Instruction for this Agent: The agent must compare candidate configurations until the blueprint threshold is met or MAX_TRAINING_ATTEMPTS is exhausted. It saves the best artifact to `ml/saved_model.pkl`, writes `ml/evaluation_report.json`, and returns metric details in `ML_Model_Artifacts`.
It should NOT write the real-time loop; its only job is offline model development and evaluation.
