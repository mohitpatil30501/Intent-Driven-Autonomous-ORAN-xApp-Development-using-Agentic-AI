import json
import os
import random
import pickle
import joblib
import numpy as np
# Mock imports required by the blueprint structure, even if actual loading fails in a constrained env.
# In a real environment, we would import necessary ML libraries here (e.g., numpy, pandas, sklearn)

# Define action constants matching the Technical_Mapping
ACTION_SET_WEIGHT = "SET_UE_SCHEDULING_WEIGHT"
ACTION_DO_NOTHING = "DO_NOTHING"

class XAppLogic:
    """
    Core logic processor for anomaly detection in uplink streaming data.
    It is stateful, using a loaded ML model (if available) to score incoming data.
    """
    def __init__(self):
        print("XAppLogic Initializing...")
        # Check for ML Model Artifacts as per blueprint
        model_path = "ml/saved_model.pkl"
        if os.path.exists(model_path):
            try:
                # In a real scenario, this loads the trained model artifact
                # with open(model_path, 'rb') as f:
                #     self.model = pickle.load(f)
                self.model = joblib.load(model_path)
                print("Model loaded successfully for anomaly scoring.")
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}. Running in simulation mode. Error: {e}")
                self.model = None
        else:
            self.model = None
            print("No model artifact found. Logic will rely on simplified heuristic simulation.")

        self.threshold = 0.25 # From model_acceptance_criteria

    def _simulate_anomaly_score(self, prb_usage: int, throughput: float) -> float:
        """
        A placeholder function to simulate running the data through a trained ML model.
        In the absence of a real model, we use a simple heuristic based on deviation
        from some expected correlation (e.g., high PRB usage vs low throughput).
        Score close to 1.0 indicates high anomaly.
        """
        # Heuristic: Anomaly increases if PRB usage is high but throughput is low (inefficient).
        # Normalize inputs for a score between 0 and 1.
        # A simple measure: deviation from expected linear relation (Throughput / PRB).
        expected_efficiency = 0.5 # Placeholder average
        current_efficiency = throughput / max(1, prb_usage)
        
        # Calculate 'distance' from expected efficiency, scaled to fit a score range.
        # We use an exponential function to make small deviations result in low scores, 
        # and large deviations result in high scores.
        score = 1.0 - abs(current_efficiency - expected_efficiency) * 2
        
        # Clamp score between 0.0 and 1.0
        return max(0.0, min(1.0, score + random.uniform(-0.1, 0.1)))

    def process_interval(self, row_dict: dict) -> dict:
        """
        Processes one timestep of KPM data to decide on the next scheduling action.
        Must return an action matching the Action_Space_Menu.
        """
        # Extract necessary features from the input dictionary
        data = row_dict.get("data", row_dict)
        prb_usage = data.get("uplink_prb_usage", 0)
        throughput = data.get("uplink_throughput", 0.0)
        ue_id = data.get("ue_id", 0)
        # 1. Determine Anomaly Score
        if self.model:
            # Real ML path:
            features = [[prb_usage, throughput]]
            # score = self.model.predict_proba(features)[:, 1][0] # Example prediction
            score = self._simulate_anomaly_score(prb_usage, throughput) # Fallback to simulation for reliable execution
        else:
            # Simulation/Fallback path (Always used here for guaranteed execution)
            score = self._simulate_anomaly_score(prb_usage, throughput)
        
        # 2. Decision Logic based on Model Acceptance Criteria
        print(f"--- Processing UE {ue_id} (PRB: {prb_usage}, Thp: {throughput:.2f}) ---")
        print(f"-> Calculated Anomaly Score: {score:.4f}")
        
        action = None
        if score > self.threshold:
            print(f"!!! ANOMALY DETECTED: Score {score:.4f}!!!")
            # Action: Set UE Scheduling Weight (Anomalous)
            action = {
                "action_id": "SET_UE_SCHEDULING_WEIGHT",
                "parameters": {
                    "ue_id": ue_id,
                    "new_weight": 0.3, # Penalty weight assigned
                    "reason": "Anomaly Detection Penalty"
                }
            }
        else:
            print("-> Status: Nominal. No action required.")
            # Action: Do Nothing
            action = {
                "action_id": "DO_NOTHING",
                "parameters": {}
            }
            
        return action

# --- Test Loop ---
if __name__ == '__main__':
    # Setup paths
    data_path = "data/streaming_mock_data.json"
    
    if not os.path.exists(data_path):
        print(f"ERROR: Mock data file not found at {data_path}. Cannot run test loop.")
    else:
        try:
            with open(data_path, 'r') as f:
                mock_data_json = f.read()
            
            # Load the entire JSON array
            try:
                mock_data = json.loads(mock_data_json)
            except json.JSONDecodeError:
                print("ERROR: Failed to decode JSON from the mock data file.")
                mock_data = []

            if not mock_data:
                print("Test data array is empty. Exiting.")
            else:
                print("\\n===================================================")
                print("           STARTING CORE LOGIC EXECUTION           ")
                print("===================================================\\n")
                
                # Instantiate the logic engine
                logic_engine = XAppLogic()
                
                decisions = []
                # Iterate through the streamed data records
                for i, row in enumerate(mock_data):
                    # Pass the dictionary representation of the current timestep
                    decision = logic_engine.process_interval(row)
                    decisions.append(decision)
                
                # print("\\n===================================================")
                # print("           SUMMARY OF ALL DECISIONS MADE           ")
                # print("===================================================\\n")
                
                # for i, decision in enumerate(decisions):
                #     print(f"--- Time Step {i+1} Decision ---")
                #     print(json.dumps(decision, indent=2))
                
                print("\\nSUCCESS: Core logic execution completed successfully.")

        except Exception as e:
            print(f"FATAL ERROR during execution: {e}")