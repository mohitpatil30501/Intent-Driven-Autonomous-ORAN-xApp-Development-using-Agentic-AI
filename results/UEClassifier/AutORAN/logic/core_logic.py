import json
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any
import joblib

# --- Constants derived from Blueprint ---
MODEL_PATH = "ml/saved_model.pkl"
THRESHOLD = 0.25
# ---------------------------------------

class XAppLogic:
    """
    The core decision-making logic for resource scheduling optimization.
    This class is designed to be stateful and process streaming data intervals.
    """
    def __init__(self):
        """
        Initializes the logic, loading the ML model if available, 
        as required by the Supervised_ML cycle_Type.
        """
        print("Initializing XAppLogic: Attempting to load ML Model...")
        self.model = None
        self.feature_names = ["downlink_throughput", "uplink_throughput"]
        
        # Per Blueprint: Load model only if ML_Model_Artifacts exists (it does)
        if os.path.exists(MODEL_PATH):
            try:
                # In a real environment, we would load the model here.
                # Since we cannot guarantee the file existence during simulation, 
                # we structure the code to attempt loading and print success/failure.
                # with open(MODEL_PATH, 'rb') as f:
                #     self.model = pickle.load(f)
                self.model = joblib.load(MODEL_PATH)
                print(f"Successfully loaded ML model from {MODEL_PATH}.")
            except Exception as e:
                print(f"WARNING: Could not load ML model from {MODEL_PATH}. Error: {e}")
                # Fallback/Mocking mechanism if actual model loading fails/is absent
                self.model = "MOCK_MODEL_ACTIVE"
                print("Falling back to mock prediction logic.")
        else:
            print(f"WARNING: Model artifact not found at {MODEL_PATH}. Initializing with mock logic.")
            self.model = "MOCK_MODEL_ACTIVE"


    def process_interval(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single time-step (row) of KPM data to decide on resource allocation.
        
        Args:
            row_dict: Dictionary containing throughput measurements for a UE.
        
        Returns:
            A dictionary representing the selected action from Action_Space_Menu.
        """
        try:
            # 1. Extract Features
            # Assuming the input row_dict adheres to the KPI structure:
            downlink_throughput = row_dict.get("downlink_throughput", 0)
            uplink_throughput = row_dict.get("uplink_throughput", 0)
            
            features = np.array([[downlink_throughput, uplink_throughput]])
            
            predicted_load_score = 0.0
            
            # 2. Inference Step (ML Core)
            if self.model == "MOCK_MODEL_ACTIVE":
                # Mock Logic: Calculate a heuristic score based on total throughput
                total_throughput = downlink_throughput + uplink_throughput
                # Normalize mock score (e.g., scale by total magnitude / max possible)
                # To simulate an anomaly spike detection:
                predicted_load_score = np.clip(total_throughput / 1e12, 0.1, 1.5)
            else:
                # Real Model Prediction (Assuming model expects (N, 2) array)
                predicted_probabilities = self.model.predict_proba(features)
                predicted_load_score = predicted_probabilities[0][1] # Assuming class 1 is 'High Load'

            # 3. Decision Making based on Blueprint Threshold
            if predicted_load_score >= THRESHOLD:
                # High Load Detected -> Boost Resources
                # We must select an action matching the schema
                # The UE ID is required, assuming it's available in the input dict
                ue_id = row_dict.get("ue_id", 0) 
                return {
                    "action_id": "UPDATE_PRB_ALLOCATION", 
                    "parameters": {
                        "ue_id": int(ue_id), 
                        "prb_allocation_adjustment": 0.2, # Boost by 20%
                        "reason_code": 1 # Code for High Load Prediction
                    }
                }
            else:
                # Normal/Low Load -> No significant change required
                return {
                    "action_id": "DO_NOTHING", 
                    "parameters": {}
                }

        except Exception as e:
            print(f"ERROR processing interval: {e}")
            # Fail safe: Default to DO_NOTHING if processing fails
            return {
                "action_id": "DO_NOTHING", 
                "parameters": {}
            }


if __name__ == '__main__':
    # --- Mandatory Test Loop ---
    try:
        data_path = "data/streaming_mock_data.json"
        if not os.path.exists(data_path):
            print(f"FATAL: Mock data file not found at {data_path}. Cannot run test loop.")
            exit(1)

        with open(data_path, 'r') as f:
            streaming_data = json.load(f)

        logic_processor = XAppLogic()
        print("\n--- Starting Stream Simulation ---")
        
        all_decisions = []
        for index, record in enumerate(streaming_data):
            print(f"\n[Step {index+1}/{len(streaming_data)}]: Processing input...")
            
            # The input JSON structure from streaming_mock_data.json might differ from the 
            # structured KPI dictionary required by the method signature (which expects 
            # throughputs and an ID). We must structure the input dictionary to match 
            # the expected schema keys used in process_interval.
            
            # Assuming the mock data only contains the throughput values for the sake of the test loop:
            # We must synthesize the required structure based on the Blueprint's KPI:
            data = record.get("data", record)
            mock_row_dict = {
                "ue_id": data.get("ue_id", index + 1), # Use index as fallback ID
                "downlink_throughput": data.get("downlink_throughput", 0),
                "uplink_throughput": data.get("uplink_throughput", 0)
            }
            
            decision = logic_processor.process_interval(mock_row_dict)
            all_decisions.append(decision)
            print(f"--> Decision Made: {decision['action_id']}")

        print("\n--- Stream Simulation Complete ---")
        print("Final Decisions Summary:")
        for i, decision in enumerate(all_decisions):
            print(f"Time Step {i+1}: {json.dumps(decision)}")

    except Exception as e:
        print(f"\n[CRITICAL FAILURE IN TEST LOOP]: {e}")

