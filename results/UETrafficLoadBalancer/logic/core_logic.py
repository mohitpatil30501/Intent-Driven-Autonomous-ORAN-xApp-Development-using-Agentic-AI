import json
import os

class XAppLogic:
    """
    Core logic component for slice load balancing.
    Processes streaming Key Performance Metric (KPM) data to decide if
    load rebalancing actions are necessary based on utilization thresholds.
    """
    def __init__(self):
        # Per blueprint: cycle_Type is Pure_Logic and ML_Model_Artifacts is empty,
        # so __init__ must be empty (no model loading).
        pass

    def process_interval(self, row_dict: dict) -> dict:
        """
        Determines the required action based on current slice metrics.

        :param row_dict: A dictionary representing one timestep of KPM data.
        :return: A dictionary matching an action from the Action_Space_Menu.
        """
        # Extract required metrics from the mock structure
        # Assuming the input structure contains the processed metrics for the slice.
        ue_count = row_dict.get("ue_count_per_slice", 0)
        util_percent = row_dict.get("utilization_percentage", 0.0)
        slice_id = row_dict.get("slice_id", 1) # Assuming slice_id is passed or known

        # Define an operational overload threshold (e.g., 85%)
        OVERLOAD_THRESHOLD = 85.0

        print(f"--- Processing time step for Slice {slice_id} ---")
        print(f"  Metrics: UE Count={ue_count}, Utilization={util_percent:.2f}%")

        # Decision Logic: Check if utilization significantly exceeds the operational threshold
        if util_percent > OVERLOAD_THRESHOLD:
            print(f"  [ALERT] Utilization ({util_percent:.2f}%) exceeds threshold ({OVERLOAD_THRESHOLD}%). Initiating load adjustment.")
            
            # Strategy: If heavily overloaded, force immediate rebalance, 
            # and concurrently update the threshold to acknowledge the transient spike.
            
            # 1. Force immediate load rebalance
            force_action = {
                "action_id": "FORCE_LOAD_REBALANCE",
                "parameters": {"slice_id": slice_id}
            }
            
            # 2. Update threshold (a defensive measure during heavy load)
            update_action = {
                "action_id": "UPDATE_LOAD_BALANCING_THRESHOLD",
                "parameters": {
                    "slice_id": slice_id,
                    "new_threshold_percent": 95, # Temporarily raising threshold acknowledgment
                    "rebalance_enabled": True
                }
            }
            
            # In a real system, these might be emitted sequentially or via a multi-action call.
            # For this simplified logic, we return the most critical/first action, 
            # but logging both is helpful for verification. We will return the force action.
            return force_action
        
        elif util_percent > (OVERLOAD_THRESHOLD - 10.0): # Warning zone (e.g., 75%-85%)
             print(f"  [WARN] Utilization is high ({util_percent:.2f}%). Monitoring closely.")
             return {
                 "action_id": "UPDATE_LOAD_BALANCING_THRESHOLD",
                 "parameters": {
                     "slice_id": slice_id,
                     "new_threshold_percent": 85, 
                     "rebalance_enabled": True
                 }
             }
        
        else:
            print("  [INFO] Utilization within normal operational bounds. No action required.")
            return {
                "action_id": "DO_NOTHING",
                "parameters": {}
            }

# Test Loop Execution Block
if __name__ == '__main__':
    # --- Setup Mock Data Loading ---
    try:
        MOCK_DATA_PATH = "data/streaming_mock_data.json"
        with open(MOCK_DATA_PATH, 'r') as f:
            data_payload = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Mock data file not found at {MOCK_DATA_PATH}. Cannot run simulation.")
        exit(1)
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {MOCK_DATA_PATH}.")
        exit(1)

    # --- Execution ---
    print("\\n======================================================")
    print("Starting XApp Logic Simulation: Slice Load Balancer")
    print("======================================================\\n")

    logic_processor = XAppLogic()
    
    all_actions = []
    for i, row in enumerate(data_payload):
        # Modify the row dictionary to include a deterministic slice_id for logging clarity
        row['slice_id'] = 100 + (i % 3) # Simulate different slices
        
        # Process the data point and capture the resulting action decision
        action = logic_processor.process_interval(row)
        all_actions.append(action)

    print("\\n======================================================")
    print("Simulation Complete. Summary of Decisions:")
    for i, action in enumerate(all_actions):
        print(f"Step {i+1}: Action Output -> {json.dumps(action)}")
    print("======================================================")