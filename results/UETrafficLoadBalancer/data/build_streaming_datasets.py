import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# --- Configuration ---
STREAMING_FILE = "data/streaming_mock_data.json"
NUM_RECORDS = 300 # Choosing a number between 100 and 500

# --- Simulation Parameters ---
# Base load parameters for synthesis
BASE_UE_COUNT = 100
BASE_UTILIZATION = 0.5 # 50%
NOISE_SCALE = 5

def synthesize_time_series_data(num_records):
    """
    Synthesizes correlated, time-series telemetry data for streaming mock data.
    Correlation: Utilization should rise as UE count rises.
    """
    # 1. Time vector (for accurate simulation timestamps)
    start_time = int(datetime.now().timestamp()) - (num_records * 60) # Start 5 minutes ago, spaced by 1 minute
    timestamps = np.arange(start_time, start_time + (num_records * 60), 60)

    # 2. Primary Signal: UE Count (Random Walk based on a diurnal cycle influence)
    t = np.arange(num_records)
    # Base diurnal cycle (simulating daily usage patterns)
    base_load_cycle = BASE_UE_COUNT + 50 * np.sin(t / 50.0)
    # Random walk component for local fluctuations
    random_walk = np.cumsum(np.random.normal(0, NOISE_SCALE, num_records))
    
    # Master Variable: ue_count_per_slice
    ue_count = base_load_cycle + random_walk
    ue_count = np.clip(ue_count, 50, 250).astype(int) # Keep within reasonable bounds

    # 3. Secondary Signal: Utilization Percentage (Correlated with UE Count)
    # Logic: Utilization = (UE Count / Capacity) * (1 + small variance)
    # Assume total capacity is fixed for simplicity, e.g., 300 UEs
    MAX_CAPACITY = 300.0
    utilization = (ue_count / MAX_CAPACITY) * 0.9 + np.random.normal(0, 0.03, num_records)
    utilization = np.clip(utilization, 0.1, 0.99)

    # 4. Structure the data to match the required JSON schema
    records = []
    for i in range(num_records):
        record = {
            "timestamp": int(timestamps[i]),
            "data": {
                "Telemetry_Variables": {
                    "INDICATION_MSG_AGENT_IF_ANS_V0": {
                        "ind": {
                            "slice": {
                                "msg": {
                                    "tstamp": int(timestamps[i]),
                                    "ue_slice_conf": {
                                        "len_ue_slice": 1, # Simplified assumption for structure fidelity
                                        "ues": {
                                            "rnti": [int(np.random.randint(1000, 9999))]
                                        }
                                    },
                                    "metrics": {
                                        "ue_count_per_slice": float(ue_count[i]),
                                     "utilization_percentage": float(utilization[i])
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        records.append(record)
    
    return records

def save_streaming_data(records):
    """Saves the list of records as a JSON array."""
    try:
        with open(STREAMING_FILE, 'w') as f:
            # The primary fix: json.dump handles standard Python types correctly.
            json.dump(records, f, indent=2)
        print(f"Successfully synthesized and saved streaming data to {STREAMING_FILE}")
    except Exception as e:
        print(f"Error saving streaming data: {e}")

if __name__ == "__main__":
    print("--- Starting O-RAN Data Generation Module ---")
    
    # 1. Generate the streaming mock data
    mock_records = synthesize_time_series_data(NUM_RECORDS)
    
    # 2. Save the data
    save_streaming_data(mock_records)
    
    print("--- Data Generation Complete ---")