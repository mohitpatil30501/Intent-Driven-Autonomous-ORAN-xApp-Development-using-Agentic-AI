import numpy as np
import pandas as pd
import json
import time
import random

# --- Configuration ---
OUTPUT_FILE = "data/streaming_mock_data.json"
NUM_STEPS = 300  # Generate 300 time steps
MIN_UE = 5
MAX_UE = 20
BASE_RATE_BPS = 50000 # Base aggregate throughput contribution per active UE

def synthesize_telemetry_data(num_steps):
    """
    Synthesizes correlated, time-series telemetry data for streaming mock JSON.
    Correlation logic: Throughput increases with PRB utilization, but congestion 
    limits the maximum sustained rate.
    """
    
    telemetry_data = []
    # Use a fixed, high starting time to prevent drift issues in mocked data
    start_time = int(time.time()) - (num_steps * 2) 
    current_timestamp = start_time
    
    print(f"Synthesizing {num_steps} time steps...")

    for i in range(num_steps):
        # 1. TEMPORAL CONTINUITY (Time advancement)
        current_timestamp += 1
        
        # 2. MASTER VARIABLE: Simulate fluctuating number of connected UEs
        # Use a slightly cyclical pattern for realism
        ues_count = int(np.clip(np.sin(i / 50.0) * 0.5 + 1.5) * (MAX_UE - MIN_UE) / 2 + MIN_UE)
        
        # 3. SIGNAL GENERATION (Vectorized Logic)
        # Base utilization (sinusoidal cycle + noise)
        base_util = 50 + 30 * np.sin(i / 70.0)
        noise_factor = np.random.normal(0, 5, 1)[0]
        # Ensure casting to standard Python int type immediately
        prb_util_percent = np.clip(base_util + noise_factor, 10, 95).astype(int)
        
        # Cross-Feature Correlation: Throughput depends on utilization and UE count
        throughput_base = ues_count * BASE_RATE_BPS * (prb_util_percent / 100.0)
        throughput_noise = np.random.normal(0, 150000, 1)[0]
        # Ensure casting to standard Python int type immediately
        aggregate_throughput_bps = int(np.clip(throughput_base + throughput_noise, 10000, 500000))
        
        # 4. STRUCTURE DATA (Simulate multiple UEs reporting in one message batch)
        ue_stats_list = []
        # Simulate UEs reporting, but ensure at least 1 UE is present
        num_reporting_ues = max(1, int(np.clip(np.sin(i / 100.0) * 0.8 + 1.5) * (MAX_UE - MIN_UE) / 2 + MIN_UE) - 1)
        
        for j in range(num_reporting_ues):
            # Simulate per-UE variation around the mean for this time step
            ue_id = 1000 + (i * 7 + j * 3) % 500
            
            # Small random jitter around the global trend for individual UE reporting
            ue_prb_util = np.clip(prb_util_percent * (0.8 + np.random.uniform(-0.1, 0.1)), 5, 95).astype(int)
            ue_throughput = int(aggregate_throughput_bps * (0.8 + np.random.uniform(-0.1, 0.1)))
            
            ue_stats_list.append({
                "ue_id": int(ue_id),
                "prb_utilization_percent": int(ue_prb_util),
                "aggregate_throughput_bps": int(ue_throughput)
            })

        message = {
            "tstamp": current_timestamp,
            "ue_stats": ue_stats_list
        }
        
        telemetry_data.append(message)

    return telemetry_data

def write_output(data):
    """Writes the list of messages to the specified JSON file."""
    print(f"Writing {len(data)} telemetry records to {OUTPUT_FILE}...")
    
    # Writing the entire list of structured messages
    with open(OUTPUT_FILE, 'w') as f:
        # The casting within the loop should handle this now
        json.dump(data, f, indent=2)
    
    print("Successfully wrote streaming mock data.")

if __name__ == "__main__":
    # --- Synthesis Logic Execution ---
    
    # 1. Generate the core data structure
    mock_data = synthesize_telemetry_data(NUM_STEPS)
    
    # 2. Write the output file
    write_output(mock_data)