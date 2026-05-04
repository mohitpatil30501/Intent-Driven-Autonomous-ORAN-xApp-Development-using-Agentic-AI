import numpy as np
import pandas as pd
import json
import os
import time

# --- Configuration ---
NUM_RECORDS = 300  # Aiming for 300 records (approx 30 seconds of 10ms intervals)
OUTPUT_FILE = "data/streaming_mock_data.json"

# --- Synthesis Logic ---
def generate_synthetic_stream_data(num_records):
    """
    Generates mathematically and logically consistent synthetic streaming data
    for slice metadata.
    """
    print("--- Starting Synthetic Data Generation ---")
    
    # 1. Time Vector
    timestamps = np.arange(num_records) * 0.01
    
    # 2. Core Signal: Active UE Count (Random Walk)
    base_ue_count = 100
    noise = np.random.normal(0, 5, num_records)
    # Constrain UE count to be positive
    active_ue_count = np.clip(base_ue_count + np.cumsum(noise), 1, 500).astype(int)
    
    # 3. Cyclic Signal: Slice Type (Cycle through plausible types)
    slice_types = ["eMBB", "URLLC", "mMTC"]
    slice_type_indices = np.arange(num_records) // 50 % len(slice_types)
    slice_type_details = [slice_types[i] for i in slice_type_indices]
    
    # 4. Correlated Signal: Slice ID (Sequential/Slightly changing)
    slice_id = 1000 + (np.arange(num_records) // 10)
    
    # 5. Correlated Signal: Slice Name (Based on type)
    slice_name = np.array([
        f"{slice_types[i]}-{i % 3}" for i in slice_type_indices
    ])
    
    # 6. Metadata Synthesis (Must correlate with active_ue_count)
    # High UE count -> Higher utilization/throughput
    congestion_factor = (active_ue_count / 500.0)**2
    
    # Example Metadata metrics:
    # Signal 1: Total Throughput (Mbps) - Increases with UE count
    throughput = 10 + 50 * (1 - congestion_factor) + np.random.normal(0, 5, num_records)
    
    # Signal 2: Average Latency (ms) - Increases when congestion increases
    latency = 5 + (congestion_factor * 15) + np.random.normal(0, 1, num_records)
    
    # Signal 3: Buffer Occupancy (%) - Stays relatively stable but related to congestion
    buffer_occupancy = 30 + (congestion_factor * 20) + np.random.normal(0, 5, num_records)
    
    # Structure the data according to the Technical Mapping schema
    telemetry_data = []
    for i in range(num_records):
        record = {
            "timestamp": int(timestamps[i] * 1000), # Convert seconds to milliseconds for better simulation
            "data": {
                "slice_id": int(slice_id[i]),
                "slice_name": str(slice_name[i]),
                "ue_statistics": {
                    "active_ue_count": int(active_ue_count[i])
                },
                "slice_type_details": str(slice_type_details[i]),
                "metadata": {
                    "total_throughput_mbps": round(max(0, throughput[i]), 2),
                    "avg_latency_ms": round(max(1, latency[i]), 1),
                    "buffer_occupancy_percent": round(min(100, max(0, buffer_occupancy[i])), 1)
                }
            }
        }
        telemetry_data.append(record)
        
    return telemetry_data

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Generate Data
    synthetic_data = generate_synthetic_stream_data(NUM_RECORDS)
    
    # 2. Write JSON Output
    print(f"Writing {NUM_RECORDS} records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(synthetic_data, f, indent=2)
    
    print("Data generation complete.")