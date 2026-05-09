import numpy as np
import pandas as pd
import json
import os

# --- Configuration ---
N_TRAIN = 5000
N_TEST = 1000
N_STREAM_MIN = 200
N_STREAM_MAX = 300

# --- Helper Functions for Synthesis ---

def generate_correlated_throughput(num_rows):
    """Generates correlated, time-series throughput data with occasional load spikes."""
    t = np.linspace(0, num_rows * 1.0, num_rows)
    
    # 1. Base Load (Diurnal cycle effect)
    base_load_factor = 0.01 + 0.005 * np.sin(t / 50)
    
    # 2. Underlying Noise/Fluctuation
    noise_dl = np.random.normal(0, 0.1, num_rows)
    noise_ul = np.random.normal(0, 0.08, num_rows)
    
    # 3. Master Variable (Simulating UE count correlation - using a derived metric)
    # Let's make a 'master_activity_factor' which influences both rates.
    master_activity = 1 + 0.1 * np.sin(t / 20) + 0.05 * np.random.normal(0, 1, num_rows)
    master_activity = np.clip(master_activity, 0.5, 1.5)

    # 4. Core Signal Generation (Bytes are large numbers)
    # Throughput = Base * Activity * Noise * Scaling_Factor
    dl_throughput = np.round(2e6 * base_load_factor * master_activity * (1 + noise_dl), 0).astype(np.uint64)
    ul_throughput = np.round(1.5e6 * base_load_factor * master_activity * (1 + noise_ul), 0).astype(np.uint64)

    # Introduce periodic anomalies (e.g., high load every 500 steps)
    spike_indices = np.arange(500, num_rows, 500)
    
    # FIX: Using advanced indexing to modify specific points
    dl_throughput[spike_indices] *= 3
    ul_throughput[spike_indices] *= 3

    return dl_throughput, ul_throughput

def synthesize_ml_data(num_rows):
    """Generates training/testing data with labels."""
    dl_throughput, ul_throughput = generate_correlated_throughput(num_rows)
    
    # 5. Label Generation (High load = High Throughput)
    # Simple thresholding based on combined throughput, mimicking the 'high_load' label.
    combined_throughput = dl_throughput + ul_throughput
    
    # Create a label: 1 if combined is above the 85th percentile, 0 otherwise.
    threshold_value = np.percentile(combined_throughput, 85)
    label = (combined_throughput >= threshold_value).astype(int)
    
    # Structure for the output CSV
    data = {
        'timestamp': np.arange(num_rows) * 10, # Simple increasing timestamp
        'ue_id': np.random.randint(1000, 9999, num_rows).astype(np.uint32),
        'downlink_throughput': dl_throughput,
        'uplink_throughput': ul_throughput,
        'high_load': label # This corresponds to the label mentioned in the blueprint
    }
    df = pd.DataFrame(data)
    return df

def synthesize_streaming_data(num_rows):
    """Generates a single set of correlated streaming data points."""
    # Generate enough points to sample from
    dl_throughput, ul_throughput = generate_correlated_throughput(num_rows)
    
    # The stream simulates reports over time, sampling every 10 steps of the generated array.
    streaming_records = []
    # We iterate in steps of 10 (the assumed interval for one report)
    for i in range(0, num_rows - 9, 10): 
        timestamp = (i + 1) * 10 # Simulate time passing: time advances with each report
        
        # Calculate the average for the next 10 data points to represent the interval
        avg_dl = np.mean(dl_throughput[i:i+10])
        avg_ul = np.mean(ul_throughput[i:i+10])
        
        record = {
            "timestamp": timestamp,
            "data": {
                # Streaming data structure must match the schema's required types
                "downlink_throughput": int(round(avg_dl)),
                "uplink_throughput": int(round(avg_ul))
            }
        }
        streaming_records.append(record)
        
    return streaming_records

# --- Main Execution ---
def main():
    print("--- Module 3 Data Engineering Pipeline Started ---")

    # 1. Synthesize ML Data (Training/Testing)
    print("Synthesizing Historical Training Data...")
    train_df = synthesize_ml_data(N_TRAIN)
    train_df.to_csv("data/historical_training_data.csv", index=False)

    print("Synthesizing Test Data...")
    test_df = synthesize_ml_data(N_TEST)
    test_df.to_csv("data/test_data.csv", index=False)

    # 2. Synthesize Streaming Data
    print("Synthesizing Streaming Mock Data...")
    stream_data = synthesize_streaming_data(N_STREAM_MAX)
    
    # Writing to JSON format as specified in the prompt
    with open("data/streaming_mock_data.json", 'w') as f:
        json.dump(stream_data, f, indent=2)

    print("--- Data Synthesis Complete ---")

if __name__ == "__main__":
    main()