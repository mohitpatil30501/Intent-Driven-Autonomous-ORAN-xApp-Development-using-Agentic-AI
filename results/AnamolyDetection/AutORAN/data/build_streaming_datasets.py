
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# --- Configuration ---
NUM_TRAINING_ROWS = 5000
NUM_TEST_ROWS = 1000
NUM_STREAM_ROWS = 300
OUTPUT_TRAINING_FILE = 'data/historical_training_data.csv'
OUTPUT_TEST_FILE = 'data/test_data.csv'
OUTPUT_STREAM_FILE = 'data/streaming_mock_data.json'

# --- Synthesis Parameters ---
# Base correlation: Throughput is proportional to PRB usage.
BASE_PRB = 20 + 10 * np.sin(np.linspace(0, 3, 1)) # Starting point for initial signal curve
BASE_THROUGHPUT_SCALE = 1.5 # Base throughput scale factor
BASE_NOISE_SCALE = 3

def generate_correlated_signals(num_rows, start_prb, start_throughput, noise_scale):
    """Generates time-series data respecting correlation."""
    print(f"Generating {num_rows} correlated signal points.")
    
    # Time index for diurnal/cyclic effect
    t = np.linspace(0, 2 * np.pi * (num_rows / 500), num_rows)
    
    # 1. Master Variable: PRB Usage (with cyclic diurnal pattern + random walk)
    # Start with a base load that oscillates (e.g., peak usage in the middle of the sequence)
    base_prb = 25 + 15 * np.sin(t)
    # Add random walk component for temporal continuity
    prb_signal = base_prb + np.cumsum(np.random.normal(0, 0.5, num_rows))
    
    # Clamp and cast to integer, ensuring it's always positive
    prb_signal = np.clip(np.round(prb_signal).astype(int), 10, 100)

    # 2. Correlated Variable: Throughput (proportional to PRB usage, plus noise)
    # Throughput = PRB * constant_factor * (1 + minor fluctuation)
    throughput_signal = (prb_signal * 0.8 + 50) + np.random.normal(0, noise_scale, num_rows)
    
    # Ensure minimum positive throughput
    throughput_signal = np.maximum(throughput_signal, 5.0)
    
    timestamps = list(range(int(datetime.now().timestamp()) - num_rows, int(datetime.now().timestamp()), 1))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'uplink_prb_usage': prb_signal,
        'uplink_throughput': throughput_signal
    })
    return df

def synthesize_ml_data(num_rows, start_prb, start_throughput):
    """Generates historical training and test data."""
    # Generate the main signal array
    df_full = generate_correlated_signals(num_rows, start_prb, start_throughput, BASE_NOISE_SCALE * 1.5)
    
    # Split into Train (80%) and Test (20%)
    split_idx = int(0.8 * num_rows)
    
    df_train = df_full.iloc[:split_idx].copy()
    df_test = df_full.iloc[split_idx:].copy()
    
    print(f"Training set size: {len(df_train)}")
    print(f"Test set size: {len(df_test)}")
    
    return df_train, df_test

def synthesize_streaming_data(num_rows, last_prb, last_throughput):
    """Generates the immediate next stream continuation."""
    # Generate a smaller, correlated segment starting near the last known point
    # We need about 50-100 rows for a convincing stream sample
    stream_df = generate_correlated_signals(num_rows, last_prb, last_throughput, BASE_NOISE_SCALE * 0.8)
    
    # Adjust timestamps to be sequential ticks from the end of the ML data
    end_timestamp = stream_df['timestamp'].max()
    stream_df['timestamp'] = np.arange(end_timestamp + 1, end_timestamp + 1 + num_rows)
    
    return stream_df

def main():
    print("--- Starting Data Generation Pipeline ---")
    
    # 1. Synthesize ML Data (Training & Test)
    # Use a seed for reproducibility across the synthetic run
    np.random.seed(42)
    
    # Generate a large enough set to derive train/test from
    df_train, df_test = synthesize_ml_data(NUM_TRAINING_ROWS + NUM_TEST_ROWS, 20, 15.0)
    
    # Save ML Dataframes
    df_train.to_csv(OUTPUT_TRAINING_FILE, index=False)
    df_test.to_csv(OUTPUT_TEST_FILE, index=False)
    
    # Determine the last point of the entire synthesized sequence for the stream continuity
    last_prb = df_test['uplink_prb_usage'].iloc[-1]
    last_throughput = df_test['uplink_throughput'].iloc[-1]
    
    # 2. Synthesize Streaming Data (Continuation)
    df_stream = synthesize_streaming_data(NUM_STREAM_ROWS, last_prb, last_throughput)
    
    # --- JSON Structure Formatting ---
    stream_data_list = []
    for i in range(len(df_stream)):
        row = df_stream.iloc[i]
        # The JSON structure requires a nested 'data' object keyed by the variables
        record = {
            "timestamp": int(row['timestamp']),
            "data": {
                "uplink_prb_usage": int(row['uplink_prb_usage']),
                "uplink_throughput": round(row['uplink_throughput'], 2)
            }
        }
        stream_data_list.append(record)
        
    # Save Stream JSON
    with open(OUTPUT_STREAM_FILE, 'w') as f:
        json.dump(stream_data_list, f, indent=2)

    print("\n--- Synthesis Complete ---")
    print(f"Saved Training Data: {OUTPUT_TRAINING_FILE}")
    print(f"Saved Test Data: {OUTPUT_TEST_FILE}")
    print(f"Saved Streaming Data: {OUTPUT_STREAM_FILE}")

if __name__ == "__main__":
    main()
