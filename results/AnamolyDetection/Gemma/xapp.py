

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple

# --- Configuration Constants ---
MODEL_PATH = "uplink_anomaly_model.joblib"
CONTAMINATION_RATE = 0.05  # Assume 5% of observations are anomalies
INITIAL_SCHEDULING_WEIGHT = 1.0 # Default weight
MAX_SYSTEM_UP_PRB = 80 # Total uplink PRB capacity in the cell

class AnomalyDetector:
    """
    Handles the unsupervised learning model for anomaly detection.
    Uses Isolation Forest to learn the 'normal' operating range.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.features = ['ul_prb_used', 'ul_throughput']
        
    def train_model(self, clean_data: pd.DataFrame):
        """
        Trains the Isolation Forest model on data assumed to be normal cell behavior.
        """
        print("\n" + "="*80)
        print("        === TRAINING ANOMALY DETECTOR PHASE ===")
        print("="*80)
        
        X_train = clean_data[self.features].values
        
        # Initialize and train Isolation Forest
        # contamination estimates the proportion of outliers in the data.
        self.model = IsolationForest(
            contamination=CONTAMINATION_RATE, 
            random_state=42
        )
        self.model.fit(X_train)
        
        # Save the model for runtime use
        joblib.dump(self.model, self.model_path)
        print(f"✅ Model trained successfully and saved to {self.model_path}.")

    def predict_anomaly(self, prb_used: float, throughput: float) -> Tuple[int, float]:
        """
        Scores a new data point against the learned baseline.
        Returns (is_anomaly: boolean, anomaly_score: float).
        """
        if self.model is None:
            raise Exception("Model is not loaded. Cannot perform detection.")

        # Input must match the feature format
        features = pd.DataFrame([[prb_used, throughput]], 
                                columns=self.features)
        
        # Decision function: returns -1 for outliers (anomalies) and 1 for normal
        prediction = self.model.predict(features)[0]
        
        # The score measures how isolated the point is; lower scores mean higher anomaly.
        anomaly_score = self.model.decision_function(features)[0]
        
        return prediction == -1, anomaly_score

class ResourceController:
    """
    The main xApp logic. Observes incoming data, detects anomalies, and recommends actions.
    """
    def __init__(self, detector: AnomalyDetector):
        self.detector = detector
        # State tracking: {UE_ID: current_weight}
        self.ue_weights: Dict[str, float] = {}

    def process_stream_data(self, ue_id: str, prb_used: float, throughput: float):
        """
        Core function called every time a new sample arrives from the MAC SM stream.
        """
        print("-" * 70)
        print(f"⚙️ PROCESSING UE: {ue_id} | UL PRB: {prb_used:.1f} | Throughput: {throughput:.2f} Mbps")
        
        # 1. Anomaly Detection
        is_anomaly, score = self.detector.predict_anomaly(prb_used, throughput)
        
        print(f"[Detector] Anomaly Detected: {'YES' if is_anomaly else 'NO'} (Score: {score:.4f})")
        
        # 2. Decision Logic
        current_weight = self.ue_weights.get(ue_id, INITIAL_SCHEDULING_WEIGHT)
        new_weight = current_weight
        action_taken = False

        if is_anomaly:
            print("🚨 ANOMALY ALERT: Potential Rogue/Malfunctioning Device Detected.")
            
            # Reduce the scheduling weight significantly (aggressive throttling)
            # We reduce it by 30-50% to force the scheduler to deprioritize it.
            reduction_factor = 0.60 
            new_weight = max(0.1, current_weight * reduction_factor)
            
            action_taken = True
        elif current_weight < INITIAL_SCHEDULING_WEIGHT * 0.9:
            # If the UE is not anomalous, but we had previously throttled it, 
            # and it's now behaving normally, slowly boost it back up.
            new_weight = min(INITIAL_SCHEDULING_WEIGHT, current_weight * 1.1)
            action_taken = True
        else:
            # Normal operation and no anomaly detected, keep current weight.
            new_weight = current_weight
        
        # 3. Enforcement & State Update
        self.ue_weights[ue_id] = new_weight
        
        # Simulate calling the MAC/Scheduler API
        if action_taken:
            self._enforce_weight_change(ue_id, current_weight, new_weight)
        else:
             print(f"[Controller] Behavior normal. Maintaining weight at {new_weight:.2f}.")


    def _enforce_weight_change(self, ue_id: str, old_weight: float, new_weight: float):
        """
        Mocks the interaction with the MAC Scheduler to apply the new priority weight.
        """
        print("\n" + "!"*70)
        print(f"🔥🔥 [ACTION RECOMMENDED] For {ue_id}: Changing Uplink Scheduling Weight.")
        print(f"       [OLD WEIGHT]: {old_weight:.2f} -> [NEW WEIGHT]: {new_weight:.2f}")
        print(f"       Objective: Resource control and potential mitigation of rogue behavior.")
        print("!"*70)


# ===================================================================
# PHASE 1: TRAINING PHASE (Must be run first)
# ===================================================================

def train_anomaly_detector():
    """
    Simulates loading a baseline of normal traffic and training the unsupervised model.
    """
    print("\n" + "="*80)
    print("        === TRAINING ANOMALY DETECTOR PHASE (BASELINE) ===")
    print("="*80)
    
    # --- MOCK DATA CREATION ---
    # Simulate several UEs over time, generating normal traffic profiles.
    normal_data_list = []
    
    # Baseline UEs (Normal behavior)
    for i in range(100):
        # Low Traffic UE (Baseline)
        dl_tbs = np.random.uniform(50, 200)
        ul_tbs = np.random.uniform(10, 50)
        normal_data_list.append((dl_tbs, ul_tbs))
    
    # Moderate Traffic UE (Baseline)
    for i in range(100):
        dl_tbs = np.random.uniform(300, 600)
        ul_tbs = np.random.uniform(50, 150)
        normal_data_list.append((dl_tbs, ul_tbs))

    # Creating the clean DataFrame for training
    df_clean = pd.DataFrame(normal_data_list, columns=['dl_aggr_tbs', 'ul_aggr_tbs'])
    print(f"✅ Generated baseline dataset size: {len(df_clean)} observations.")
    
    # Initialize and train the detector
    detector = AnomalyDetector(MODEL_PATH)
    detector.train_model(df_clean)
    
    return detector

# =========================================================
# PHASE 2: RUNTIME SIMULATION
# ===================================================================

def run_live_simulation(detector: AnomalyDetector):
    """
    Simulates receiving live data samples, including normal and anomalous packets.
    """
    print("\n\n" + "="*80)
    print("     ⚡️  STARTING LIVE RUNTIME SIMULATION (ANOMALY DETECTION) ⚡️")
    print("="*80)

    controller = ResourceController(detector)
    
    # Simulated Data Scenarios:
    # 1. Normal Operation: Low UL usage (Should be flagged as normal)
    # 2. Minor Anomaly: Moderate UL usage, slightly outside normal range (Might trigger warning/slight reduction).
    # 3. Major Anomaly (Rogue/Malfunction): Extremely high UL PRB usage (Should trigger strong reduction).
    # 4. Recovery: Data returns to normal after an anomaly.
    
    live_data_samples = [
        # 1. Normal Low Load
        ("UE_001", 50.0, 120.0), 
        # 2. Minor Anomaly (Slightly elevated)
        ("UE_002", 70.0, 160.0), 
        # 3. Major Anomaly (Rogue Device)
        ("UE_003", 300.0, 250.0), 
        # 4. Recovery Check
        ("UE_003", 70.0, 130.0) 
    ]
    
    for i, (ue_id, prb_used, throughput) in enumerate(live_data_samples):
        print("\n\n" + "-"*70)
        print(f"               --- SIMULATION CYCLE {i+1} ---")
        controller.process_stream_data(ue_id, prb_used, throughput)
        time.sleep(0.5) # Pause for readability

if __name__ == "__main__":
    # 1. Train the detector first to establish the cell baseline
    trained_detector = train_anomaly_detector()
    
    # 2. Run the live simulation
    run_live_simulation(trained_detector)

