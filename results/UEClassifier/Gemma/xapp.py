

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple

# --- Configuration Constants ---
MODEL_PATH = "load_classifier_model.joblib"
INITIAL_PRB_ALLOCATION = 15 # Base PRB allocation assumed if no prediction is made

class ResourceController:
    """
    The core xApp intelligence. Handles prediction, decision-making, and resource recommendation.
    """
    def __init__(self, model_path: str):
        print("="*80)
        print(f"⚡️ [xApp Initializing] Loading Model from: {model_path}")
        try:
            # Load the pre-trained model
            self.model = joblib.load(model_path)
            print("✅ Model loaded successfully. Resource distribution intelligence active.")
        except FileNotFoundError:
            print("❌ ERROR: Model file not found. Run the 'TRAIN' phase first.")
            self.model = None

    def predict_load_state(self, dl_bytes: float, ul_bytes: float) -> Tuple[int, float]:
        """
        Uses the loaded model to predict if a UE is high or low load.
        Returns (predicted_load_label, probability).
        """
        if self.model is None:
            raise Exception("Model is not loaded. Cannot make predictions.")

        # The input must be formatted as a 2D array/DataFrame matching the training features
        features = pd.DataFrame([[dl_bytes, ul_bytes]], 
                                columns=['dl_aggr_tbs', 'ul_aggr_tbs'])
        
        # Predict the label (0 or 1)
        prediction = self.model.predict(features)[0]
        # Predict the probability (useful for confidence scoring)
        probability = self.model.predict_proba(features)[0]
        
        return int(prediction), probability[1]

    def determine_resource_action(self, 
                                  dl_bytes: float, 
                                  ul_bytes: float, 
                                  prediction_confidence: float) -> str:
        """
        Translates the model prediction into a specific resource allocation action.
        """
        if self.model is None:
             return "ERROR: Cannot make decision (Model missing)."

        # Prediction interpretation (Assuming 1 = High Load, 0 = Low Load)
        predicted_load = self.model.predict(pd.DataFrame([[dl_bytes, ul_bytes]], 
                                                      columns=['dl_aggr_tbs', 'ul_aggr_tbs']))[0]

        print(f"\n[Controller] Model Prediction: Load State = {'High' if predicted_load == 1 else 'Low'} Load.")
        
        if predicted_load == 1:
            # High load predicted: BOOST resources
            boost_ratio = 1.20 # +20% allocation
            print(f"📈 Decision: Predicted High Load. Boosting resources by 20%.")
            return f"BOOST: Increase PRB allocation by 20%. New estimated PRB budget: {int(INITIAL_PRB_ALLOCATION * boost_ratio)} PRBs."
        else:
            # Low load predicted: REDUCE resources (to free up capacity for others)
            reduce_ratio = 0.85 # -15% allocation
            print(f"📉 Decision: Predicted Low Load. Reducing resources by 15%.")
            return f"REDUCE: Decrease PRB allocation by 15%. New estimated PRB budget: {int(INITIAL_PRB_ALLOCATION * reduce_ratio)} PRBs."


# ==========================================================
# PHASE 1: TRAINING PHASE (Run this first!)
# ==========================================================

def train_classifier(data_path: str = "historical_load_data.csv"):
    """
    Loads historical data, trains the RandomForestClassifier, and saves the model.
    """
    print("\n" + "="*80)
    print("        =================== TRAINING PHASE =====================")
    print("="*80)
    
    try:
        # 1. Load Data
        # We use a simulated dataset since the input format is specified.
        # Create a mock file matching the required structure:
        historical_data = {
            'dl_aggr_tbs': [1024, 2048, 512, 512, 4096, 8192, 2048, 1024, 1024, 2048],
            'ul_aggr_tbs': [512, 2048, 128, 512, 1024, 512, 1024, 512, 1024, 2048],
            'high_load': [1, 1, 0, 0, 1, 1, 1, 0, 0, 1] # 1 = High Load, 0 = Low Load
        }
        df_historical = pd.DataFrame(historical_data)
        df_historical.to_csv(data_path, index=False)
        print(f"✅ Mock Historical Data created and saved to {data_path}")

        # 2. Separate Features (X) and Target (y)
        X = df_historical[['dl_aggr_tbs', 'ul_aggr_tbs']].values
        y = df_historical['high_load'].values

        # 3. Train Model
        # Using RandomForest for robustness
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 4. Evaluation (Optional but recommended)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"\n[Evaluation] Model trained successfully. Accuracy on training data: {accuracy:.2f}")

        # 5. Save Model
        joblib.dump(model, MODEL_PATH)
        print(f"⭐ Success! Model saved to {MODEL_PATH} and ready for runtime use.")

    except Exception as e:
        print(f"An error occurred during training: {e}")


# ==========================================================
# PHASE 2: RUNTIME SIMULATION
# ==========================================================

def run_live_simulation():
    """
    Simulates the xApp running continuously, receiving live data and making decisions.
    """
    print("\n\n" + "="*80)
    print("     ⚡️  STARTING LIVE RUNTIME SIMULATION (DECISION MAKING) ⚡️")
    print("="*80)

    controller = ResourceController(model_path=MODEL_PATH)
    
    if controller.model is None:
        print("\n[STOP] Cannot run simulation. Please run the 'TRAIN' phase first.")
        return

    # Simulated live data samples (matching the [DL_BYTES, UL_BYTES] format)
    # Note: The values are chosen to trigger different outcomes:
    # - Sample 1: High DL, High UL -> Predicted High Load -> BOOST
    # - Sample 2: Low DL, Low UL -> Predicted Low Load -> REDUCE
    # - Sample 3: Moderate DL, Moderate UL -> Predicted High Load (due to recent patterns) -> BOOST
    live_data_samples = [
        (4096, 2048), 
        (512, 128), 
        (2048, 1024)
    ]
    
    for i, (dl_bytes, ul_bytes) in enumerate(live_data_samples):
        print("\n\n" + "#"*70)
        print(f"               [ LIVE DATA SAMPLE {i+1} ]")
        print(f"                Received Data: DL={dl_bytes} bytes, UL={ul_bytes} bytes")
        print("#"*70)
        
        try:
            # 1. Prediction
            prediction_label, confidence = controller.predict_load_state(dl_bytes, ul_bytes)
            
            print(f"[Metrics] Predicted Load: {'High' if prediction_label == 1 else 'Low'} (Confidence: {confidence*100:.2f}%)")
            
            # 2. Decision & Resource Recommendation
            action_result = controller.determine_resource_action(dl_bytes, ul_bytes, confidence)
            
            print("\n=========================================================")
            print(f"✅ FINAL XAPP OUTPUT (Action Recommendation): {action_result}")
            print("=========================================================\n")
            
        except Exception as e:
            print(f"[CRITICAL ERROR] Simulation halted: {e}")
            break


if __name__ == "__main__":
    # Step 1: ALWAYS run the training phase first to create the model
    train_classifier()
    
    # Step 2: Run the runtime simulation using the generated model
    run_live_simulation()

