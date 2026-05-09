import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ricxappframe.xapp_frame import Xapp

# Constants
MSG_TYPE_MAC_INDICATION = 12002
MSG_TYPE_MAC_CONTROL = 12001

class SmartResourceXApp:
    def __init__(self, training_data_path):
        self.model = self._train_model(training_data_path)
        self.xapp = Xapp(entry_function=self.start, rmr_port=4560)
        self.xapp.register_callback(self.inference_handler, MSG_TYPE_MAC_INDICATION)

    def _train_model(self, path):
        """Trains the classifier on historical DL/UL TBS data"""
        df = pd.read_csv(path)
        # Features: dl_aggr_tbs, ul_aggr_tbs | Label: high_load
        X = df[['dl_aggr_tbs', 'ul_aggr_tbs']]
        y = df['high_load']
        
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X, y)
        print("Model trained successfully on historical patterns.")
        return clf

    def start(self, xapp):
        print("SmartResource xApp Active: Monitoring MAC Stream...")

    def inference_handler(self, summary, sbuf):
        """Processes live stream and applies ML-based PRB control"""
        try:
            payload = json.loads(summary['payload'])
            # The live stream mirrors historical format: dl_aggr_tbs, ul_aggr_tbs
            ue_id = payload.get("du_ue_id")
            live_data = [[payload['dl_aggr_tbs'], payload['ul_aggr_tbs']]]

            # Predict Load
            prediction = self.model.predict(live_data)[0]
            
            if prediction == 1:  # High-load detected
                self.send_control(ue_id, action="BOOST", ratio=0.9)
            else:                # Low-load detected
                self.send_control(ue_id, action="REDUCE", ratio=0.3)

        except Exception as e:
            print(f"Inference error: {e}")
        finally:
            self.xapp.rmr_free(sbuf)

    def send_control(self, ue_id, action, ratio):
        """Sends control message back to E2 Node"""
        ctrl = {
            "ue_id": ue_id,
            "prb_allocation_ratio": ratio,
            "action_type": action
        }
        self.xapp.rmr_send(json.dumps(ctrl).encode(), MSG_TYPE_MAC_CONTROL)
        print(f"Decision: {action} for UE {ue_id} (Ratio: {ratio})")

    def run(self):
        self.xapp.run()

if __name__ == "__main__":
    # Assumes a CSV with columns: dl_aggr_tbs, ul_aggr_tbs, high_load
    smart_xapp = SmartResourceXApp("historical_ue_data.csv")
    smart_xapp.run()
