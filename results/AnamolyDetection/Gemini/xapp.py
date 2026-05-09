import json
import numpy as np
from sklearn.ensemble import IsolationForest
from ricxappframe.xapp_frame import Xapp

# RMR Message Types (standard examples)
MSG_TYPE_MAC_INDICATION = 12002
MSG_TYPE_MAC_CONTROL = 12001

class AnomalyDetectionXApp:
    def __init__(self):
        # contamination=0.05 assumes 5% of devices might be anomalous
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.xapp = Xapp(entry_function=self.start, rmr_port=4560)
        self.xapp.register_callback(self.anomaly_handler, MSG_TYPE_MAC_INDICATION)

    def start(self, xapp):
        print("AnomalyDetectionMonitor: Running unsupervised UL analysis...")

    def anomaly_handler(self, summary, sbuf):
        try:
            payload = json.loads(summary['payload'])
            ue_data = payload.get("ue_list", [])
            
            if len(ue_data) < 5:  # Need a minimum sample size to establish a baseline
                return

            # 1. Prepare Features: [ul_prb_usage, ul_aggr_tbs]
            features = np.array([[u['ul_prb_usage'], u['ul_aggr_tbs']] for u in ue_data])
            ue_ids = [u['du_ue_id'] for u in ue_data]

            # 2. Fit and Predict (-1 for anomaly, 1 for normal)
            predictions = self.model.fit_predict(features)

            # 3. Process Results
            for i, result in enumerate(predictions):
                if result == -1:
                    ue_id = ue_ids[i]
                    print(f"!!! ANOMALY DETECTED: UE {ue_id} deviates from cell baseline.")
                    self.mitigate_rogue_ue(ue_id)

        except Exception as e:
            print(f"Processing error: {e}")
        finally:
            self.xapp.rmr_free(sbuf)

    def mitigate_rogue_ue(self, ue_id):
        """Reduces scheduling weight to protect cell resources"""
        ctrl_msg = {
            "ue_id": ue_id,
            "action": "SET_SCHEDULING_WEIGHT",
            "weight": 0.1,  # Minimum priority
            "reason": "Unsupervised Anomaly Detection"
        }
        self.xapp.rmr_send(json.dumps(ctrl_msg).encode(), MSG_TYPE_MAC_CONTROL)

    def run(self):
        self.xapp.run()

if __name__ == "__main__":
    monitor = AnomalyDetectionXApp()
    monitor.run()

