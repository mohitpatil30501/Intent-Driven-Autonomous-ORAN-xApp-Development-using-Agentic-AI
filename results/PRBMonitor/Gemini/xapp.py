from ricxappframe.xapp_frame import Xapp
import json

# Configuration constants
PRB_THRESHOLD = 85.0
THROTTLE_RATIO = 0.5  # Reduce allocation by 50%
RIC_CONTROL_MSG_TYPE = 12001  # Example message type for MAC Control
RIC_INDICATION_MSG_TYPE = 12002

class ResourceGuardXApp:
    def __init__(self):
        # Initialize xApp with a generic entry point
        self.xapp = Xapp(entry_function=self.start, rmr_port=4560)
        self.xapp.register_callback(self.mac_metrics_handler, RIC_INDICATION_MSG_TYPE)

    def start(self, xapp):
        print("ResourceGuard xApp: Monitoring DL PRB Usage...")

    def mac_metrics_handler(self, summary, sbuf):
        """Processes incoming MAC Indication messages"""
        try:
            payload = json.loads(summary['payload'])
            ue_list = payload.get("ue_metrics", [])

            for ue in ue_list:
                ue_id = ue.get("du_ue_id")
                prb_usage = ue.get("dl_prb_usage_percent", 0)
                throughput = ue.get("dl_aggr_throughput_mbps", 0)

                print(f"UE: {ue_id} | PRB: {prb_usage}% | TP: {throughput} Mbps")

                # Throttling Logic
                if prb_usage > PRB_THRESHOLD:
                    print(f"!!! ALERT: UE {ue_id} exceeding threshold ({prb_usage}%). Throttling...")
                    self.throttle_ue(ue_id)

        except Exception as e:
            print(f"Error parsing MAC metrics: {e}")
        finally:
            self.xapp.rmr_free(sbuf)

    def throttle_ue(self, ue_id):
        """Sends an E2 Control message to reduce PRB allocation ratio"""
        control_payload = {
            "ue_id": ue_id,
            "action": "SET_PRB_QUOTA",
            "max_prb_ratio": THROTTLE_RATIO,
            "priority": "LOW"
        }
        
        # Send via RMR to the E2 Termination
        success = self.xapp.rmr_send(json.dumps(control_payload).encode(), RIC_CONTROL_MSG_TYPE)
        if success:
            print(f"Control command sent for UE {ue_id}")

    def run(self):
        self.xapp.run()

if __name__ == "__main__":
    monitor = ResourceGuardXApp()
    monitor.run()

