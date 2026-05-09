import json
from ricxappframe.xapp_frame import Xapp

# RMR Message Types for Slice Service Model
MSG_TYPE_SLICE_INDICATION = 12005
MSG_TYPE_SLICE_CONTROL = 12006

class SliceLoadBalancer:
    def __init__(self):
        # Configuration
        self.UE_LIMIT = 50
        self.UTIL_LIMIT = 80.0
        
        # Initialize xApp
        self.xapp = Xapp(entry_function=self.start, rmr_port=4560)
        self.xapp.register_callback(self.slice_monitor_handler, MSG_TYPE_SLICE_INDICATION)

    def start(self, xapp):
        print("SliceLoadBalancer: Monitoring slice health and UE distribution...")

    def slice_monitor_handler(self, summary, sbuf):
        """Processes periodic slice telemetry"""
        try:
            payload = json.loads(summary['payload'])
            slices = payload.get("slices", [])

            for slc in slices:
                s_id = slc.get("slice_id")
                ue_count = slc.get("ue_count", 0)
                utilization = slc.get("utilization_percent", 0.0)

                print(f"Slice {s_id} -> UEs: {ue_count}, Util: {utilization}%")

                # Threshold Logic
                if ue_count > self.UE_LIMIT or utilization > self.UTIL_LIMIT:
                    reason = "UE count" if ue_count > self.UE_LIMIT else "Utilization"
                    print(f"!!! CRITICAL: Slice {s_id} overloaded by {reason}. Triggering Steering.")
                    self.steer_traffic(s_id)

        except Exception as e:
            print(f"Telemetry Error: {e}")
        finally:
            self.xapp.rmr_free(sbuf)

    def steer_traffic(self, slice_id):
        """
        Sends a Control Message to the E2 Node via Slice Service Model.
        Action: Admission Control or UE Re-association.
        """
        control_payload = {
            "slice_id": slice_id,
            "command": "ADMISSION_CONTROL_ENABLE",
            "action": "STEER_NEW_UES_TO_SECONDARY",
            "max_prb_quota": 70  # Cap the overloaded slice to save cell resources
        }
        
        success = self.xapp.rmr_send(json.dumps(control_payload).encode(), MSG_TYPE_SLICE_CONTROL)
        if success:
            print(f"Control signal successfully sent for Slice {slice_id}")

    def run(self):
        self.xapp.run()

if __name__ == "__main__":
    lb = SliceLoadBalancer()
    lb.run()

