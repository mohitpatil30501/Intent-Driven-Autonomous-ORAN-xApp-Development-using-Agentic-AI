import sys
import time
import xapp_sdk as ric

# Dynamically import the core logic written by Module 5
from logic.core_logic import XAppLogic

class UniversalCallback(ric.mac_cb):
    def __init__(self, node_id, app_logic):
        super().__init__()
        self.node_id = node_id
        self.app_logic = app_logic

    def handle(self, ind):
        # --- 1. EXTRACT TELEMETRY ---
        # Map the FlexRIC C-struct (ind) to a flat Python dictionary list
        row_dict = []
        for ue_stat in ind.ue_stats:
            row_dict.append({
                'ue_id': ue_stat.ue_id,
                'prb_utilization_percent': ue_stat.prb_utilization_percent,
                'aggregate_throughput_bps': ue_stat.aggregate_throughput_bps
            })
        
        # --- 2. EXECUTE CORE LOGIC ---
        decision = self.app_logic.process_interval(row_dict)

        # --- 3. EXECUTE CONTROL ACTIONS ---
        # Translate the dictionary decision back to FlexRIC control structures
        if decision.get("action_list"):
            for action_set in decision["action_list"]:
                if action_set.get("action_id") == "SET_PRB_ALLOCATION_RATIO":
                    params = action_set["parameters"]
                    ue_id = params["ue_id"]
                    prb_ratio = params["prb_ratio"]
                    min_prb_ratio = params["minimum_prb_ratio"]
                    
                    ctrl_msg = ric.mac_cb_ctrl_msg_t()
                    ctrl_msg.action = 1 # Placeholder: Assuming 1 corresponds to SET_PRB_ALLOCATION_RATIO
                    ctrl_msg.ue_id = ue_id
                    ctrl_msg.dl_prb_ratio = prb_ratio
                    ctrl_msg.min_dl_prb_ratio = min_prb_ratio # Assuming a dedicated field name in the C struct
                    ric.control_mac_sm(self.node_id, ctrl_msg)

def main():
    print("Initializing FlexRIC xApp...")
    ric.init()
    
    conn = ric.conn_e2_nodes()
    if len(conn) == 0:
        print("Error: No E2 Nodes connected.")
        return

    # Instantiate the independent logic brain (which loads the ML model if present)
    app_logic = XAppLogic()
    handlers=[]

    # Subscribe to the Service Model for all connected nodes
    for i in range(len(conn)):
        print(f"Subscribing to E2 Node: {conn[i].id}")
        cb = UniversalCallback(conn[i].id, app_logic)
        
        hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_s_10, cb)
        handlers.append(hndlr)

    print("Subscription successful. Listening for indications...")

    # Keep alive loop
    while ric.try_stop == 0:
        time.sleep(1)

    # Cleanup
    print("Shutting down...")
    for h in handlers:
        ric.rm_report_mac_sm(h)

if __name__ == '__main__':
    main()