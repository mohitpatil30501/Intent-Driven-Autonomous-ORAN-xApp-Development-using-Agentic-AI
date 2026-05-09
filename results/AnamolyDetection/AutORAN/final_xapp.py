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
        # Map the FlexRIC C-struct (ind) to a flat Python dictionary
        row_dict = {}
        
        # Telemetry Mapping for MAC Indication: Iterate over the UE stats array
        if hasattr(ind, 'mac_stats') and ind.mac_stats:
            for i in range(len(ind.mac_stats)):
                stat = ind.mac_stats[i]
                # Keying the dictionary by UE ID to facilitate ML input processing
                row_dict[stat.ue_id] = {
                    "uplink_prb_usage": stat.uplink_prb_usage,
                    "uplink_throughput": stat.uplink_throughput
                }
        
        # --- 2. EXECUTE CORE LOGIC ---
        decision = self.app_logic.process_interval(row_dict)

        # --- 3. EXECUTE CONTROL ACTIONS ---
        # Translate the dictionary decision back to FlexRIC control structures
        if decision.get("action_id") == "SET_UE_SCHEDULING_WEIGHT":
            # Assuming decision["parameters"] contains {"ue_id": ..., "new_weight": ...}
            params = decision["parameters"]
            ue_id = params["ue_id"]
            new_weight = params["new_weight"]
            
            ctrl_msg = ric.control_mac_sm_ctrl_msg_t()
            # Assuming action ID 0 maps to SET_UE_SCHEDULING_WEIGHT based on typical SDK usage
            ctrl_msg.action = 0 
            ctrl_msg.ue_id = ue_id
            ctrl_msg.new_weight = new_weight
            ric.control_mac_sm(self.node_id, ctrl_msg)
        elif decision.get("action_id") == "DO_NOTHING":
            pass

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
        
        hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_ms_10, cb)
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