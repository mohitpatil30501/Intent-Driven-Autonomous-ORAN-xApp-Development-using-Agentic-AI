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
        if ind.kpi_indication_message:
            first_ue_stat = ind.kpi_indication_message[0]
            row_dict['ue_id'] = first_ue_stat.ue_id
            row_dict['downlink_throughput'] = first_ue_stat.downlink_throughput
            row_dict['uplink_throughput'] = first_ue_stat.uplink_throughput
        
        # --- 2. EXECUTE CORE LOGIC ---
        decision = self.app_logic.process_interval(row_dict)

        # --- 3. EXECUTE CONTROL ACTIONS ---
        if decision.get("action_id") == "UPDATE_PRB_ALLOCATION":
            # Parameters expected: ue_id, prb_allocation_adjustment, reason_code
            if "parameters" in decision:
                params = decision["parameters"]
                ue_id = params.get("ue_id")
                prb_adj = params.get("prb_allocation_adjustment")
                reason = params.get("reason_code")
                
                ctrl_msg = ric.mac_cb_ctrl_msg_t()
                ctrl_msg.action = 1 # Assuming UPDATE_PRB_ALLOCATION maps to action ID 1
                ctrl_msg.ue_id = ue_id
                ctrl_msg.dl_prb_ratio = prb_adj
                ctrl_msg.reason_code = reason
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