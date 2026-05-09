import sys
import time
import xapp_sdk as ric

# Dynamically import the core logic written by Module 5
from logic.core_logic import XAppLogic

class UniversalCallback(ric.slice_cb):
    def __init__(self, node_id, app_logic):
        super().__init__()
        self.node_id = node_id
        self.app_logic = app_logic

    def handle(self, ind):
        # --- 1. EXTRACT TELEMETRY ---
        # Map the FlexRIC C-struct (ind) to a flat Python dictionary
        row_dict = {}
        if hasattr(ind, 'slice_stats_v0') and ind.slice_stats_v0 and hasattr(ind.slice_stats_v0, 'metrics'):
            metrics = ind.slice_stats_v0.metrics
            row_dict['ue_count_per_slice'] = metrics.get("ue_count_per_slice", 0.0)
            row_dict['utilization_percentage'] = metrics.get("utilization_percentage", 0.0)
        
        # --- 2. EXECUTE CORE LOGIC ---
        decision = self.app_logic.process_interval(row_dict)

        # --- 3. EXECUTE CONTROL ACTIONS ---
        # Translate the dictionary decision back to FlexRIC control structures
        if decision.get("action_id") == "UPDATE_LOAD_BALANCING_THRESHOLD":
            slice_id = int(decision["parameters"]["slice_id"])
            new_threshold = int(decision["parameters"]["new_threshold_percent"])
            rebalance = bool(decision["parameters"].get("rebalance_enabled", False))
            
            ctrl_msg = ric.slice_cb_ctrl_msg_t()
            ctrl_msg.slice_id = slice_id
            ctrl_msg.new_threshold_percent = new_threshold
            ctrl_msg.rebalance_enabled = rebalance
            ric.control_slice_sm(self.node_id, ctrl_msg)

        elif decision.get("action_id") == "FORCE_LOAD_REBALANCE":
            slice_id = int(decision["parameters"]["slice_id"])
            
            ctrl_msg = ric.slice_cb_ctrl_msg_t()
            ctrl_msg.slice_id = slice_id
            ctrl_msg.force_rebalance = True
            ric.control_slice_sm(self.node_id, ctrl_msg)

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
        
        hndlr = ric.report_slice_sm(conn[i].id, ric.Interval_ms_500, cb)
        handlers.append(hndlr)

    print("Subscription successful. Listening for indications...")

    # Keep alive loop
    while ric.try_stop == 0:
        time.sleep(1)

    # Cleanup
    print("Shutting down...")
    for h in handlers:
        ric.rm_report_slice_sm(h)

if __name__ == '__main__':
    main()