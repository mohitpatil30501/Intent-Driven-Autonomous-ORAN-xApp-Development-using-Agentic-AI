import sys
import time
import xapp_sdk as ric

# Dynamically import the core logic written by Module 5
from logic.core_logic import XAppLogic

class UniversalCallback(ric.slice_cb):
    def __init__(self, node_id, app_logic):
        ric.slice_cb.__init__(self)
        self.node_id = node_id
        self.app_logic = app_logic

    def handle(self, ind):
        # --- 1. EXTRACT TELEMETRY ---
        # Map the FlexRIC C-struct (ind) to a flat Python dictionary
        row_dict = {}
        row_dict['slice_id'] = ind.slice_id
        row_dict['slice_name'] = ind.slice_name
        row_dict['active_ue_count'] = ind.ue_statistics.active_ue_count
        row_dict['slice_type'] = ind.slice_type_details
        row_dict['metadata'] = ind.metadata
        
        # --- 2. EXECUTE CORE LOGIC ---
        decision = self.app_logic.process_interval(row_dict)

        # --- 3. EXECUTE CONTROL ACTIONS ---
        # No control actions defined in the Action Space Menu (only DO_NOTHING)
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
        
        hndlr = ric.report_slice_sm(conn[i].id, ric.Interval_ms_10, cb)
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