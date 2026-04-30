import sys
import time
import xapp_sdk as ric

# Dynamically import the core logic written by Module 5
from logic.core_logic import XAppLogic

class UniversalCallback({{ SM_CALLBACK_BASE }}):
    def __init__(self, node_id, app_logic):
        {{ SM_CALLBACK_BASE }}.__init__(self)
        self.node_id = node_id
        self.app_logic = app_logic

    def handle(self, ind):
        # --- 1. EXTRACT TELEMETRY ---
        # Map the FlexRIC C-struct (ind) to a flat Python dictionary
        row_dict = {}
        {{ TELEMETRY_MAPPING_CODE }}
        
        # --- 2. EXECUTE CORE LOGIC ---
        decision = self.app_logic.process_interval(row_dict)

        # --- 3. EXECUTE CONTROL ACTIONS ---
        # Translate the dictionary decision back to FlexRIC control structures
        {{ CONTROL_MAPPING_CODE }}

def main():
    print("Initializing FlexRIC xApp...")
    ric.init()
    
    conn = ric.conn_e2_nodes()
    if len(conn) == 0:
        print("Error: No E2 Nodes connected.")
        return

    # Instantiate the independent logic brain (which loads the ML model if present)
    app_logic = XAppLogic()
    handlers =[]

    # Subscribe to the Service Model for all connected nodes
    for i in range(len(conn)):
        print(f"Subscribing to E2 Node: {conn[i].id}")
        cb = UniversalCallback(conn[i].id, app_logic)
        
        hndlr = {{ SM_REPORT_FUNCTION }}(conn[i].id, {{ REPORT_INTERVAL }}, cb)
        handlers.append(hndlr)

    print("Subscription successful. Listening for indications...")

    # Keep alive loop
    while ric.try_stop == 0:
        time.sleep(1)

    # Cleanup
    print("Shutting down...")
    for h in handlers:
        {{ SM_RM_REPORT_FUNCTION }}(h)

if __name__ == '__main__':
    main()