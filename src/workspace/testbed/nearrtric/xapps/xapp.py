import sys
import time
import xapp_sdk as ric

# Dynamically import the core logic written by Module 5
from logic.core_logic import XAppLogic

def swig_to_dict(obj, max_depth=5, _depth=0):
    """
    Safely convert SWIG object (like FlexRIC 'ind') to Python dict
    """
    if _depth > max_depth:
        return "..."

    # Primitive types
    if isinstance(obj, (int, float, str, bool)):
        return obj

    if isinstance(obj, bytes):
        try:
            return obj.decode()
        except:
            return str(obj)

    # Python list/tuple
    if isinstance(obj, (list, tuple)):
        return [swig_to_dict(x, max_depth, _depth+1) for x in obj]

    result = {}

    for attr in dir(obj):
        # 🚫 skip garbage/internal SWIG stuff
        if attr.startswith("_") or attr in ("this", "thisown"):
            continue

        try:
            val = getattr(obj, attr)

            # Skip methods/functions
            if callable(val):
                continue

            # Handle arrays (SWIG often exposes via __len__ + __getitem__)
            if hasattr(val, "__len__") and hasattr(val, "__getitem__") and not isinstance(val, (str, bytes)):
                try:
                    result[attr] = [
                        swig_to_dict(val[i], max_depth, _depth+1)
                        for i in range(len(val))
                    ]
                    continue
                except:
                    pass

            # Nested struct
            result[attr] = swig_to_dict(val, max_depth, _depth+1)

        except Exception as e:
            result[attr] = f"<error: {e}>"

    return result

class UniversalCallback(ric.slice_cb):
    def __init__(self, node_id, app_logic):
        ric.slice_cb.__init__(self)
        self.node_id = node_id
        self.app_logic = app_logic

    def handle(self, ind):
        # --- 1. EXTRACT TELEMETRY ---
        # Map the FlexRIC C-struct (ind) to a flat Python dictionary
        row_dict = {}
        # row_dict['slice_id'] = ind.msg.slice_conf.dl.len_slices
        # row_dict['slice_name'] = ind.slice_name
        # row_dict['active_ue_count'] = ind.ue_statistics.active_ue_count
        # row_dict['slice_type'] = ind.slice_type_details
        # row_dict['metadata'] = ind.metadata
        
        # --- 2. EXECUTE CORE LOGIC ---
        # decision = self.app_logic.process_interval(row_dict)
        converted_dict = swig_to_dict(ind, 10)
        print(converted_dict.keys())

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

    time.sleep(5)

    # Keep alive loop
    # while ric.try_stop == 0:
    #     time.sleep(1)

    # Cleanup
    print("Shutting down...")
    for h in handlers:
        ric.rm_report_slice_sm(h)

if __name__ == '__main__':
    main()