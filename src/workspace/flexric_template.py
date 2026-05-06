import sys
import time
import xapp_sdk as ric

# --- INLINED CORE LOGIC ---
# {{ INLINED_LOGIC_CODE }}

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

class UniversalCallback({{ SM_CALLBACK_BASE }}):
    def __init__(self, node_id, app_logic):
        {{ SM_CALLBACK_BASE }}.__init__(self)
        self.node_id = node_id
        self.app_logic = app_logic

    def handle(self, ind):
        # --- 1. EXTRACT TELEMETRY ---
        # Map the FlexRIC C-struct (ind) to a flat Python dictionary
        row_dict = swig_to_dict(ind, 10)
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

    # Run the xApp for a duration (can be controlled via environment or signals)
    try:
        while ric.try_stop == 0:
            time.sleep(1)
    except KeyboardInterrupt:
        print("xApp stopped by user.")
    finally:
        print("Cleaning up subscriptions...")
        for h in handlers:
            ric.{{ SM_REMOVE_REPORT_FUNCTION }}(h)
        print("xApp terminated.")

if __name__ == '__main__':
    main()