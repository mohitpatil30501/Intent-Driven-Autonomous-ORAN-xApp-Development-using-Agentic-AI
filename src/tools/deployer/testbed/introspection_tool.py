import os
import sys
import time
from langchain_core.tools import tool

# Add parent dir to path to import testbed_tool
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from testbed_tool import deploy_xapp_to_testbed

@tool
def inspect_service_model_runtime(service_model: str, max_depth: int = 2) -> str:
    """
    Deploys a probe xApp to the testbed to inspect the runtime data structure 
    of a specific Service Model (SM). Use this to see exactly what fields 
    are available in the 'ind' (indication) object for MAC, KPM, or RLC.
    
    Args:
        service_model (str): The SM to inspect (e.g., 'mac', 'kpm', 'rlc').
        max_depth (int): How deep to recurse into the object attributes.
        
    Returns:
        str: A tree-like representation of the SM indication attributes.
    """
    sm = service_model.lower()
    
    # Mapping SM names to FlexRIC class names and report functions
    sm_config = {
        "mac": {
            "callback_class": "mac_cb",
            "report_fn": "report_mac_sm",
            "interval": "Interval_ms_10"
        },
        "kpm": {
            "callback_class": "kpm_cb",
            "report_fn": "report_kpm_sm",
            "interval": "Interval_ms_10"
        },
        "rlc": {
            "callback_class": "rlc_cb",
            "report_fn": "report_rlc_sm",
            "interval": "Interval_ms_10"
        },
        "slice": {
            "callback_class": "slice_cb",
            "report_fn": "report_slice_sm",
            "interval": "Interval_ms_10"
        }
    }
    
    if sm not in sm_config:
        return f"Error: Unsupported Service Model '{service_model}'. Supported: {list(sm_config.keys())}"
    
    cfg = sm_config[sm]
    
    probe_code = f"""
import xapp_sdk as ric
import time
import sys
import json

def swig_to_dict(obj, max_depth=5, _depth=0):
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

    result = {{}}

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
            result[attr] = f"<error: {{e}}>"

    return result

class SchemaAwareCallback(ric.{cfg['callback_class']}):
    def __init__(self, max_depth={max_depth}):
        ric.{cfg['callback_class']}.__init__(self)
        self.max_depth = max_depth
        self.inspected = False

    def handle(self, ind):
        if not self.inspected:
            print("[AGENT_INSPECTION_START]")
            print(swig_to_dict(ind, max_depth=self.max_depth))
            print("[AGENT_INSPECTION_END]")
            self.inspected = True
            ric.try_stop = 1

def main():
    ric.init()
    conn = ric.conn_e2_nodes()
    if len(conn) == 0:
        print("Error: No E2 Nodes connected.")
        return

    print(f"Subscribing to E2 Node: {{conn[0].id}} for introspection...")
    cb = SchemaAwareCallback(max_depth={max_depth})
    hndlr = ric.{cfg['report_fn']}(conn[0].id, ric.{cfg['interval']}, cb)

    time.sleep(10)

    # Cleanup subscription
    print("Cleaning up subscription...")
    ric.{cfg['report_fn'].replace('report_', 'rm_report_')}(hndlr)

if __name__ == '__main__':
    main()
"""
    
    try:
        # We call the function logic with a custom timeout of 55s to allow for a small buffer
        # before the 60s hard timeout of the tool itself.
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(deploy_xapp_to_testbed.func, probe_code, timeout=120)
            try:
                logs = future.result(timeout=120)
            except concurrent.futures.TimeoutError:
                return "Error: Introspection tool timed out after 60 seconds. The testbed might be slow or unresponsive. Please try again or use semantic search for static mapping."

        if "[AGENT_INSPECTION_START]" in logs:
            parts = logs.split("[AGENT_INSPECTION_START]")
            return parts[1].split("[AGENT_INSPECTION_END]")[0].strip()
        
        return f"Probe execution log summary:\\n{logs[:500]}...\\n\\nError: Could not find inspection markers. The xApp might have crashed or timed out."
    except Exception as e:
        return f"Error during introspection: {str(e)}"
