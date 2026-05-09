from ricxappframe.xapp_frame import Xapp
import json

def entry(self):
    """The main execution loop for the SliceDetailMonitor"""
    self.logger.info("SliceDetailMonitor started. Listening for 10ms updates...")

def slice_data_handler(self, summary, sbuf):
    """
    Callback function triggered when a message is received from the RIC.
    """
    # 1. Parse the incoming payload (usually ASN.1 or JSON depending on RIC version)
    try:
        payload = json.loads(summary['payload'])
        
        # 2. Extract specific slice details
        # Note: The keys depend on your specific E2 Service Model (E2SM-KPM or NI)
        slice_id = payload.get("slice_id", "N/A")
        slice_name = payload.get("slice_name", "Unknown")
        active_ues = payload.get("active_ue_count", 0)
        slice_type = payload.get("slice_type", "eMBB") # e.g., eMBB, URLLC, mMTC

        # 3. Simple Logging (Overwrites or appends current state)
        print(f"--- Slice Update Received ---")
        print(f"ID: {slice_id} | Name: {slice_name} | Type: {slice_type} | Active UEs: {active_ues}")
        
    except Exception as e:
        self.logger.error(f"Error decoding slice data: {e}")
    finally:
        # Free the RMR buffer
        self.rmr_free(sbuf)

# Initialize the xApp
# rmr_port: the port for receiving 10ms indications
# post_init: the function to run once the xApp starts
slice_monitor = Xapp(entry_function=entry, 
                     rmr_port=4560, 
                     post_init=None)

# Register the handler for the specific message type (e.g., Message Type 12345)
slice_monitor.register_callback(slice_data_handler, 12345)

slice_monitor.run()

