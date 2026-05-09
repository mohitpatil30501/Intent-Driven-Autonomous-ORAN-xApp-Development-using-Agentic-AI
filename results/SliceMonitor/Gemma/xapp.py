import time
import random
from typing import List, Dict, Any

# --- Configuration ---
# Simulate the data rate (10ms interval)
SAMPLING_INTERVAL_MS = 10
# Simulation limit for testing the logging loop
MAX_MONITOR_CYCLES = 50

class SliceDetailMonitor:
    """
    A subscription-based xApp designed to monitor and log current details
    for all active network slices received from the RIC.
   
    Since this is read-only monitoring, no actions are performed.
    """
    def __init__(self):
        print("===================================================")
        print("🚀 SliceDetailMonitor Initializing...")
        print("Listening for slice detail data payload...")
        print("===================================================")
        self.cycle_count = 0
        self.running = True

    def _generate_mock_slice_data(self) -> List[Dict[str, Any]]:
        """
        MOCK FUNCTION: Simulates the payload received from the RIC every 10ms.
        In a real xApp, this data would be provided by the platform API.
        """
        slice_types = ["eMBB", "uRLLC", "mMTC"]
        slices = []
       
        # Simulate 2 to 4 active slices
        num_slices = random.randint(2, 4)
       
        for i in range(num_slices):
            slice_id = f"SLICE_{random.randint(1000, 9999)}"
            slice_name = f"Service_A_{i+1}"
            slice_type = random.choice(slice_types)
            active_ues = random.randint(10, 500)
           
            slice_details = {
                "slice_id": slice_id,
                "slice_name": slice_name,
                "slice_type": slice_type,
                "active_ue_count": active_ues,
                "status": "ACTIVE",
                "is_priority": random.choice([True, False])
            }
            slices.append(slice_details)
       
        return slices

    def on_data_received(self, payload: Any) -> None:
        """
        CORE XAPP LOGIC: This method is called by the RIC platform
        whenever new data is available (every 10ms in this scenario).
       
        :param payload: Expected to be a list of dictionaries,
                         where each dictionary represents a slice.
        """
        self.cycle_count += 1
        current_time = time.strftime("%H:%M:%S")
       
        # Use mock data if the payload is None (for testing)
        if payload is None:
            slices_data = self._generate_mock_slice_data()
            print(f"\n--- MOCK DATA RECEIVED @ {current_time} (Cycle {self.cycle_count}) ---")
        else:
            slices_data = payload
            print(f"\n--- LIVE DATA RECEIVED @ {current_time} (Cycle {self.cycle_count}) ---")

        if not slices_data:
            print("⚠️ WARNING: No slice data received in this cycle.")
            return

        print(f"✅ Successfully monitored {len(slices_data)} active slices.")
       
        print("\n" + "="*70)
        print(f"| {'Slice ID':<15} | {'Slice Name':<20} | {'Type':<8} | {'UE Count':<8} |")
        print("="*70)

        # Iterate and log details for each slice
        for slice in slices_data:
            slice_id = slice.get("slice_id", "N/A")
            slice_name = slice.get("slice_name", "N/A")
            slice_type = slice.get("slice_type", "N/A")
            ue_count = slice.get("active_ue_count", 0)
           
            # Format the output row
            print(f"| {slice_id:<15} | {slice_name:<20} | {slice_type:<8} | {ue_count:<8} |")
       
        print("="*70)
        print("Monitoring complete for this cycle. Waiting for next data packet...")

    def run_simulation(self):
        """
        Simulates the continuous 10ms monitoring loop for demonstration purposes.
        In a real xApp, the platform handles the timing and calling of on_data_received.
        """
        try:
            for i in range(MAX_MONITOR_CYCLES):
                if not self.running:
                    break
               
                # Simulate data reception
                self.on_data_received(payload=None)
               
                # Wait for the simulated 10ms interval
                time.sleep(SAMPLING_INTERVAL_MS / 1000.0)
               
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped manually by user (Ctrl+C).")
        finally:
            print("\nMonitor shut down gracefully.")

if __name__ == "__main__":
    monitor = SliceDetailMonitor()
    # Run the simulation loop to demonstrate the continuous monitoring process
    monitor.run_simulation()

