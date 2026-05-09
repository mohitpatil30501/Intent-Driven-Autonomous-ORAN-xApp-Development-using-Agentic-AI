import time
import random
from typing import Dict, Any, List

# --- Configuration Constants ---
THROTTLING_THRESHOLD = 0.85  # 85% utilization triggers throttling
INITIAL_PRB_ALLOCATION = 20  # Initial PRB allocation per UE (out of a larger capacity)
MAX_SYSTEM_CAPACITY = 100    # Total system PRBs available
SAMPLING_INTERVAL_SEC = 1    # How often the xApp runs its cycle

class ResourceMonitor:
    """
    Simulates the interface to the RAN control plane (e.g., N2/N3/x2 interface).
    In a real xApp, this would consume streaming data (e.g., Kafka or gRPC streams).
    """
    def __init__(self, total_capacity: int):
        self.total_capacity = total_capacity
        print(f"[Monitor] Initialized. Total System PRB Capacity: {total_capacity}")

    def get_current_load_data(self) -> List[Dict[str, Any]]:
        """
        Mocks the retrieval of usage data for active UEs.
        In a real environment, this would query performance counters.
        """
        # Simulate a changing load scenario
        u_e_ids = [f"UE_{i:03d}" for i in range(3)] # Monitor 3 UEs
        
        load_data = []
        for ue_id in u_e_ids:
            # Simulate usage: UE_000 is highly loaded, UE_001 is moderate, UE_002 is low.
            if ue_id == "UE_000":
                prb_used = random.randint(16, 20) # High Usage
                throughput = random.uniform(20, 35) # High Throughput
            elif ue_id == "UE_001":
                prb_used = random.randint(8, 12)
                throughput = random.uniform(10, 18)
            else:
                prb_used = random.randint(1, 5) # Low Usage
                throughput = random.uniform(1, 5)
            
            load_data.append({
                'ue_id': ue_id,
                'prb_used': prb_used,
                'throughput': throughput
            })
        return load_data


class ResourceAllocator:
    """
    Mocks the crucial interaction with the MAC/Scheduling plane.
    This API call is what *enforces* the resource change.
    """
    @staticmethod
    def set_ue_priority(ue_id: str, new_ratio: float, new_prb_limit: int) -> bool:
        """
        Simulates calling the scheduler function to adjust resource grants.
        """
        print("="*50)
        print(f"[ALLOCATOR] *** SCHEDULING ADJUSTMENT FOR {ue_id} ***")
        print(f"[ALLOCATOR] Action: Priority reduced.")
        print(f"[ALLOCATOR] New PRB Allocation Ratio: {new_ratio*100:.2f}%")
        print(f"[ALLOCATOR] Enforcing New PRB Limit: {new_prb_limit} PRBs.")
        print("="*50)
        return True

    @staticmethod
    def log_throttling_event(ue_id: str, old_util: float, new_util: float):
        """Logs the decision for auditing purposes."""
        print(f"🚨 THROTTLING EVENT: {ue_id} (Utilization: {old_util:.2f} -> {new_util:.2f})")


class SchedulerController:
    """
    The core intelligence (the xApp). It implements the monitoring, decision-making,
    and enforcement logic based on the MAC service model requirements.
    """
    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor
        # State dictionary to track current allocations: {UE_ID: {'prb_limit': X, 'priority_ratio': Y}}
        self.ue_state: Dict[str, Dict[str, float]] = {}
        
        print("\n[Controller] Initializing MAC Scheduling Policy Engine.")

    def initialize_ue_state(self, ue_id: str):
        """Sets the default, initial allocation state for a new UE."""
        if ue_id not in self.ue_state:
            self.ue_state[ue_id] = {
                'prb_limit': INITIAL_PRB_ALLOCATION,
                'priority_ratio': 1.0 # 100% of initial allocation
            }
            print(f"[Controller] Initialized default state for {ue_id}: {INITIAL_PRB_ALLOCATION} PRBs.")

    def calculate_utilization(self, ue_id: str, prb_used: int, allocated_limit: int) -> float:
        """Calculates current utilization percentage."""
        if allocated_limit == 0:
            return 0.0
        return prb_used / allocated_limit

    def check_and_throttle(self, load_data: List[Dict[str, Any]]):
        """
        The main decision loop. Iterates through UEs and determines if throttling is required.
        """
        print("\n" + "-"*70)
        print("[Controller] Running scheduling policy check...")
        
        for data_point in load_data:
            ue_id = data_point['ue_id']
            prb_used = data_point['prb_used']
            throughput = data_point['throughput']
            
            self.initialize_ue_state(ue_id)
            
            # Retrieve the current state/allocation
            state = self.ue_state[ue_id]
            current_limit = state['prb_limit']
            
            # 1. Monitor and Calculate Utilization
            current_util = self.calculate_utilization(ue_id, prb_used, current_limit)
            
            print(f"\n[Monitor] {ue_id}: Usage={prb_used} PRBs, Util={current_util*100:.2f}%, Throughput={throughput:.2f} Mbps")
            
            # 2. Decision Logic (Check against threshold)
            if current_util > THROTTLING_THRESHOLD:
                print(f"⚠️ [Decision] {ue_id} exceeded threshold ({THROTTLING_THRESHOLD*100:.0f}%). Action needed.")
                
                # 3. Calculate Throttling Parameters
                # New ratio: Reduce the current ratio by a penalty factor (e.g., 10%)
                new_ratio = state['priority_ratio'] * 0.90 
                
                # New PRB limit: Apply the ratio to the initial allocation
                new_limit = int(INITIAL_PRB_ALLOCATION * new_ratio)
                
                # Ensure limit doesn't drop below a safety floor (e.g., 1 PRB)
                new_limit = max(1, new_limit) 
                
                # 4. Enforcement (Call the API Mock)
                ResourceAllocator.set_ue_priority(
                    ue_id, 
                    new_ratio, 
                    new_limit
                )
                ResourceAllocator.log_throttling_event(ue_id, current_util, current_util)
                
                # 5. Update Internal State
                self.ue_state[ue_id]['prb_limit'] = new_limit
                self.ue_state[ue_id]['priority_ratio'] = new_ratio
                
            else:
                # If usage is stable, confirm current allocation is acceptable
                print(f"[Controller] {ue_id} utilization is acceptable ({current_util*100:.2f}%). Maintaining resource allocation.")


# =================================================
# --- Main Execution Function ---
# =================================================

def run_xapp_simulation():
    """
    Runs the simulation loop to demonstrate the continuous monitoring process.
    """
    print("==============================================================")
    print(" STARTING MAC SERVICE MODEL XAPP SIMULATION (PRB Throttling)")
    print("==============================================================")

    # Initialize Components
    monitor = ResourceMonitor(total_capacity=MAX_SYSTEM_CAPACITY)
    controller = SchedulerController(monitor=monitor)
    
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            print(f"\n\n==============================================================")
            print(f"                SIMULATION CYCLE {cycle_count}")
            print(f"==============================================================")

            # 1. Data Ingestion
            load_data = monitor.get_current_load_data()
            
            # 2. Decision Making and Enforcement
            controller.check_and_throttle(load_data)
            
            print("\n--------------------------------------------------------------")
            print("SIMULATION PAUSED. Waiting for next sample...")
            time.sleep(SAMPLING_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n[Controller] Simulation stopped by user (Ctrl+C). Shutting down.")
    except Exception as e:
        print(f"\n[Controller] An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_xapp_simulation()

