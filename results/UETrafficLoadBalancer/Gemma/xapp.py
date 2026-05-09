

import time
import random
from typing import Dict, List, Tuple

# --- Configuration Constants ---
MAX_UE_COUNT_THRESHOLD = 50 
MAX_UTILIZATION_THRESHOLD = 0.80 # 80%
INITIAL_SLOT_CAPACITY = 1000 # Base resource capacity unit for the whole service

class SliceMonitor:
    """
    Simulates the interface to the Network Slice Management (NSM) plane.
    It provides aggregated, real-time metrics for all active slices.
    """
    def __init__(self, slice_ids: List[str], total_capacity: int):
        self.slice_ids = slice_ids
        self.total_capacity = total_capacity
        self.current_state: Dict[str, Dict[str, float]] = {}
        print("\n[Monitor] Initialized. Monitoring " + ", ".join(slice_ids) + " slices.")

    def initialize_state(self, slice_id: str, initial_uses: float = 0.0, initial_u_count: int = 10):
        """Sets the baseline state for a slice."""
        self.current_state[slice_id] = {
            'ue_count': initial_u_count,
            'current_load_units': initial_uses,
            'max_capacity': INITIAL_SLOT_CAPACITY,
            'utilization': 0.0,
            'priority_weight': 1.0 # Starts at 100% priority
        }

    def get_current_load_snapshot(self, slice_id: str, load_increment: float = 0.0, ue_count_delta: int = 0) -> None:
        """
        Mocks the reception of a new load snapshot for a specific slice.
        This function updates the internal state based on real-time telemetry.
        """
        if slice_id not in self.current_state:
            print(f"Warning: Slice {slice_id} not initialized.")
            return

        state = self.current_state[slice_id]
        
        # Update metrics based on the incoming increment
        state['current_load_units'] = state['current_load_units'] + load_increment
        state['ue_count'] = state['ue_count'] + ue_count_delta
        
        # Recalculate utilization
        state['utilization'] = state['current_load_units'] / state['max_capacity']
        
        print(f"\n--- Monitoring {slice_id} ---")
        print(f"  UE Count: {state['ue_count']} / {MAX_UE_COUNT_THRESHOLD}")
        print(f"  Utilization: {state['utilization']*100:.2f}% (Load: {state['current_load_units']:.0f} units)")

class PolicyEngine:
    """
    The core xApp logic. Implements the threshold-based load balancing policy.
    """
    def __init__(self, monitor: SliceMonitor):
        self.monitor = monitor
        self.load_history: Dict[str, float] = {} # Tracks current recommended weights

    def check_and_rebalance_slices(self, snapshot_data: Dict[str, dict]) -> None:
        """
        Iterates through all slices, checking against the defined thresholds 
        and triggering rebalancing if necessary.
        """
        print("\n" + "="*70)
        print("⚡️ [POLICY ENGINE] Running Slice Load Balancing Policy Check...")
        print("="*70)
        
        # 1. First Pass: Check all slices for congestion
        overloaded_slices: List[str] = []
        for slice_id, data in snapshot_data.items():
            util = data['utilization']
            ue_count = data['ue_count']
            
            is_overloaded = (util > MAX_UTILIZATION_THRESHOLD) or (ue_count > MAX_UE_COUNT_THRESHOLD)
            
            if is_overloaded:
                overloaded_slices.append((slice_id, data))
                print(f"🚨 ALERT: {slice_id} is Overloaded! (U: {util*100:.1f}%, C: {ue_count})")
            else:
                print(f"✅ {slice_id} Status OK. (U: {util*100:.1f}%, C: {ue_count})")

        # 2. Second Pass: Act only if congestion is detected
        if not overloaded_slices:
            print("\n[Policy] No slices exceeded critical thresholds. No rebalancing action needed.")
            return

        print("\n[Policy] Initiating Resource Rebalancing and Traffic Steering...")
        
        # 3. Rebalancing Logic
        for overloaded_id, overloaded_data in overloaded_slices:
            
            total_overload_units = 0.0
            overload_reasons = []

            if overloaded_data['utilization'] > MAX_UTILIZATION_THRESHOLD:
                # Calculate load units that need shedding due to utilization
                excess_util = overloaded_data['utilization'] - MAX_UTILIZATION_THRESHOLD
                excess_units = excess_util * overloaded_data['max_capacity']
                total_overload_units += excess_units
                overload_reasons.append(f"Utilization ({excess_util*100:.1f}% excess)")
            
            if overloaded_data['ue_count'] > MAX_UE_COUNT_THRESHOLD:
                # Calculate resource "cost" based on excessive user count
                excess_users = overloaded_data['ue_count'] - MAX_UE_COUNT_THRESHOLD
                # Estimate load based on average load per UE
                total_overload_units += (excess_users * 5) # Assuming 5 units per user excess
                overload_reasons.append(f"User Count ({excess_users} excess)")


            # 4. Distribute Excess Load to Healthy Slices
            remaining_slices = [sid for sid in self.monitor.slice_ids if sid != overloaded_id]
            
            if not remaining_slices:
                print(f"WARNING: {overloaded_id} is overloaded and no other slices are available for rebalancing.")
                continue

            # Proportional distribution: Send the excess load to the healthiest slices.
            load_to_distribute = total_overload_units
            remaining_slots = sum(self.monitor.current_state[sid]['max_capacity'] for sid in remaining_slices)
            
            print(f"   -> Total excess load identified: {load_to_distribute:.1f} units.")

            # Target a balanced state: distribute the excess load proportionally
            for target_id in remaining_slices:
                target_state = self.monitor.current_state[target_id]
                
                # Calculate the target weight boost for this slice
                # Weight boost is proportional to its remaining capacity relative to the total remaining capacity.
                boost_ratio = target_state['max_capacity'] / remaining_slots
                
                # Calculate the target new weight: minimum of (1.2, 1.0 + boost_ratio)
                # We use min(1.2, ...) to cap the boost, preventing excessive over-prioritization.
                new_weight = min(1.2, 1.0 + (1.0 * boost_ratio * 0.2))
                
                # Signal the change
                self._signal_control_signal(overloaded_id, target_id, new_weight)


    def _signal_control_signal(self, source_slice: str, target_slice: str, weight: float):
        """
        Mocks sending the control plane signal to the Slice Service Model (SSM).
        This is where the actual resource re-allocation would occur.
        """
        print("\n" + "!"*70)
        print(f"✨ [SSM SIGNAL] Policy Triggered: Load Balancing between {source_slice} and {target_slice}.")
        print(f"  -> Action: Increasing {target_slice}'s scheduling weight to {weight*100:.1f}% (Boost).")
        print(f"  -> Impact: This shifts traffic steering preference toward {target_slice} to alleviate load on {source_slice}.")
        print("!"*70)


# =========================================================================
# MAIN EXECUTION
# =======================================================================

def run_xapp_simulation():
    """
    Runs the simulation through multiple cycles to demonstrate the load balancing.
    """
    print("==================================================================")
    print("   🚀  SLICE SERVICE MODEL XAPP SIMULATION (Load Balancer) 🚀   ")
    print("==================================================================")

    # 1. Setup Monitoring State
    slice_ids = ["Consumer_A", "Enterprise_B", "IoT_C"]
    monitor = SliceMonitor(slice_ids, total_capacity=INITIAL_SLOT_CAPACITY)

    # Initialize default state for all slices
    for sid in slice_ids:
        monitor.initialize_state(sid)
    
    policy_engine = PolicyEngine(monitor)
    
    # --- SCENARIO 1: Normal Operation (No Overload) ---
    print("\n\n################# CYCLE 1: NORMAL OPERATION ################")
    # Simulate minor, normal load increases
    for sid in slice_ids:
        monitor.get_current_load_snapshot(sid, load_increment=random.uniform(50, 100), ue_count_delta=random.randint(1, 5))
    
    policy_engine.check_and_rebalance_slices(monitor.current_state)

    # --- SCENARIO 2: Overload Detection (Consumer_A is overloaded) ---
    print("\n\n\n################# CYCLE 2: OVERLOAD DETECTION ################")
    # Manually force one slice to exceed both thresholds
    monitor.current_state["Consumer_A"]['current_load_units'] = INITIAL_SLOT_CAPACITY * 0.95 
    monitor.current_state["Consumer_A"]['ue_count'] = 60 # Exceeds 50
    
    # Resetting other slices slightly to make the difference clear
    monitor.current_state["Enterprise_B"]['current_load_units'] = INITIAL_SLOT_CAPACITY * 0.50
    monitor.current_state["Enterprise_B"]['ue_count'] = 30
    
    monitor.get_current_load_snapshot("Consumer_A", 0, 0) # Log the snapshot
    policy_engine.check_and_rebalance_slices(monitor.current_state)
    
    # --- SCENARIO 3: Subsequent Cycle (Load reduction is monitored) ---
    print("\n\n\n################# CYCLE 3: LOAD REDUCTION MONITORING ################")
    # Simulate the overloaded slice (Consumer_A) reducing its load slightly 
    monitor.current_state["Consumer_A"]['current_load_units'] *= 0.95 # 5% natural drop
    monitor.current_state["Consumer_A"]['ue_count'] -= 5
    
    # Simulate the healthy slices picking up the slack
    monitor.current_state["Enterprise_B"]['current_load_units'] += 50
    monitor.current_state["Enterprise_B"]['ue_count'] += 10
    
    monitor.get_current_load_snapshot("Consumer_A", 0, 0)
    policy_engine.check_and_rebalance_slices(monitor.current_state)


if __name__ == "__main__":
    run_xapp_simulation()

