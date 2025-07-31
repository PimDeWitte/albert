#!/usr/bin/env python3
"""
Debug script to trace why quantum calculations aren't happening in UnifiedTrajectoryCalculator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theory_loader import TheoryLoader
from physics_agent.unified_trajectory_calculator import UnifiedTrajectoryCalculator
import torch


def trace_quantum_calculation():
    """Trace through the quantum calculation flow."""
    
    print("=== DEBUGGING QUANTUM CALCULATIONS ===")
    print()
    
    # Load a quantum theory
    loader = TheoryLoader('physics_agent/theories')
    theories = loader.discover_theories()
    
    # Find Alena Tensor theory
    alena_key = None
    for key, info in theories.items():
        if 'alena' in key.lower():
            alena_key = key
            break
    
    if not alena_key:
        print("ERROR: Could not find Alena Tensor theory!")
        return
    
    # Instantiate the theory
    theory = loader.instantiate_theory(alena_key)
    print(f"Theory: {theory.name}")
    print(f"Category: {theory.category}")
    print(f"enable_quantum: {theory.enable_quantum}")
    print()
    
    # Create UnifiedTrajectoryCalculator
    calculator = UnifiedTrajectoryCalculator(
        theory,
        enable_quantum=True,
        enable_classical=True
    )
    
    print(f"Calculator initialized:")
    print(f"  calculator.enable_quantum: {calculator.enable_quantum}")
    print(f"  calculator.enable_classical: {calculator.enable_classical}")
    print(f"  calculator.quantum_integrator: {calculator.quantum_integrator}")
    print()
    
    # Set up initial conditions
    initial_conditions = {
        'r': 12.0,  # In geometric units
        'E': 0.96,
        'Lz': 4.0,
        't': 0.0,
        'phi': 0.0
    }
    
    # Run unified trajectory with debug
    print("Running compute_unified_trajectory...")
    
    # Monkey-patch to add debug output
    original_compute = calculator.compute_unified_trajectory
    
    def debug_compute(initial_conditions, final_state=None, **kwargs):
        print(f"\n[DEBUG] compute_unified_trajectory called:")
        print(f"  final_state passed in: {final_state}")
        print(f"  enable_quantum: {calculator.enable_quantum}")
        
        # Call original
        results = original_compute(initial_conditions, final_state, **kwargs)
        
        print(f"\n[DEBUG] Results keys: {list(results.keys())}")
        
        if 'classical' in results:
            classical = results['classical']
            print(f"  Classical trajectory points: {classical.get('time_steps', 0)}")
            if 'trajectory' in classical:
                traj = classical['trajectory']
                print(f"  Classical trajectory shape: {traj.shape if hasattr(traj, 'shape') else len(traj)}")
        
        if 'quantum' in results:
            print("  ✓ QUANTUM CALCULATIONS PERFORMED!")
            quantum = results['quantum']
            print(f"  Quantum keys: {list(quantum.keys())}")
        else:
            print("  ✗ NO QUANTUM CALCULATIONS!")
        
        return results
    
    calculator.compute_unified_trajectory = debug_compute
    
    # Run with small number of steps for quick test
    results = calculator.compute_unified_trajectory(
        initial_conditions,
        time_steps=10,
        step_size=0.01,
        quantum_method='monte_carlo',
        quantum_samples=10
    )
    
    print("\n=== ANALYSIS ===")
    
    # Check what happened
    if 'quantum' not in results:
        print("PROBLEM: Quantum calculations were skipped!")
        print("\nPossible reasons:")
        print("1. enable_quantum is False")
        print(f"   - calculator.enable_quantum = {calculator.enable_quantum}")
        print(f"   - theory.enable_quantum = {theory.enable_quantum}")
        print("2. final_state is None")
        print("   - This requires classical trajectory to complete first")
        print("3. quantum_integrator is None")
        print(f"   - calculator.quantum_integrator = {calculator.quantum_integrator}")
        
        # Check if theory has quantum integrator
        if hasattr(theory, '_quantum_integrator'):
            print(f"   - theory._quantum_integrator = {theory._quantum_integrator}")


if __name__ == "__main__":
    trace_quantum_calculation()