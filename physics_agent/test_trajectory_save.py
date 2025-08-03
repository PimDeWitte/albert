#!/usr/bin/env python3
"""Quick test to check if trajectories are saved."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_comprehensive_final import test_trajectory_vs_kerr, save_particle_trajectories_to_run
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.theories.defaults.baselines.kerr import Kerr
from physics_agent.theory_engine_core import TheoryEngine

def main():
    """Test trajectory saving with just 2 theories."""
    run_dir = "test_trajectory_save_run"
    os.makedirs(run_dir, exist_ok=True)
    
    # Create engine
    engine = TheoryEngine(verbose=False)
    
    theories = [
        ("Schwarzschild", Schwarzschild(), "baseline"),
        ("Kerr", Kerr(), "classical"),
    ]
    
    results = []
    
    for name, theory, category in theories:
        print(f"\nTesting {name}...")
        theory.category = category
        
        # Run trajectory test
        test_result = test_trajectory_vs_kerr(theory, engine, n_steps=100)  # Use fewer steps for speed
        
        # Create result structure
        result = {
            'theory': name,
            'category': category,
            'solver_tests': [test_result]
        }
        results.append(result)
        
        print(f"  Test result: {test_result['status']}")
        if 'particle_results' in test_result:
            print(f"  Particle results: {list(test_result['particle_results'].keys())}")
        else:
            print(f"  No particle_results in test result. Keys: {list(test_result.keys())}")
            if 'error' in test_result:
                print(f"  Error: {test_result['error']}")
    
    # Save trajectories
    print("\nSaving trajectories...")
    save_particle_trajectories_to_run(run_dir, results)
    
    # Check what was saved
    traj_dir = os.path.join(run_dir, 'particle_trajectories')
    if os.path.exists(traj_dir):
        files = os.listdir(traj_dir)
        print(f"\nSaved files: {files}")
    else:
        print("\nNo trajectory directory created!")

if __name__ == "__main__":
    main()