#!/usr/bin/env python3
"""Test Yukawa trajectory visualization to ensure gradient warnings are fixed."""

import sys
sys.path.insert(0, '.')

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.yukawa.theory import Yukawa
from physics_agent.generate_theory_trajectory_plots import create_trajectory_plots
import os

def main():
    print("Testing Yukawa trajectory visualization...\n")
    
    # Initialize engine
    engine = TheoryEngine(black_hole_preset='primordial_mini', verbose=False)
    
    # Initialize Yukawa theory
    theory = Yukawa()
    
    # Parameters
    r0_si = 10 * engine.length_scale
    n_steps = 2000
    dtau_si = engine.bh_preset.integration_parameters['dtau_geometric'] * engine.time_scale
    
    print(f"Running trajectory for {theory.name}...")
    print(f"  Steps: {n_steps}")
    print(f"  Initial radius: 10M")
    
    # Run trajectory
    hist, solver_tag, _ = engine.run_trajectory(
        theory, r0_si, n_steps, dtau_si,
        verbose=False
    )
    
    if hist is not None:
        print(f"  Trajectory computed successfully")
        print(f"  Solver: {solver_tag}")
        print(f"  Shape: {hist.shape}")
        
        # Create visualization
        output_dir = "test_yukawa_viz"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating visualization...")
        plot_path, orbit_path = create_trajectory_plots(
            hist, theory.name, engine, output_dir
        )
        
        print(f"\nVisualization complete!")
        print(f"  Full plot: {plot_path}")
        print(f"  Orbit plot: {orbit_path}")
        
        # Check if warnings appeared
        print("\nIf no numpy warnings appeared above, the gradient calculation is fixed!")
    else:
        print("ERROR: No trajectory returned!")

if __name__ == "__main__":
    main()