#!/usr/bin/env python3
"""Simple debug script to check trajectory initialization without cache."""

import torch
import numpy as np

# Add the project root to Python path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild

def main():
    print("=== Simple Trajectory Debug ===\n")
    
    # Create engine with verbose output
    engine = TheoryEngine(black_hole_preset='primordial_mini', verbose=True)
    
    print(f"Black hole parameters:")
    print(f"  Mass: {engine.M_si:.3e} kg")
    print(f"  Schwarzschild radius: {engine.length_scale:.3e} m")
    print(f"  Time scale: {engine.time_scale:.3e} s\n")
    
    # Test with Schwarzschild
    theory = Schwarzschild()
    
    # Set initial radius at 10 Schwarzschild radii
    r0_si = 10 * engine.length_scale
    print(f"Initial conditions:")
    print(f"  r0 = 10 Rs = {r0_si:.3e} m")
    print(f"  This is {r0_si / engine.length_scale:.1f} Schwarzschild radii\n")
    
    # Run a very short trajectory without cache
    n_steps = 5
    dtau_si = 1e-6  # Small timestep in seconds
    
    print(f"Running {n_steps}-step trajectory with dtau = {dtau_si:.3e} s...")
    print("(no_cache=True to force recalculation)\n")
    
    hist, solver_tag, _ = engine.run_trajectory(
        theory, r0_si, n_steps, dtau_si, 
        no_cache=True,  # Force fresh calculation
        verbose=True
    )
    
    if hist is not None:
        print(f"\nTrajectory computed with solver: {solver_tag}")
        print(f"Shape: {hist.shape}")
        print("\nTrajectory points (t[s], r[m], theta[rad], phi[rad]):")
        
        for i in range(len(hist)):
            t, r, theta, phi = hist[i]
            print(f"  Step {i}: t={t:.3e}, r={r:.3e} ({r/engine.length_scale:.1f} Rs), θ={theta:.3f}, φ={phi:.3f}")
        
        # Check if particle is moving
        r_values = hist[:, 1]
        r_range = r_values.max() - r_values.min()
        print(f"\nRadius statistics:")
        print(f"  Range: {r_range:.3e} m")
        print(f"  Is particle stuck? {r_range < 1e-15}")
        
        if r_range < 1e-15:
            print("\n⚠️  WARNING: Particle appears to be stuck at r ≈ 0!")
            print("This suggests an issue with unit conversion or initial condition setup.")
    else:
        print("ERROR: No trajectory returned!")

if __name__ == "__main__":
    main()