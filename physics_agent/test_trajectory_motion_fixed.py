#!/usr/bin/env python3
"""Test trajectory motion with correct timestep."""

import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected

def test_trajectory_motion():
    print("=== Testing Trajectory Motion with Correct Timestep ===\n")
    
    # Create engine
    engine = TheoryEngine(black_hole_preset='primordial_mini', verbose=False)
    
    print(f"Black hole: {engine.bh_preset.name}")
    print(f"  Mass: {engine.M_si:.3e} kg ({engine.M_si/1.989e30:.3e} M☉)")
    print(f"  Schwarzschild radius: {2*engine.length_scale:.3e} m\n")
    
    # Test theories
    theories = [
        ("Schwarzschild", Schwarzschild()),
        ("Quantum Corrected", QuantumCorrected())
    ]
    
    # Initial conditions
    r0_si = 10 * engine.length_scale  # 10M in geometric units
    n_steps = 100
    
    # Use recommended timestep from config
    dtau_geom = engine.bh_preset.integration_parameters['dtau_geometric']
    dtau_si = dtau_geom * engine.time_scale
    
    print(f"Integration parameters:")
    print(f"  Initial radius: {r0_si:.3e} m (10 Rs)")
    print(f"  Timestep: {dtau_si:.3e} s ({dtau_geom} geometric units)")
    print(f"  Steps: {n_steps}\n")
    
    for name, theory in theories:
        print(f"\nTesting {name}...")
        
        # Run trajectory without cache
        hist, solver_tag, _ = engine.run_trajectory(
            theory, r0_si, n_steps, dtau_si, 
            no_cache=True,
            verbose=False
        )
        
        if hist is not None:
            print(f"  Solver: {solver_tag}")
            print(f"  Trajectory shape: {hist.shape}")
            
            # Analyze motion
            t_vals = hist[:, 0]
            r_vals = hist[:, 1]
            theta_vals = hist[:, 2]
            phi_vals = hist[:, 3]
            
            # Check ranges
            print(f"\n  Coordinate ranges:")
            print(f"    t: {t_vals[0]:.3e} to {t_vals[-1]:.3e} s")
            print(f"    r: {r_vals.min():.3e} to {r_vals.max():.3e} m")
            print(f"    φ: {phi_vals[0]:.3f} to {phi_vals[-1]:.3f} rad")
            
            # Calculate motion metrics
            r_range = r_vals.max() - r_vals.min()
            phi_range = phi_vals[-1] - phi_vals[0]
            
            print(f"\n  Motion analysis:")
            print(f"    Radial variation: {r_range:.3e} m ({r_range/r0_si*100:.1f}% of r0)")
            print(f"    Angular motion: {phi_range:.3f} rad ({phi_range/(2*np.pi):.2f} orbits)")
            
            # Convert to 3D and calculate distance
            x = r_vals * np.sin(theta_vals) * np.cos(phi_vals)
            y = r_vals * np.sin(theta_vals) * np.sin(phi_vals)
            z = r_vals * np.cos(theta_vals)
            
            xyz = np.stack([x, y, z], axis=1)
            diffs = np.diff(xyz, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            total_distance = distances.sum()
            
            print(f"    Total 3D distance: {total_distance:.3e} m")
            print(f"    Average speed: {total_distance/(t_vals[-1]-t_vals[0]):.3e} m/s")
            print(f"    (c = {engine.c_si:.3e} m/s for reference)")
            
            # Check if stuck
            is_stuck = r_range < 1e-15 * r0_si
            print(f"\n  Particle stuck at origin? {'YES ⚠️' if is_stuck else 'NO ✓'}")
            
            # Show first and last few points
            print(f"\n  First 3 points:")
            for i in range(min(3, len(hist))):
                print(f"    {i}: r={r_vals[i]:.3e} m, φ={phi_vals[i]:.3f} rad")
                
            if len(hist) > 6:
                print(f"\n  Last 3 points:")
                for i in range(len(hist)-3, len(hist)):
                    print(f"    {i}: r={r_vals[i]:.3e} m, φ={phi_vals[i]:.3f} rad")
        else:
            print("  ERROR: No trajectory returned!")
            
    print("\n" + "="*60)

if __name__ == "__main__":
    test_trajectory_motion()