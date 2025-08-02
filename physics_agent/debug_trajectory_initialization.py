#!/usr/bin/env python3
"""Debug script to trace where r becomes 0 in trajectory initialization."""

import torch
import sys
sys.path.append('.')

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT

def debug_trajectory_initialization():
    # Initialize engine with primordial mini black hole
    engine = TheoryEngine(black_hole_preset='primordial_mini', verbose=True)
    
    print(f"Engine parameters:")
    print(f"  M_si = {engine.M_si} kg")
    print(f"  length_scale = {engine.length_scale} m")
    print(f"  time_scale = {engine.time_scale} s")
    
    # Test with Schwarzschild
    theory = Schwarzschild()
    
    # Set up initial radius
    r0_si = 10 * engine.length_scale
    print(f"\nInitial radius:")
    print(f"  r0_si = {r0_si} m")
    print(f"  r0_si / length_scale = {r0_si / engine.length_scale} (should be 10)")
    
    # Convert to geometric units
    r0_geom = r0_si / engine.length_scale
    print(f"  r0_geom = {r0_geom}")
    
    # Get initial conditions
    y0_sym, y0_gen, _ = engine.get_initial_conditions(theory, torch.tensor(r0_geom))
    
    print(f"\nInitial conditions from get_initial_conditions:")
    print(f"  y0_symmetric = {y0_sym}")
    print(f"  y0_general = {y0_gen}")
    print(f"  r from y0_sym[1] = {y0_sym[1].item()}")
    print(f"  r from y0_gen[1] = {y0_gen[1].item()}")
    
    # Now try running a short trajectory
    print(f"\nRunning trajectory...")
    n_steps = 10
    dtau_si = engine.bh_preset.integration_parameters['dtau_geometric'] * engine.time_scale
    
    hist, solver_tag, step_times = engine.run_trajectory(
        theory, r0_si, n_steps, dtau_si, no_cache=True
    )
    
    if hist is not None:
        print(f"\nTrajectory results:")
        print(f"  Solver: {solver_tag}")
        print(f"  Shape: {hist.shape}")
        print(f"  First 3 points:")
        for i in range(min(3, len(hist))):
            print(f"    {i}: t={hist[i,0]:.3e}, r={hist[i,1]:.6e}, θ={hist[i,2]:.6f}, φ={hist[i,3]:.6f}")
            
        # Check if r is changing
        r_values = hist[:, 1]
        print(f"\n  r statistics:")
        print(f"    min = {r_values.min().item():.6e}")
        print(f"    max = {r_values.max().item():.6e}")
        print(f"    std = {r_values.std().item():.6e}")
        print(f"    r[0] = {r_values[0].item():.6e}")
        print(f"    r[-1] = {r_values[-1].item():.6e}")
    
    # Now test with Quantum Corrected theory
    print(f"\n\n{'='*60}")
    print("Testing with Quantum Corrected theory...")
    theory2 = QuantumCorrected()
    
    hist2, solver_tag2, _ = engine.run_trajectory(
        theory2, r0_si, n_steps, dtau_si, no_cache=True
    )
    
    if hist2 is not None:
        print(f"\nQuantum Corrected results:")
        print(f"  Solver: {solver_tag2}")
        r_values2 = hist2[:, 1]
        print(f"  r statistics:")
        print(f"    min = {r_values2.min().item():.6e}")
        print(f"    max = {r_values2.max().item():.6e}")
        print(f"    r[0] = {r_values2[0].item():.6e}")

if __name__ == "__main__":
    debug_trajectory_initialization()