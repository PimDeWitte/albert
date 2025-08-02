#!/usr/bin/env python3
"""Test trajectory accuracy and distance calculations."""

import sys
sys.path.insert(0, '.')

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.kerr.theory import Kerr
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
import torch
import numpy as np

def test_trajectory_accuracy():
    """Test that trajectory calculations are accurate."""
    
    # Initialize engine
    engine = TheoryEngine(black_hole_preset='primordial_mini', verbose=False)
    
    # Test with Kerr (should match baseline exactly)
    kerr = Kerr(a=0.0)  # Schwarzschild limit
    
    # Initial conditions for circular orbit at r=10M
    r0_si = 10 * engine.length_scale
    n_steps = 1000
    dtau_si = engine.bh_preset.integration_parameters['dtau_geometric'] * engine.time_scale
    
    print(f"Testing trajectory accuracy...")
    print(f"Black hole: {engine.bh_preset.name}")
    print(f"Initial radius: {r0_si/engine.length_scale:.1f}M = {r0_si:.2e} m")
    print(f"Steps: {n_steps}")
    print(f"Time step: {dtau_si:.2e} s = {dtau_si/engine.time_scale:.3f} geometric units")
    print()
    
    # Run Kerr trajectory
    print("Running Kerr trajectory...")
    hist, solver_tag, _ = engine.run_trajectory(kerr, r0_si, n_steps, dtau_si)
    
    if hist is None:
        print("ERROR: Failed to compute trajectory!")
        return
    
    print(f"Trajectory shape: {hist.shape}")
    print(f"Solver: {solver_tag}")
    
    # Calculate distance traveled
    def spherical_to_cartesian(traj):
        r = traj[:, 1]
        theta = traj[:, 2]
        phi = traj[:, 3]
        
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        
        return torch.stack([x, y, z], dim=1)
    
    xyz = spherical_to_cartesian(hist)
    diffs = torch.diff(xyz, dim=0)
    distance_si = torch.sum(torch.norm(diffs, dim=1)).item()
    distance_M = distance_si / engine.length_scale
    
    # Extract trajectory info
    r = hist[:, 1].numpy()
    phi = hist[:, 3].numpy()
    t = hist[:, 0].numpy()
    
    r_M = r / engine.length_scale
    
    # Calculate orbits
    n_orbits = np.abs(phi[-1] - phi[0]) / (2 * np.pi)
    
    # Expected values for circular orbit
    # Circumference = 2πr
    expected_distance_per_orbit = 2 * np.pi * r_M[0]
    expected_total_distance = expected_distance_per_orbit * n_orbits
    
    print(f"\nTrajectory Analysis:")
    print(f"Initial r: {r_M[0]:.2f}M")
    print(f"Final r: {r_M[-1]:.2f}M")
    print(f"Min r: {np.min(r_M):.2f}M")
    print(f"Max r: {np.max(r_M):.2f}M")
    print(f"Orbits completed: {n_orbits:.2f}")
    print(f"Distance traveled: {distance_M:.1f}M")
    print(f"Expected distance: {expected_total_distance:.1f}M")
    print(f"Ratio: {distance_M/expected_total_distance:.3f}")
    
    # Test quantum theory
    print("\n" + "="*60)
    print("Testing Quantum Corrected theory...")
    
    qc = QuantumCorrected()
    hist_qc, solver_tag_qc, _ = engine.run_trajectory(qc, r0_si, n_steps, dtau_si)
    
    if hist_qc is not None:
        xyz_qc = spherical_to_cartesian(hist_qc)
        diffs_qc = torch.diff(xyz_qc, dim=0)
        distance_qc_M = torch.sum(torch.norm(diffs_qc, dim=1)).item() / engine.length_scale
        
        # Calculate loss
        min_len = min(len(hist), len(hist_qc))
        loss = torch.mean((hist[:min_len, 1] - hist_qc[:min_len, 1])**2).item()
        
        print(f"Solver: {solver_tag_qc}")
        print(f"Distance traveled: {distance_qc_M:.1f}M")
        print(f"Ratio to Kerr: {distance_qc_M/distance_M:.3f}")
        print(f"Radial MSE loss vs Kerr: {loss:.2e}")
    else:
        print("ERROR: Failed to compute quantum trajectory!")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_trajectory_accuracy()