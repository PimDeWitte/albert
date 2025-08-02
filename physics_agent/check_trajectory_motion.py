#!/usr/bin/env python3
"""Check if particles in cached trajectories are actually moving."""

import torch
import numpy as np
import sys

def check_trajectory_motion(cache_file):
    """Load and analyze a cached trajectory file."""
    print(f"\nAnalyzing: {cache_file}")
    print("-" * 60)
    
    # Load the cached data
    data = torch.load(cache_file, map_location='cpu')
    
    # Extract trajectory data
    if isinstance(data, dict) and 'trajectory' in data:
        hist = data['trajectory']
    else:
        hist = data
    
    print(f"Trajectory shape: {hist.shape}")
    print(f"Number of steps: {len(hist)}")
    
    # Check the structure - typically [t, r, theta, phi, ...]
    if hist.shape[1] >= 4:
        t_vals = hist[:, 0]
        r_vals = hist[:, 1]  
        theta_vals = hist[:, 2]
        phi_vals = hist[:, 3]
        
        # Check if coordinates are changing
        print(f"\nCoordinate ranges:")
        print(f"  t: {t_vals.min():.6e} to {t_vals.max():.6e} (Δ={t_vals.max() - t_vals.min():.6e})")
        print(f"  r: {r_vals.min():.6f} to {r_vals.max():.6f} (Δ={r_vals.max() - r_vals.min():.6e})")
        print(f"  θ: {theta_vals.min():.6f} to {theta_vals.max():.6f} (Δ={theta_vals.max() - theta_vals.min():.6e})")
        print(f"  φ: {phi_vals.min():.6f} to {phi_vals.max():.6f} (Δ={phi_vals.max() - phi_vals.min():.6e})")
        
        # Calculate actual movement
        print(f"\nMovement analysis:")
        
        # Check if r is changing
        r_std = r_vals.std().item()
        r_change = (r_vals[-1] - r_vals[0]).item()
        print(f"  r std dev: {r_std:.6e}")
        print(f"  r total change: {r_change:.6e}")
        
        # Check if phi is changing (angular motion)
        phi_std = phi_vals.std().item()
        phi_change = (phi_vals[-1] - phi_vals[0]).item()
        print(f"  φ std dev: {phi_std:.6e}")
        print(f"  φ total change: {phi_change:.6e}")
        
        # Calculate distance traveled in 3D space
        if hist.shape[1] >= 4:
            # Convert spherical to Cartesian
            x = r_vals * torch.sin(theta_vals) * torch.cos(phi_vals)
            y = r_vals * torch.sin(theta_vals) * torch.sin(phi_vals) 
            z = r_vals * torch.cos(theta_vals)
            
            xyz = torch.stack([x, y, z], dim=1)
            diffs = torch.diff(xyz, dim=0)
            distances = torch.norm(diffs, dim=1)
            total_distance = distances.sum().item()
            
            print(f"\n  Total 3D distance traveled: {total_distance:.6e}")
            print(f"  Average step size: {distances.mean().item():.6e}")
        
        # Check if particle is stationary
        is_stationary = r_std < 1e-10 and phi_std < 1e-10
        print(f"\n  Particle is {'STATIONARY' if is_stationary else 'MOVING'}")
        
        # Show first few and last few points
        print(f"\nFirst 5 trajectory points:")
        for i in range(min(5, len(hist))):
            print(f"  {i}: t={hist[i,0]:.3e}, r={hist[i,1]:.6f}, θ={hist[i,2]:.6f}, φ={hist[i,3]:.6f}")
            
        if len(hist) > 10:
            print(f"\nLast 5 trajectory points:")
            for i in range(max(0, len(hist)-5), len(hist)):
                print(f"  {i}: t={hist[i,0]:.3e}, r={hist[i,1]:.6f}, θ={hist[i,2]:.6f}, φ={hist[i,3]:.6f}")
    
    return hist

if __name__ == "__main__":
    # Check a few different cached trajectories
    cache_dir = "physics_agent/cache/trajectories/1.0.0/Primordial_Mini_Black_Hole/"
    
    test_files = [
        "Schwarzschild_a8b7beb9b5aa8c8f_steps_1000.pt",
        "Quantum_Corrected_____0_01__bd62a33baccd4a63_steps_1000.pt",
        "Loop_Quantum_Gravity____0_2375__bed441297ce27c9a_steps_1000.pt",
    ]
    
    for file in test_files:
        try:
            hist = check_trajectory_motion(cache_dir + file)
        except Exception as e:
            print(f"Error loading {file}: {e}")