#!/usr/bin/env python3
"""Verify 3D trajectory data to understand motion patterns."""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, '.')

def analyze_trajectory():
    """Analyze a trajectory to understand its 3D motion."""
    
    cache_dir = 'physics_agent/cache/trajectories/1.0.0/Primordial_Mini_Black_Hole'
    
    # Find a Schwarzschild trajectory
    test_file = None
    for f in os.listdir(cache_dir):
        if 'Schwarzschild' in f and f.endswith('.pt'):
            test_file = f
            break
    
    if not test_file:
        print("No trajectory found")
        return
        
    print(f"Analyzing: {test_file}")
    
    # Load trajectory
    data = torch.load(os.path.join(cache_dir, test_file), weights_only=True)
    if isinstance(data, dict) and 'trajectory' in data:
        traj = data['trajectory']
    else:
        traj = data
    
    print(f"\nTrajectory shape: {traj.shape}")
    print(f"Number of steps: {traj.shape[0]}")
    
    # Extract coordinates
    t = traj[:, 0].numpy()
    r = traj[:, 1].numpy()
    theta = traj[:, 2].numpy()
    phi = traj[:, 3].numpy()
    
    # Convert to Cartesian for visualization
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Analyze ranges
    print(f"\nCoordinate ranges:")
    print(f"  r: {r.min():.6e} to {r.max():.6e} meters")
    print(f"  theta: {theta.min():.6f} to {theta.max():.6f} radians")
    print(f"  phi: {phi.min():.6f} to {phi.max():.6f} radians")
    
    print(f"\nCartesian ranges:")
    print(f"  x: {x.min():.6e} to {x.max():.6e} meters")
    print(f"  y: {y.min():.6e} to {y.max():.6e} meters")
    print(f"  z: {z.min():.6e} to {z.max():.6e} meters")
    
    # Check if motion is planar
    theta_variation = theta.max() - theta.min()
    print(f"\nTheta variation: {theta_variation:.6f} radians")
    if theta_variation < 0.01:
        print("Motion is essentially in the equatorial plane (theta ≈ π/2)")
        print(f"Average theta: {theta.mean():.6f} radians ({theta.mean() * 180 / np.pi:.2f} degrees)")
    
    # Sample some points
    print(f"\nSample points (first 5 and last 5):")
    print("Step |     r (m)      |  theta (rad) |   phi (rad)  |     x (m)      |     y (m)      |     z (m)")
    print("-" * 100)
    
    for i in [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]:
        print(f"{i:4d} | {r[i]:14.6e} | {theta[i]:12.6f} | {phi[i]:12.6f} | {x[i]:14.6e} | {y[i]:14.6e} | {z[i]:14.6e}")
    
    # Convert to geometric units for better understanding
    Rs = 1.485e-12  # Schwarzschild radius for primordial mini black hole
    r_geom = r / Rs
    
    print(f"\nIn geometric units (M = Schwarzschild radii):")
    print(f"  r: {r_geom.min():.2f} to {r_geom.max():.2f} M")
    print(f"  Circular orbit at ISCO would be at r = 6 M")

if __name__ == "__main__":
    analyze_trajectory()