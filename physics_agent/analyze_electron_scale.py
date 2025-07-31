#!/usr/bin/env python3
"""
Analyze electron trajectory scale to determine proper zoom for visualization.
"""

import os
import torch
import numpy as np
import json

def analyze_trajectory_scale(traj_file, info_file):
    """Analyze the scale of a particle trajectory."""
    
    # Load trajectory
    traj = torch.load(traj_file)
    if isinstance(traj, torch.Tensor):
        traj = traj.cpu().numpy()
    
    # Load particle info
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    particle_props = info.get('particle_properties', {})
    print(f"\nParticle: {particle_props.get('name', info.get('particle_name', 'Unknown'))}")
    print(f"Mass: {particle_props.get('mass', 'N/A')} kg")
    print(f"Charge: {particle_props.get('charge', 0)} C")
    
    # Extract coordinates (assuming [t, r, phi, ...] format)
    t = traj[:, 0]
    r = traj[:, 1]
    phi = traj[:, 2]
    
    # Statistics
    # The trajectory is stored in SI units (meters)!
    # To convert to Schwarzschild radii: divide by actual Schwarzschild radius
    rs_meters = 2953.4  # Schwarzschild radius for solar mass
    r_rs = r / rs_meters  # Convert meters to Schwarzschild radii
    
    print(f"\nTrajectory statistics:")
    print(f"  Points: {len(traj)}")
    print(f"  Time range: [{t.min():.3e}, {t.max():.3e}]")
    print(f"  Radial range: [{r_rs.min():.3f}, {r_rs.max():.3f}] Schwarzschild radii")
    print(f"  Mean radius: {r_rs.mean():.3f} Schwarzschild radii")
    print(f"  Radial std dev: {r_rs.std():.6f} Schwarzschild radii")
    
    # Convert to Cartesian for visualization scale (using Schwarzschild radii)
    x_rs = r_rs * np.cos(phi)
    y_rs = r_rs * np.sin(phi)
    
    x_range = x_rs.max() - x_rs.min()
    y_range = y_rs.max() - y_rs.min()
    max_range = max(x_range, y_range, r_rs.max() - r_rs.min())
    
    print(f"\nCartesian ranges:")
    print(f"  X range: {x_range:.9e} rs")
    print(f"  Y range: {y_range:.9e} rs")
    print(f"  Max range: {max_range:.9e} rs")
    
    # Scale recommendations
    print(f"\nScale analysis:")
    print(f"  Current plot range: ±20 rs (4.0e+01 rs total)")
    
    if max_range < 1e-10:
        # Quantum scale
        suggested_range = max(r_rs.mean() * 0.1, 1e-15)
        print(f"  QUANTUM REGIME DETECTED!")
        print(f"  Trajectory variation: {max_range:.3e} rs")
        print(f"  Suggested plot range: ±{suggested_range:.3e} rs")
        print(f"  Required zoom factor: {20/suggested_range:.0e}x")
        print(f"  \n  >> The current visualization is {20/suggested_range:.0e} times too large!")
    else:
        # Classical scale
        suggested_range = max_range * 2
        print(f"  Classical trajectory detected")
        print(f"  Suggested plot range: ±{suggested_range:.3e} rs")
        print(f"  Required zoom factor: {20/suggested_range:.1f}x")
    
    return {
        'r_mean': r_rs.mean(),
        'r_range': r_rs.max() - r_rs.min(),
        'max_range': max_range,
        'suggested_range': suggested_range,
        'zoom_factor': 20/suggested_range
    }


def main():
    # Use the most recent run directory
    import glob
    run_dirs = sorted(glob.glob("runs/run_*_float64"))
    if run_dirs:
        latest_run = run_dirs[-1]
        print(f"Using latest run: {latest_run}")
        base_dir = f"{latest_run}/Alena‑Tensor‑γ_+0_00/particles"
    else:
        print("No run directories found!")
        return
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return
    
    print("="*70)
    print("TRAJECTORY SCALE ANALYSIS - ALENA TENSOR THEORY")
    print("="*70)
    
    # Analyze all particles
    particles = ['electron', 'proton', 'photon', 'neutrino']
    results = {}
    
    for particle in particles:
        traj_file = os.path.join(base_dir, f"{particle}_trajectory.pt")
        info_file = os.path.join(base_dir, f"{particle}_info.json")
        
        if os.path.exists(traj_file) and os.path.exists(info_file):
            results[particle] = analyze_trajectory_scale(traj_file, info_file)
    
    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION RECOMMENDATIONS:")
    print("="*70)
    
    for particle, data in results.items():
        print(f"\n{particle.upper()}:")
        print(f"  Current visualization scale: ±20 rs")
        print(f"  Recommended scale: ±{data['suggested_range']:.3e} rs")
        print(f"  ZOOM IN BY: {data['zoom_factor']:.0e}x")
        
        if data['zoom_factor'] > 1e10:
            print(f"  ⚠️  EXTREME ZOOM NEEDED - Quantum effects at play!")
    
    # Find the electron specifically
    if 'electron' in results:
        e_data = results['electron']
        print(f"\n{'='*70}")
        print("ELECTRON VISUALIZATION FIX:")
        print(f"{'='*70}")
        print(f"The electron trajectory is {e_data['zoom_factor']:.0e} times smaller than shown!")
        print(f"No wonder you can't see the quantum uncertainty region properly.")
        print(f"\nTo fix the visualization, the axis limits should be:")
        print(f"  ax.set_xlim(-{e_data['suggested_range']:.3e}, {e_data['suggested_range']:.3e})")
        print(f"  ax.set_ylim(-{e_data['suggested_range']:.3e}, {e_data['suggested_range']:.3e})")
        print(f"\nNot:")
        print(f"  ax.set_xlim(-20, 20)  # Current wrong scale")


if __name__ == "__main__":
    main()