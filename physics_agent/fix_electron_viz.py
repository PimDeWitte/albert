#!/usr/bin/env python3
"""
Fix electron visualization scale to properly show quantum effects.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.constants import ELECTRON_MASS


def analyze_electron_scale(run_dir):
    """Analyze the actual scale of electron trajectories to determine proper zoom."""
    
    # Look for trajectory data
    for subdir in os.listdir(run_dir):
        if 'Alena' in subdir:
            theory_dir = os.path.join(run_dir, subdir)
            traj_file = os.path.join(theory_dir, 'trajectory_electron.npy')
            
            if os.path.exists(traj_file):
                # Load trajectory
                traj = np.load(traj_file)
                
                # Extract coordinates (assuming [t, r, phi, ...] format)
                t = traj[:, 0]
                r = traj[:, 1]
                phi = traj[:, 2]
                
                # Convert to Cartesian
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                
                print(f"Electron trajectory analysis for {subdir}:")
                print(f"  Number of points: {len(traj)}")
                print(f"  Time range: [{t.min():.3e}, {t.max():.3e}]")
                print(f"  Radial range: [{r.min():.6e}, {r.max():.6e}] Schwarzschild radii")
                print(f"  Mean radius: {r.mean():.6e} Schwarzschild radii")
                print(f"  Radial variation: {r.std():.6e} Schwarzschild radii")
                
                # Calculate quantum uncertainty scale
                # For an electron near a black hole, the quantum uncertainty is roughly
                # Δr ~ λ_Compton / r_s where λ_Compton = h/(m_e * c)
                # In geometric units where G=c=1, this becomes roughly 10^-20 for solar mass BH
                quantum_scale = 1e-20  # Rough estimate
                
                print(f"\nScale recommendations:")
                if r.std() < 1e-10:
                    # Very small variation - likely quantum dominated
                    suggested_range = max(r.mean() * 0.1, 1e-18)
                    print(f"  Very small trajectory variation detected")
                    print(f"  Suggested plot range: ±{suggested_range:.3e} Schwarzschild radii")
                    print(f"  This is {suggested_range/20:.0e}x smaller than current ±20 range!")
                else:
                    # Larger variation
                    suggested_range = max(r.max() - r.min(), r.mean() * 0.1) * 5
                    print(f"  Suggested plot range: ±{suggested_range:.3e} Schwarzschild radii")
                
                return {
                    'trajectory': traj,
                    'r_mean': r.mean(),
                    'r_std': r.std(),
                    'suggested_range': suggested_range
                }
    
    return None


def create_zoomed_visualization(traj_data, output_file):
    """Create a properly zoomed visualization of the electron trajectory."""
    
    traj = traj_data['trajectory']
    suggested_range = traj_data['suggested_range']
    
    # Extract coordinates
    t = traj[:, 0]
    r = traj[:, 1]
    phi = traj[:, 2]
    
    # Convert to Cartesian
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = t
    
    # Create figure
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Plot trajectory with gradient color
    points = ax.scatter(x, y, z, c=t, cmap='plasma', s=20, alpha=0.8)
    
    # Add quantum uncertainty visualization
    # Show as a cloud around the mean position
    r_mean = traj_data['r_mean']
    uncertainty_scale = max(traj_data['r_std'] * 3, suggested_range * 0.1)
    
    # Create uncertainty cloud
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = r_mean * np.outer(np.cos(u), np.sin(v))
    y_sphere = r_mean * np.outer(np.sin(u), np.sin(v))
    z_sphere = np.mean(z) + uncertainty_scale * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='cyan', alpha=0.2)
    
    # Set proper axis limits - THIS IS THE KEY FIX
    ax.set_xlim(-suggested_range, suggested_range)
    ax.set_ylim(-suggested_range, suggested_range)
    ax.set_zlim(z.min(), z.max())
    
    # Labels with scientific notation
    ax.set_xlabel(f'X (Schwarzschild radii)', color='white')
    ax.set_ylabel(f'Y (Schwarzschild radii)', color='white')
    ax.set_zlabel('Time', color='white')
    
    # Title
    ax.set_title(f'Electron Trajectory - Properly Zoomed\nScale: ±{suggested_range:.2e} rs', 
                 color='white', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(points, ax=ax, pad=0.1)
    cbar.set_label('Time', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Grid and styling
    ax.grid(True, alpha=0.3)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(colors='white')
    
    # Add scale reference
    ax.text2D(0.05, 0.95, f'Current viz shows ±20 rs\nThis shows ±{suggested_range:.2e} rs\n'
              f'Zoom factor: {20/suggested_range:.0e}x', 
              transform=ax.transAxes, color='yellow', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, facecolor='black')
    print(f"\nSaved zoomed visualization to: {output_file}")
    
    return fig


def main():
    # Find the latest run directory
    run_dir = "runs/run_20250731_120700_float64"
    
    if not os.path.exists(run_dir):
        print(f"Run directory not found: {run_dir}")
        return
    
    print(f"Analyzing electron trajectories in: {run_dir}")
    print("="*60)
    
    # Analyze scale
    traj_data = analyze_electron_scale(run_dir)
    
    if traj_data:
        # Create zoomed visualization
        output_file = "electron_trajectory_zoomed.png"
        create_zoomed_visualization(traj_data, output_file)
        
        print("\n" + "="*60)
        print("CONCLUSION:")
        print(f"The visualization scale was off by a factor of ~{20/traj_data['suggested_range']:.0e}!")
        print("Electron quantum effects happen at MUCH smaller scales than planetary orbits.")
        print("The fixed visualization properly shows the quantum uncertainty region.")
    else:
        print("No electron trajectory data found.")


if __name__ == "__main__":
    main()