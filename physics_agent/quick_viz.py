#!/usr/bin/env python3
"""
Quick visualization script for theory runs.
Simpler version that focuses on just visualizing one theory.

Usage:
    python quick_viz.py runs/run_20250719_153447_float64
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add physics_agent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def visualize_trajectory(trajectory_path, output_path=None):
    """Create a quick visualization of a trajectory"""
    
    # Load trajectory
    print(f"Loading trajectory from: {trajectory_path}")
    hist = torch.load(trajectory_path, map_location='cpu')
    
    # Extract components
    t = hist[:, 0].numpy()
    r = hist[:, 1].numpy()
    phi = hist[:, 2].numpy()
    
    # Convert to Cartesian
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Trajectory Analysis: {os.path.dirname(trajectory_path)}', fontsize=14)
    
    # 1. XY Orbit
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x, y, 'b-', alpha=0.8, linewidth=1)
    ax1.scatter(x[0], y[0], c='green', s=100, label='Start', zorder=5)
    ax1.scatter(x[-1], y[-1], c='red', s=100, label='End', zorder=5)
    
    # Add step markers
    step_interval = max(1, len(x) // 50)
    for i in range(0, len(x), step_interval):
        ax1.scatter(x[i], y[i], c='yellow', s=20, edgecolors='black', linewidth=0.5)
        if i % (step_interval * 5) == 0:
            ax1.annotate(f'{i}', (x[i], y[i]), fontsize=6)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Orbital Trajectory (XY)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Radial evolution
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(t, r, 'r-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Radius (m)')
    ax2.set_title('Radial Distance vs Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Angular evolution
    ax3 = plt.subplot(2, 3, 3)
    phi_unwrapped = np.unwrap(phi)
    ax3.plot(t, phi_unwrapped / (2 * np.pi), 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (revolutions)')
    ax3.set_title('Angular Position vs Time')
    ax3.grid(True, alpha=0.3)
    
    # 4. Phase space (r vs dr/dt)
    ax4 = plt.subplot(2, 3, 4)
    dr_dt = np.gradient(r, t)
    ax4.plot(r[10:-10], dr_dt[10:-10], 'b-', alpha=0.8)
    ax4.set_xlabel('r (m)')
    ax4.set_ylabel('dr/dt (m/s)')
    ax4.set_title('Phase Space (r vs dr/dt)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle=':', alpha=0.5)
    
    # 5. Energy proxy (simplified)
    ax5 = plt.subplot(2, 3, 5)
    # Kinetic energy proxy: v² ≈ (dr/dt)² + r²(dφ/dt)²
    dphi_dt = np.gradient(phi_unwrapped, t)
    v_squared = dr_dt**2 + r**2 * dphi_dt**2
    ax5.plot(t[10:-10], v_squared[10:-10], 'm-', linewidth=1)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('v² (m²/s²)')
    ax5.set_title('Velocity Squared vs Time')
    ax5.grid(True, alpha=0.3)
    
    # 6. 3D trajectory
    ax6 = plt.subplot(2, 3, 6, projection='3d')
    ax6.plot(x, y, t, 'b-', alpha=0.8)
    
    # Add step markers in 3D
    for i in range(0, len(x), step_interval * 2):
        ax6.scatter(x[i], y[i], t[i], c='yellow', s=20, edgecolors='black')
        if i % (step_interval * 10) == 0:
            ax6.text(x[i], y[i], t[i], f'{i}', fontsize=6)
    
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_zlabel('Time (s)')
    ax6.set_title('3D Trajectory (X, Y, Time)')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Quick visualization of theory trajectories')
    parser.add_argument('path', help='Path to run directory or trajectory file')
    parser.add_argument('--output', '-o', help='Output filename (default: show plot)')
    
    args = parser.parse_args()
    
    # Import numpy here after argparse
    global np
    import numpy as np
    
    # Determine if path is a directory or file
    if os.path.isdir(args.path):
        # Look for trajectory files in directory
        run_dir = args.path
        trajectory_files = ['trajectory.pt', 'trajectory_cached.pt']
        
        trajectory_path = None
        for fname in trajectory_files:
            fpath = os.path.join(run_dir, fname)
            if os.path.exists(fpath):
                trajectory_path = fpath
                break
        
        if not trajectory_path:
            # Check subdirectories
            for subdir in os.listdir(run_dir):
                subpath = os.path.join(run_dir, subdir)
                if os.path.isdir(subpath):
                    for fname in trajectory_files:
                        fpath = os.path.join(subpath, fname)
                        if os.path.exists(fpath):
                            trajectory_path = fpath
                            break
                    if trajectory_path:
                        break
        
        if not trajectory_path:
            print(f"Error: No trajectory file found in {run_dir}")
            sys.exit(1)
            
    elif os.path.isfile(args.path) and args.path.endswith('.pt'):
        trajectory_path = args.path
    else:
        print(f"Error: {args.path} is not a valid directory or .pt file")
        sys.exit(1)
    
    # Create output path if not specified
    if args.output is None and os.path.isdir(args.path):
        args.output = os.path.join(args.path, 'quick_visualization.png')
    
    # Generate visualization
    visualize_trajectory(trajectory_path, args.output)

if __name__ == "__main__":
    main() 