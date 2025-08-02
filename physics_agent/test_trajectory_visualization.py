#!/usr/bin/env python3
"""Test trajectory visualization to ensure particles appear to be moving."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '.')

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild

def visualize_trajectory(hist, title, length_scale):
    """Create 3D visualization of trajectory."""
    
    # Extract coordinates
    t = hist[:, 0].numpy()
    r = hist[:, 1].numpy()
    theta = hist[:, 2].numpy()
    phi = hist[:, 3].numpy()
    
    # Convert to Cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 5))
    
    # 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(x/length_scale, y/length_scale, z/length_scale, 'b-', linewidth=1)
    ax1.scatter(x[0]/length_scale, y[0]/length_scale, z[0]/length_scale, 
                color='green', s=100, label='Start')
    ax1.scatter(x[-1]/length_scale, y[-1]/length_scale, z[-1]/length_scale, 
                color='red', s=100, label='End')
    ax1.set_xlabel('x/M')
    ax1.set_ylabel('y/M')
    ax1.set_zlabel('z/M')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # r vs phi (orbital plot)
    ax2 = fig.add_subplot(132)
    ax2.plot(phi, r/length_scale, 'b-')
    ax2.scatter(phi[0], r[0]/length_scale, color='green', s=100, label='Start')
    ax2.scatter(phi[-1], r[-1]/length_scale, color='red', s=100, label='End')
    ax2.set_xlabel('φ (rad)')
    ax2.set_ylabel('r/M')
    ax2.set_title('r vs φ')
    ax2.grid(True)
    ax2.legend()
    
    # Polar plot
    ax3 = fig.add_subplot(133, projection='polar')
    ax3.plot(phi, r/length_scale, 'b-')
    ax3.scatter(phi[0], r[0]/length_scale, color='green', s=100)
    ax3.scatter(phi[-1], r[-1]/length_scale, color='red', s=100)
    ax3.set_title('Polar View')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    # Print statistics
    print(f"\nTrajectory statistics for {title}:")
    print(f"  Total points: {len(hist)}")
    print(f"  Time span: {t[0]:.3e} to {t[-1]:.3e} s")
    print(f"  r range: {r.min()/length_scale:.3f} to {r.max()/length_scale:.3f} M")
    print(f"  φ range: {phi[0]:.3f} to {phi[-1]:.3f} rad ({(phi[-1]-phi[0])/(2*np.pi):.2f} orbits)")
    print(f"  Distance from origin: {np.sqrt(x[-1]**2 + y[-1]**2 + z[-1]**2)/length_scale:.3f} M")
    
    # Check if trajectory looks stuck
    total_3d_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
    print(f"  Total 3D distance traveled: {total_3d_distance/length_scale:.1f} M")
    
    if total_3d_distance/length_scale < 0.1:
        print("  ⚠️ WARNING: Particle appears to be stuck!")
    else:
        print("  ✓ Particle is moving normally")
    
    return fig

def main():
    print("=== Testing Trajectory Visualization ===\n")
    
    # Initialize engine
    engine = TheoryEngine(black_hole_preset='primordial_mini', verbose=False)
    theory = Schwarzschild()
    
    # Parameters
    r0_si = 10 * engine.length_scale
    n_steps = 500
    dtau_si = engine.bh_preset.integration_parameters['dtau_geometric'] * engine.time_scale
    
    print(f"Running trajectory with:")
    print(f"  Black hole: {engine.bh_preset.name}")
    print(f"  Theory: {theory.name}")
    print(f"  Initial r: {r0_si/engine.length_scale:.1f}M")
    print(f"  Steps: {n_steps}")
    
    # Run trajectory
    hist, solver_tag, _ = engine.run_trajectory(
        theory, r0_si, n_steps, dtau_si,
        no_cache=True,
        verbose=False
    )
    
    if hist is not None:
        print(f"\nTrajectory computed with: {solver_tag}")
        
        # Visualize
        fig = visualize_trajectory(hist, f"{theory.name} Trajectory", engine.length_scale)
        
        # Save plot
        output_file = "test_trajectory_visualization.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")
        
        # Also create a simple x-y plot
        fig2, ax = plt.subplots(figsize=(6, 6))
        r = hist[:, 1].numpy()
        phi = hist[:, 3].numpy()
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        ax.plot(x/engine.length_scale, y/engine.length_scale, 'b-', alpha=0.7)
        ax.scatter(x[0]/engine.length_scale, y[0]/engine.length_scale, 
                   color='green', s=100, zorder=5, label='Start')
        ax.scatter(x[-1]/engine.length_scale, y[-1]/engine.length_scale, 
                   color='red', s=100, zorder=5, label='End')
        
        # Add circle for Schwarzschild radius
        circle = plt.Circle((0, 0), 2, fill=False, color='black', linestyle='--', label='Horizon')
        ax.add_patch(circle)
        
        ax.set_xlabel('x/M')
        ax.set_ylabel('y/M')
        ax.set_title('Top-down view (x-y plane)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.savefig("test_trajectory_xy.png", dpi=150, bbox_inches='tight')
        print(f"X-Y plot saved to: test_trajectory_xy.png")
        
        plt.show()
    else:
        print("ERROR: No trajectory returned!")

if __name__ == "__main__":
    main()