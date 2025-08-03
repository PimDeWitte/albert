#!/usr/bin/env python3
"""Generate trajectory visualization plots for all theories."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
sys.path.insert(0, '.')

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.evaluation import ALL_THEORIES

def create_trajectory_plots(hist, theory_name, engine, output_dir):
    """Create and save trajectory visualization plots for a theory."""
    
    # Extract coordinates
    t = hist[:, 0].numpy()
    r = hist[:, 1].numpy()
    theta = hist[:, 2].numpy()
    phi = hist[:, 3].numpy()
    
    # Convert to Cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Convert to geometric units (M)
    length_scale = engine.length_scale
    x_M = x / length_scale
    y_M = y / length_scale
    z_M = z / length_scale
    r_M = r / length_scale
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(x_M, y_M, z_M, 'b-', linewidth=0.8, alpha=0.7)
    ax1.scatter(x_M[0], y_M[0], z_M[0], color='green', s=100, label='Start', zorder=5)
    ax1.scatter(x_M[-1], y_M[-1], z_M[-1], color='red', s=100, label='End', zorder=5)
    
    # Add black hole
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_bh = 2 * np.outer(np.cos(u), np.sin(v))
    y_bh = 2 * np.outer(np.sin(u), np.sin(v))
    z_bh = 2 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_bh, y_bh, z_bh, color='black', alpha=0.3)
    
    ax1.set_xlabel('x/M')
    ax1.set_ylabel('y/M')
    ax1.set_zlabel('z/M')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # 2. X-Y plane (top view)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(x_M, y_M, 'b-', linewidth=0.8, alpha=0.7)
    ax2.scatter(x_M[0], y_M[0], color='green', s=100, zorder=5, label='Start')
    ax2.scatter(x_M[-1], y_M[-1], color='red', s=100, zorder=5, label='End')
    
    # Add circles for key radii
    circle_horizon = plt.Circle((0, 0), 2, fill=False, color='black', linestyle='-', linewidth=2, label='Horizon (2M)')
    circle_photon = plt.Circle((0, 0), 3, fill=False, color='orange', linestyle='--', label='Photon sphere (3M)')
    circle_isco = plt.Circle((0, 0), 6, fill=False, color='purple', linestyle=':', label='ISCO (6M)')
    ax2.add_patch(circle_horizon)
    ax2.add_patch(circle_photon)
    ax2.add_patch(circle_isco)
    
    ax2.set_xlabel('x/M')
    ax2.set_ylabel('y/M')
    ax2.set_title('X-Y Plane View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # 3. r vs t
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t * 1e20, r_M, 'b-', linewidth=1)
    ax3.axhline(y=2, color='black', linestyle='-', alpha=0.5, label='Horizon')
    ax3.axhline(y=6, color='purple', linestyle=':', alpha=0.5, label='ISCO')
    ax3.set_xlabel('t (×10⁻²⁰ s)')
    ax3.set_ylabel('r/M')
    ax3.set_title('Radial Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Polar plot (r vs φ)
    ax4 = fig.add_subplot(2, 3, 4, projection='polar')
    ax4.plot(phi, r_M, 'b-', linewidth=1)
    ax4.scatter(phi[0], r_M[0], color='green', s=100, zorder=5)
    ax4.scatter(phi[-1], r_M[-1], color='red', s=100, zorder=5)
    ax4.set_title('Polar View (r vs φ)')
    ax4.set_ylim(0, max(15, r_M.max() * 1.1))
    
    # 5. Phase space (r vs dr/dt)
    if len(hist) > 1:
        # Safe gradient calculation to avoid divide-by-zero warnings
        dt = np.diff(t)
        if np.any(dt > 0):
            # Manual gradient to avoid warnings
            dr_dt = np.zeros_like(r)
            dr_dt[0] = (r[1] - r[0]) / dt[0] if dt[0] > 0 else 0
            for i in range(1, len(r) - 1):
                if t[i+1] - t[i-1] > 0:
                    dr_dt[i] = (r[i+1] - r[i-1]) / (t[i+1] - t[i-1])
            if len(dt) > 0 and dt[-1] > 0:
                dr_dt[-1] = (r[-1] - r[-2]) / dt[-1]
        else:
            dr_dt = np.zeros_like(r)
            
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(r_M[1:], dr_dt[1:] / engine.c_si, 'b-', linewidth=0.8, alpha=0.7)
        ax5.scatter(r_M[0], dr_dt[0] / engine.c_si, color='green', s=100, zorder=5, label='Start')
        ax5.scatter(r_M[-1], dr_dt[-1] / engine.c_si, color='red', s=100, zorder=5, label='End')
        ax5.set_xlabel('r/M')
        ax5.set_ylabel('dr/dt / c')
        ax5.set_title('Phase Space')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # 6. Statistics panel
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    total_time = t[-1] - t[0]
    total_orbits = (phi[-1] - phi[0]) / (2 * np.pi)
    avg_r = r_M.mean()
    r_variation = r_M.std() / avg_r * 100
    
    # 3D distance
    diffs = np.diff(np.column_stack([x, y, z]), axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    total_distance = distances.sum()
    avg_speed = total_distance / total_time if total_time > 0 else 0
    
    stats_text = f"""Trajectory Statistics
    
Theory: {theory_name}
Total points: {len(hist)}
Time span: {total_time:.3e} s
Initial r: {r_M[0]:.2f} M
Final r: {r_M[-1]:.2f} M

Motion metrics:
• Orbits completed: {total_orbits:.2f}
• Average r: {avg_r:.2f} M
• r variation: {r_variation:.1f}%
• Total 3D distance: {total_distance/length_scale:.1f} M
• Average speed: {avg_speed/engine.c_si:.3f} c

Angular motion:
• Δφ = {phi[-1] - phi[0]:.3f} rad
• ω = {(phi[-1] - phi[0])/total_time if total_time > 0 else 0:.3e} rad/s
"""
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add theory description
    try:
        from physics_agent.trajectory_plot_descriptions import get_theory_description
        theory_desc = get_theory_description(theory_name)
        # Add description text box
        fig.text(0.02, 0.02, theory_desc, transform=fig.transFigure, 
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                wrap=True)
    except:
        pass
    
    # Overall title
    fig.suptitle(f'{theory_name} - Trajectory Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.15, 1, 0.96])  # Leave space for description
    
    # Save the plot
    safe_name = theory_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    output_path = os.path.join(output_dir, f'{safe_name}_trajectory.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    # Also create a simplified orbit plot
    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_M, y_M, 'b-', linewidth=1, alpha=0.8)
    ax.scatter(x_M[0], y_M[0], color='green', s=150, zorder=5, label='Start', edgecolor='darkgreen', linewidth=2)
    ax.scatter(x_M[-1], y_M[-1], color='red', s=150, zorder=5, label='End', edgecolor='darkred', linewidth=2)
    
    # Add reference circles
    circle_horizon = plt.Circle((0, 0), 2, fill=True, color='black', alpha=0.8)
    circle_photon = plt.Circle((0, 0), 3, fill=False, color='orange', linestyle='--', linewidth=2)
    circle_isco = plt.Circle((0, 0), 6, fill=False, color='purple', linestyle=':', linewidth=2)
    ax.add_patch(circle_horizon)
    ax.add_patch(circle_photon)
    ax.add_patch(circle_isco)
    
    ax.set_xlabel('x/M', fontsize=14)
    ax.set_ylabel('y/M', fontsize=14)
    ax.set_title(f'{theory_name} - Orbital Trajectory', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Set appropriate limits
    max_coord = max(abs(x_M).max(), abs(y_M).max()) * 1.1
    ax.set_xlim(-max_coord, max_coord)
    ax.set_ylim(-max_coord, max_coord)
    
    orbit_path = os.path.join(output_dir, f'{safe_name}_orbit.png')
    plt.savefig(orbit_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path, orbit_path


def main():
    print("=== Generating Trajectory Visualizations for All Theories ===\n")
    
    # Create output directory
    output_dir = "physics_agent/trajectory_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Initialize engine
    engine = TheoryEngine(black_hole_preset='primordial_mini', verbose=False)
    
    # Trajectory parameters
    r0_si = 10 * engine.length_scale  # 10M
    n_steps = 2000  # More steps for smoother plots
    dtau_si = engine.bh_preset.integration_parameters['dtau_geometric'] * engine.time_scale
    
    print(f"Trajectory parameters:")
    print(f"  Initial radius: 10M")
    print(f"  Steps: {n_steps}")
    print(f"  Time step: {dtau_si:.3e} s\n")
    
    # Process each theory
    successful = []
    failed = []
    
    for theory_name, theory_class, category in ALL_THEORIES:
        print(f"\nProcessing {theory_name} [{category}]...")
        
        try:
            # Initialize theory
            if theory_name == "Kerr":
                theory = theory_class(a=0.0)
            elif theory_name == "Kerr-Newman":
                theory = theory_class(a=0.0, Q=0.0)
            else:
                theory = theory_class()
            
            # Run trajectory (use cache if available for speed)
            hist, solver_tag, _ = engine.run_trajectory(
                theory, r0_si, n_steps, dtau_si,
                verbose=False
            )
            
            if hist is not None and len(hist) > 10:
                # Generate plots
                plot_path, orbit_path = create_trajectory_plots(
                    hist, theory.name, engine, output_dir
                )
                successful.append((theory.name, plot_path, orbit_path))
            else:
                print(f"  ✗ Failed: No valid trajectory")
                failed.append(theory.name)
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
            failed.append(theory_name)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully generated plots for {len(successful)} theories:")
    for name, plot, orbit in successful:
        print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed for {len(failed)} theories:")
        for name in failed:
            print(f"  ✗ {name}")
    
    # Create an index HTML file
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Theory Trajectory Visualizations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .theory { margin-bottom: 40px; border: 1px solid #ccc; padding: 20px; }
        .theory h2 { color: #0066cc; }
        img { max-width: 100%; height: auto; margin: 10px; }
        .plots { display: flex; flex-wrap: wrap; gap: 20px; }
    </style>
</head>
<body>
    <h1>Gravitational Theory Trajectory Visualizations</h1>
    <p>Generated trajectories for test particle orbiting at r=10M around a primordial mini black hole.</p>
"""
    
    for name, plot_path, orbit_path in successful:
        html_content += f"""
    <div class="theory">
        <h2>{name}</h2>
        <div class="plots">
            <div>
                <h3>Full Analysis</h3>
                <img src="{os.path.basename(plot_path)}" alt="{name} trajectory">
            </div>
            <div>
                <h3>Orbital View</h3>
                <img src="{os.path.basename(orbit_path)}" alt="{name} orbit">
            </div>
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nVisualization index created: {index_path}")
    print(f"Open in browser: file://{os.path.abspath(index_path)}")


if __name__ == "__main__":
    main()