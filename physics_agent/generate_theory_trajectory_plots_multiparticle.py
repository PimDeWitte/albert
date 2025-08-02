#!/usr/bin/env python3
"""
Generate trajectory visualizations for all theories and all particles.
Saves outputs to the run directory.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theory_visualizer import TheoryVisualizer
from physics_agent.particle_loader import ParticleLoader
# Import all theories
from physics_agent.theories.newtonian_limit.theory import NewtonianLimit
from physics_agent.theories.defaults.baselines.kerr import Kerr
from physics_agent.theories.defaults.baselines.kerr_newman import KerrNewman
from physics_agent.theories.yukawa.theory import Yukawa
from physics_agent.theories.einstein_teleparallel.theory import EinsteinTeleparallel
from physics_agent.theories.spinor_conformal.theory import SpinorConformal
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild

# Import quantum theories
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
from physics_agent.theories.string.theory import StringTheory
from physics_agent.theories.asymptotic_safety.theory import AsymptoticSafetyTheory
from physics_agent.theories.loop_quantum_gravity.theory import LoopQuantumGravity
from physics_agent.theories.non_commutative_geometry.theory import NonCommutativeGeometry
from physics_agent.theories.twistor_theory.theory import TwistorTheory
from physics_agent.theories.aalto_gauge_gravity.theory import AaltoGaugeGravity
from physics_agent.theories.causal_dynamical_triangulations.theory import CausalDynamicalTriangulations

# Define all theories list
ALL_THEORIES = [
    ('Schwarzschild', Schwarzschild, 'baseline'),
    ('Newtonian Limit', NewtonianLimit, 'classical'),
    ('Kerr', lambda: Kerr(a=0.0), 'classical'),
    ('Kerr-Newman', lambda: KerrNewman(a=0.0, Q=0.0), 'classical'),
    ('Yukawa', Yukawa, 'classical'),
    ('Einstein Teleparallel', EinsteinTeleparallel, 'classical'),
    ('Spinor Conformal', SpinorConformal, 'classical'),
    # Quantum theories
    ('Quantum Corrected', QuantumCorrected, 'quantum'),
    ('String Theory', StringTheory, 'quantum'),
    ('Asymptotic Safety', AsymptoticSafetyTheory, 'quantum'),
    ('Loop Quantum Gravity', LoopQuantumGravity, 'quantum'),
    ('Non-Commutative Geometry', NonCommutativeGeometry, 'quantum'),
    ('Twistor Theory', TwistorTheory, 'quantum'),
    ('Aalto Gauge Gravity', AaltoGaugeGravity, 'quantum'),
    ('Causal Dynamical Triangulations', CausalDynamicalTriangulations, 'quantum'),
]

def create_trajectory_plots(hist, theory_name, particle_name, engine, output_dir):
    """Create comprehensive trajectory plots for a single theory and particle."""
    
    # Convert history to numpy if it's a tensor
    if isinstance(hist, torch.Tensor):
        hist = hist.cpu().numpy()
    
    # Extract trajectory components
    t = hist[:, 0]
    r = hist[:, 1] 
    theta = hist[:, 2]
    phi = hist[:, 3]
    
    # Convert to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Normalize by M
    r_M = r / engine.length_scale
    x_M = x / engine.length_scale
    y_M = y / engine.length_scale
    z_M = z / engine.length_scale
    t_M = t / engine.time_scale * engine.c_si
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'{theory_name} - {particle_name.capitalize()} Trajectory Analysis', fontsize=16, y=0.98)
    
    # 1. 3D trajectory (projected to 2D)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(x_M, y_M, 'b-', linewidth=0.8, alpha=0.7)
    ax1.scatter(x_M[0], y_M[0], color='green', s=100, zorder=5, label='Start')
    ax1.scatter(x_M[-1], y_M[-1], color='red', s=100, zorder=5, label='End')
    
    # Add key radii
    horizon = Circle((0, 0), 2, fill=True, color='black', alpha=0.8, label='Horizon')
    photon_sphere = Circle((0, 0), 3, fill=False, color='orange', linestyle='--', linewidth=2, label='Photon sphere')
    isco = Circle((0, 0), 6, fill=False, color='blue', linestyle=':', linewidth=2, label='ISCO')
    
    ax1.add_patch(horizon)
    ax1.add_patch(photon_sphere)
    ax1.add_patch(isco)
    
    ax1.set_xlabel('X/M')
    ax1.set_ylabel('Y/M')
    ax1.set_title('Orbital Trajectory (X-Y Projection)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Set reasonable limits
    max_r = max(15, np.max(np.sqrt(x_M**2 + y_M**2)) * 1.1)
    ax1.set_xlim(-max_r, max_r)
    ax1.set_ylim(-max_r, max_r)
    
    # 2. Radial evolution
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t_M, r_M, 'b-', linewidth=1.5)
    ax2.axhline(y=2, color='black', linestyle='-', alpha=0.5, label='Horizon')
    ax2.axhline(y=3, color='orange', linestyle='--', alpha=0.5, label='Photon sphere')
    ax2.axhline(y=6, color='blue', linestyle=':', alpha=0.5, label='ISCO')
    ax2.set_xlabel('t/M')
    ax2.set_ylabel('r/M')
    ax2.set_title('Radial Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Angular evolution (r vs phi)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(phi, r_M, 'b-', linewidth=1.5)
    ax3.set_xlabel('φ (radians)')
    ax3.set_ylabel('r/M')
    ax3.set_title('Angular Evolution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Polar plot
    ax4 = fig.add_subplot(2, 3, 4, projection='polar')
    ax4.plot(phi, r_M, 'b-', linewidth=0.8)
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
        ax5.set_ylabel('(dr/dt)/c')
        ax5.set_title('Phase Space')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # 6. Statistics panel
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    n_orbits = np.abs(phi[-1] - phi[0]) / (2 * np.pi)
    dr = np.diff(r)
    dphi = np.diff(phi)
    dtheta = np.diff(theta)
    ds = np.sqrt(dr**2 + r[:-1]**2 * dtheta**2 + r[:-1]**2 * np.sin(theta[:-1])**2 * dphi**2)
    total_distance = np.sum(ds) / engine.length_scale
    
    stats_text = f"""
{theory_name}
Particle: {particle_name.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Black Hole: {engine.bh_preset.name}
Mass: {engine.bh_preset.mass_kg:.2e} kg
Schwarzschild radius: {engine.bh_preset.schwarzschild_radius_m:.2e} m

Trajectory Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Initial radius: {r_M[0]:.2f} M
Final radius: {r_M[-1]:.2f} M
Min radius: {np.min(r_M):.2f} M
Max radius: {np.max(r_M):.2f} M

Orbits completed: {n_orbits:.2f}
Total distance: {total_distance:.1f} M
Duration: {t_M[-1]:.2f} M

Steps: {len(r)}
"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plots
    safe_theory_name = theory_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    safe_particle_name = particle_name.replace(' ', '_')
    
    trajectory_path = os.path.join(output_dir, f'{safe_theory_name}_{safe_particle_name}_trajectory.png')
    plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create a simple orbit plot
    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_M, y_M, 'b-', linewidth=1.5, alpha=0.8)
    ax.scatter(x_M[0], y_M[0], color='green', s=150, zorder=5, label='Start', edgecolor='darkgreen', linewidth=2)
    ax.scatter(x_M[-1], y_M[-1], color='red', s=150, zorder=5, label='End', edgecolor='darkred', linewidth=2)
    
    # Add black hole and key radii
    horizon = Circle((0, 0), 2, fill=True, color='black', alpha=0.9)
    photon_sphere = Circle((0, 0), 3, fill=False, color='orange', linestyle='--', linewidth=2.5)
    isco = Circle((0, 0), 6, fill=False, color='blue', linestyle=':', linewidth=2.5)
    
    ax.add_patch(horizon)
    ax.add_patch(photon_sphere)
    ax.add_patch(isco)
    
    ax.set_xlabel('X/M', fontsize=14)
    ax.set_ylabel('Y/M', fontsize=14)
    ax.set_title(f'{theory_name} - {particle_name.capitalize()} Orbit', fontsize=16, pad=20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    
    max_r = max(15, np.max(np.sqrt(x_M**2 + y_M**2)) * 1.1)
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    
    orbit_path = os.path.join(output_dir, f'{safe_theory_name}_{safe_particle_name}_orbit.png')
    plt.savefig(orbit_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return trajectory_path, orbit_path

def generate_trajectory_visualizations_for_run(run_dir, n_steps=2000):
    """Generate trajectory visualizations for all theories and particles in a run."""
    
    # Create visualizations directory in the run
    viz_dir = os.path.join(run_dir, 'trajectory_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Initialize engine
    engine = TheoryEngine(black_hole_preset='primordial_mini', verbose=False)
    
    # Get particle loader
    particle_loader = ParticleLoader()
    particles = ['electron', 'neutrino', 'photon', 'proton']
    
    # Initialize all theories
    all_theories = []
    for theory_name, theory_class, category in ALL_THEORIES:
        if callable(theory_class):
            theory = theory_class()
        else:
            theory = theory_class()
        all_theories.append(theory)
    
    # Track what we've generated
    generated_files = []
    
    print("="*60)
    print(f"GENERATING TRAJECTORY VISUALIZATIONS")
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {viz_dir}")
    print("="*60)
    
    for theory in all_theories:
        theory_name = theory.name
        print(f"\nProcessing {theory_name}...")
        
        for particle_name in particles:
            print(f"  Generating trajectory for {particle_name}...")
            
            # Get particle
            particle = particle_loader.get_particle(particle_name)
            
            # Check cache first
            cache_dir = os.path.join(os.path.dirname(__file__), 'cache/trajectories/1.0.0/Primordial_Mini_Black_Hole')
            cache_pattern = f"{theory_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}_"
            
            cached_trajectory = None
            if os.path.exists(cache_dir):
                for cache_file in os.listdir(cache_dir):
                    if cache_file.startswith(cache_pattern) and cache_file.endswith('.pt'):
                        cache_path = os.path.join(cache_dir, cache_file)
                        try:
                            data = torch.load(cache_path, map_location='cpu')
                            # Handle different cache formats
                            if isinstance(data, dict) and 'trajectory' in data:
                                cached_trajectory = data['trajectory']
                            elif isinstance(data, torch.Tensor):
                                cached_trajectory = data
                            else:
                                continue
                            
                            steps_in_cache = len(cached_trajectory)
                            if steps_in_cache >= n_steps:
                                print(f"    Found cached trajectory: {steps_in_cache} steps")
                                if steps_in_cache > n_steps:
                                    cached_trajectory = cached_trajectory[:n_steps]
                                    print(f"    Truncating to {n_steps} steps")
                                break
                        except Exception as e:
                            pass  # Silently skip problematic cache files
            
            # If no cached trajectory, compute it
            if cached_trajectory is not None:
                hist = cached_trajectory
            else:
                print(f"    Computing trajectory...")
                # Initial conditions
                r0_si = 10 * engine.length_scale
                dtau_si = engine.bh_preset.integration_parameters['dtau_geometric'] * engine.time_scale
                
                # Run trajectory with particle name
                hist, _, _ = engine.run_trajectory(
                    theory, r0_si, n_steps, dtau_si,
                    use_quantum=hasattr(theory, 'enable_quantum') and theory.enable_quantum,
                    particle_name=particle_name  # Pass the particle name (lowercase)
                )
                
                if hist is None:
                    print(f"    WARNING: Failed to compute trajectory for {particle_name}")
                    continue
            
            # Generate plots
            try:
                traj_path, orbit_path = create_trajectory_plots(
                    hist, theory_name, particle_name, engine, viz_dir
                )
                generated_files.extend([traj_path, orbit_path])
                print(f"    ✓ Saved trajectory and orbit plots")
            except Exception as e:
                print(f"    ERROR: Failed to generate plots: {e}")
    
    # Create HTML index
    print("\nGenerating HTML index...")
    create_visualization_index(viz_dir, generated_files)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Generated {len(generated_files)} visualization files")
    print(f"Output directory: {viz_dir}")
    print(f"Index: {os.path.join(viz_dir, 'index.html')}")
    
    return viz_dir

def create_visualization_index(viz_dir, generated_files):
    """Create an HTML index for all visualizations."""
    
    # Group files by theory and particle
    visualizations = {}
    
    for file_path in generated_files:
        filename = os.path.basename(file_path)
        if '_trajectory.png' in filename:
            parts = filename.replace('_trajectory.png', '').rsplit('_', 1)
            if len(parts) == 2:
                theory, particle = parts
                if theory not in visualizations:
                    visualizations[theory] = {}
                if particle not in visualizations[theory]:
                    visualizations[theory][particle] = {}
                visualizations[theory][particle]['trajectory'] = filename
        elif '_orbit.png' in filename:
            parts = filename.replace('_orbit.png', '').rsplit('_', 1)
            if len(parts) == 2:
                theory, particle = parts
                if theory not in visualizations:
                    visualizations[theory] = {}
                if particle not in visualizations[theory]:
                    visualizations[theory][particle] = {}
                visualizations[theory][particle]['orbit'] = filename
    
    # Generate HTML
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Trajectory Visualizations</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1 { color: #333; }
        .theory-section { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .theory-title { color: #2c3e50; margin-bottom: 15px; }
        .particle-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .particle-card { background: #f8f9fa; padding: 15px; border-radius: 6px; }
        .particle-name { font-weight: bold; color: #34495e; margin-bottom: 10px; }
        .viz-links { display: flex; gap: 15px; }
        .viz-links a { color: #3498db; text-decoration: none; }
        .viz-links a:hover { text-decoration: underline; }
        .timestamp { color: #7f8c8d; font-size: 0.9em; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Trajectory Visualizations</h1>
"""
    
    for theory in sorted(visualizations.keys()):
        theory_display = theory.replace('_', ' ')
        html_content += f'    <div class="theory-section">\n'
        html_content += f'        <h2 class="theory-title">{theory_display}</h2>\n'
        html_content += f'        <div class="particle-grid">\n'
        
        for particle in sorted(visualizations[theory].keys()):
            html_content += f'            <div class="particle-card">\n'
            html_content += f'                <div class="particle-name">{particle.capitalize()}</div>\n'
            html_content += f'                <div class="viz-links">\n'
            
            if 'trajectory' in visualizations[theory][particle]:
                traj_file = visualizations[theory][particle]['trajectory']
                html_content += f'                    <a href="{traj_file}" target="_blank">📊 Full Analysis</a>\n'
            
            if 'orbit' in visualizations[theory][particle]:
                orbit_file = visualizations[theory][particle]['orbit']
                html_content += f'                    <a href="{orbit_file}" target="_blank">🌐 Orbit View</a>\n'
            
            html_content += f'                </div>\n'
            html_content += f'            </div>\n'
        
        html_content += f'        </div>\n'
        html_content += f'    </div>\n'
    
    html_content += f'    <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>\n'
    html_content += '</body>\n</html>'
    
    index_path = os.path.join(viz_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate trajectory visualizations for a run')
    parser.add_argument('run_dir', nargs='?', help='Run directory (creates new one if not specified)')
    parser.add_argument('--steps', type=int, default=2000, help='Number of trajectory steps')
    
    args = parser.parse_args()
    
    # If no run directory specified, create a new one
    if args.run_dir:
        run_dir = args.run_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"runs/trajectory_viz_run_{timestamp}"
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created new run directory: {run_dir}")
    
    generate_trajectory_visualizations_for_run(run_dir, args.steps)