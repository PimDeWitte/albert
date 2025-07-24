"""
Simplified Theory Visualizer for gravitational theory simulations.
Focuses on creating a single, publication-quality 3D trajectory plot.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import torch

# Conditional import for sympy for displaying Lagrangian
try:
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

from physics_agent.base_theory import GravitationalTheory, Tensor

class TheoryVisualizer:
    """
    Handles visualization for gravitational theories, focusing on a single, clear,
    publication-quality 3D plot.
    """
    
    def __init__(self, engine):
        """
        Initialize the visualizer with a reference to the engine.
        
        Args:
            engine: TheoryEngine instance for accessing physics computations
        """
        self.engine = engine
    
    def generate_comparison_plot(self, model: GravitationalTheory, hist: Tensor, baseline_results: dict, 
                                baseline_theories: dict, plot_filename: str, rs_val: float, 
                                validations_dict: dict = None, particle_info: dict = None):
        """
        Generate a single, high-quality 3D trajectory comparison plot.
        
        <reason>chain: This creates the publication-quality visualization with proper event horizon, 
        singularity, and trajectory rendering as shown in the reference yellow image</reason>
        """
        print("  Generating trajectory comparison plot...")
        
        # Use dark background for better contrast
        plt.style.use('dark_background')
        
        # Create figure with 3D axes
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert main theory trajectory - CONVERT TO GEOMETRIC UNITS
        t = self._to_numpy(hist[:, 0]) / self.engine.time_scale
        r = self._to_numpy(hist[:, 1]) / self.engine.length_scale
        phi = self._to_numpy(hist[:, 2])
        
        # <reason>chain: Ensure we have meaningful trajectory data before plotting</reason>
        if len(t) < 2:
            print(f"  Warning: Trajectory too short ({len(t)} points), skipping visualization")
            return
        
        # <reason>chain: Check if r is changing - if not, the trajectory has issues</reason>
        if np.std(r) < 1e-10:  # If r doesn't change at all
            print(f"  WARNING: Trajectory has no radial motion! r_mean={np.mean(r):.3f}, r_std={np.std(r):.3e}")
            print(f"    Initial r={r[0]:.3f}, final r={r[-1]:.3f}")
            print(f"    This suggests a problem with the geodesic solver or initial conditions")
        
        # <reason>chain: Check if this is a quantum theory and add uncertainty visualization</reason>
        theory_category = getattr(model, 'category', 'unknown')
        is_quantum = theory_category == 'quantum' and hasattr(model, 'quantum_integrator')
        
        if is_quantum and model.quantum_integrator is not None:
            # <reason>chain: Visualize quantum uncertainty around the trajectory</reason>
            self._add_quantum_uncertainty_cloud(ax, r, phi, t, model)
        
        # <reason>chain: Auto-calculate proper marker intervals based on trajectory length</reason>
        # If we have a full trajectory, ensure we show markers that end nicely at the end
        total_steps = len(t)
        desired_markers = 50  # Target number of markers
        
        # Calculate step interval to get close to desired markers AND end at the last point
        step_interval = max(1, (total_steps - 1) // desired_markers)
        # Adjust to ensure we include the last point
        if (total_steps - 1) % step_interval != 0:
            # Find the closest interval that divides evenly
            best_interval = step_interval
            min_remainder = (total_steps - 1) % step_interval
            
            # Check nearby intervals
            for test_interval in range(max(1, step_interval - 5), step_interval + 6):
                remainder = (total_steps - 1) % test_interval
                if remainder < min_remainder or (remainder == 0 and min_remainder > 0):
                    best_interval = test_interval
                    min_remainder = remainder
                    if remainder == 0:
                        break
            
            step_interval = best_interval
            
        print(f"  Trajectory has {total_steps} points, using step interval {step_interval} for markers")
        print(f"  Trajectory range: r=[{r.min():.3f}, {r.max():.3f}], phi=[{phi.min():.3f}, {phi.max():.3f}]")
            
        # Convert to Cartesian coordinates
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = t  # Use time as z-axis
        
        # <reason>chain: Plot main theory trajectory with bright red color and thicker line</reason>
        # Make the main trajectory more visible - lower zorder for proper depth
        main_particle_name = "Primary Particle"
        if particle_info and particle_info.get('particle'):
            particle = particle_info['particle']
            solver_tag = particle_info.get('tag', '')
            
            # Build label with particle name and solver type
            label_parts = [f"{particle.name.capitalize()} ({model.name})"]
            
            # Add solver type to label - always include "Solver" and 4D/6D designation
            # <reason>chain: Check for UGM and quantum theories first before defaulting to symmetric/general</reason>
            if 'ugm' in solver_tag:
                # <reason>chain: UGM theories use special UGM solver that handles gauge fields</reason>
                label_parts.append("[UGM Solver (6D)]")
            elif 'quantum_unified' in solver_tag or 'quantum' in solver_tag:
                label_parts.append("[Quantum Path Integral (6D)]")
            elif 'null' in solver_tag or particle.particle_type == 'massless':
                label_parts.append("[Null Geodesic Solver (4D)]")
            elif 'charged' in solver_tag:
                if 'symmetric' in solver_tag:
                    label_parts.append("[Charged Particle Solver (4D)]")
                else:
                    label_parts.append("[Charged Particle Solver (6D)]")
            elif 'symmetric' in solver_tag:
                label_parts.append("[Symmetric Spacetime Solver (4D)]")
            elif 'general' in solver_tag:
                label_parts.append("[General Geodesic Solver (6D)]")
            else:
                # <reason>chain: Add explicit UGM and quantum theory check for proper labeling</reason>
                # Check if this is a UGM theory
                is_ugm = (hasattr(model, 'use_ugm_solver') and model.use_ugm_solver) or \
                         model.__class__.__name__ == 'UnifiedGaugeModel' or \
                         'ugm' in model.__class__.__name__.lower()
                
                if is_ugm:
                    label_parts.append("[UGM Solver (6D)]")
                else:
                    # Infer from theory properties and particle
                    theory_category = getattr(model, 'category', 'unknown')
                    if theory_category == 'quantum':
                        # <reason>chain: Quantum theories now use proper path integral solver</reason>
                        # with classical geodesics as the stationary path
                        label_parts.append("[Quantum Path Integral (6D)]")
                    elif particle and particle.particle_type == 'massless':
                        label_parts.append("[Null Geodesic Solver (4D)]")
                    elif particle and particle.charge != 0:
                        # Check if model is symmetric to determine 4D vs 6D
                        if hasattr(model, 'is_symmetric') and model.is_symmetric:
                            label_parts.append("[Charged Particle Solver (4D)]")
                        else:
                            label_parts.append("[Charged Particle Solver (6D)]")
                    else:
                        # Default based on model symmetry
                        if hasattr(model, 'is_symmetric') and model.is_symmetric:
                            label_parts.append("[Geodesic Solver (4D)]")
                        else:
                            label_parts.append("[Geodesic Solver (6D)]")
            
            main_particle_name = " ".join(label_parts)
        ax.plot(x, y, z, color='red', linewidth=4, alpha=0.9, label=main_particle_name, zorder=5)
        
        # Add step markers along the main trajectory
        marker_indices = list(range(0, total_steps, step_interval))
        # Ensure the last point is included
        if marker_indices[-1] != total_steps - 1:
            marker_indices.append(total_steps - 1)
            
        for i in marker_indices:
            # <reason>chain: Make step markers smaller and less prominent</reason>
            ax.scatter(x[i], y[i], z[i], c='white', s=10, 
                      edgecolors='red', linewidth=0.5, alpha=0.6, zorder=6)
            # Show step numbers at regular intervals
            # Show every 2nd marker instead of every 4th for better visibility
            if i == 0 or i == total_steps - 1 or (i // step_interval) % 2 == 0:
                ax.text(x[i], y[i], z[i], f'{i}', fontsize=6, color='white',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5),
                       zorder=7)
        
        # <reason>chain: Initialize list to store z values for axis limits</reason>
        all_z = [z]
        
        # <reason>chain: Plot additional charged/uncharged particle trajectories if available</reason>
        if particle_info:
            # Get main particle name for comparison
            main_particle_name = particle_info.get('particle', {}).name if particle_info.get('particle') else None
            
            # Plot charged particle trajectory if different from main
            charged_info = particle_info.get('charged_particle')
            if charged_info and charged_info.get('trajectory') is not None:
                charged_hist = charged_info['trajectory']
                charged_particle = charged_info['particle']
                
                # Skip if this is the same as the main trajectory
                if charged_info['name'] != main_particle_name:
                    t_charged = self._to_numpy(charged_hist[:, 0]) / self.engine.time_scale
                    r_charged = self._to_numpy(charged_hist[:, 1]) / self.engine.length_scale
                    phi_charged = self._to_numpy(charged_hist[:, 2])
                    
                    x_charged = r_charged * np.cos(phi_charged)
                    y_charged = r_charged * np.sin(phi_charged)
                    z_charged = t_charged
                    
                    all_z.append(z_charged)
                    
                    # Build label with particle name, charge, and solver type
                    charge_label_parts = [f"{charged_particle.name.capitalize()}"]
                    charge_label_parts.append(f"(q={charged_particle.charge:.1e}C)")
                    # Charged particles always use charged particle solver
                    charge_label_parts.append("[Charged Particle Solver]")
                    charge_label = " ".join(charge_label_parts)
                    ax.plot(x_charged, y_charged, z_charged, color='blue', linewidth=3, alpha=0.8, 
                           label=charge_label, zorder=5)
            
            # Plot uncharged particle trajectory if different from main
            uncharged_info = particle_info.get('uncharged_particle')
            if uncharged_info and uncharged_info.get('trajectory') is not None:
                uncharged_hist = uncharged_info['trajectory']
                uncharged_particle = uncharged_info['particle']
                
                # Skip if this is the same as the main trajectory
                if uncharged_info['name'] != main_particle_name:
                    t_uncharged = self._to_numpy(uncharged_hist[:, 0]) / self.engine.time_scale
                    r_uncharged = self._to_numpy(uncharged_hist[:, 1]) / self.engine.length_scale
                    phi_uncharged = self._to_numpy(uncharged_hist[:, 2])
                    
                    x_uncharged = r_uncharged * np.cos(phi_uncharged)
                    y_uncharged = r_uncharged * np.sin(phi_uncharged)
                    z_uncharged = t_uncharged
                    
                    all_z.append(z_uncharged)
                    
                    # Build label with particle name and solver type
                    neutral_label_parts = [f"{uncharged_particle.name.capitalize()} (neutral)"]
                    # Determine solver type for neutral particle with 4D/6D designation
                    if uncharged_particle.particle_type == 'massless':
                        neutral_label_parts.append("[Null Geodesic Solver (4D)]")
                    else:
                        # Massive neutral particles - check model symmetry
                        if hasattr(model, 'is_symmetric') and model.is_symmetric:
                            neutral_label_parts.append("[Symmetric Spacetime Solver (4D)]")
                        else:
                            neutral_label_parts.append("[General Geodesic Solver (6D)]")
                    neutral_label = " ".join(neutral_label_parts)
                    ax.plot(x_uncharged, y_uncharged, z_uncharged, color='orange', linewidth=3, alpha=0.8,
                           label=neutral_label, zorder=5)
        
        # <reason>chain: Define baseline colors for all baselines</reason>
        baseline_colors = {
            'Schwarzschild': ('gray', '--'),
            'General Relativity': ('gray', '--'),
            'Reissner-Nordström': ('cyan', '-.'),
            'Kerr': ('yellow', ':'),      # Yellow for Kerr
            'Kerr-Newman': ('green', '--'), # Green for Kerr-Newman  
            'DilatonGravity_GHS': ('orange', '-.')
        }
        
        # <reason>chain: Process baselines and add checkpoints</reason>
        print(f"    Processing {len(baseline_results)} baselines for checkpoints...")
        for baseline_name, baseline_hist in baseline_results.items():
            print(f"      Baseline: {baseline_name}")
            if baseline_hist is None or len(baseline_hist) < 2:
                print(f"        Skipping - no data or too short")
                continue
                
            # Convert baseline trajectory to geometric units
            t_b = self._to_numpy(baseline_hist[:, 0]) / self.engine.time_scale
            r_b = self._to_numpy(baseline_hist[:, 1]) / self.engine.length_scale
            phi_b = self._to_numpy(baseline_hist[:, 2])
            
            # Skip if invalid data
            if not np.isfinite(r_b).all() or r_b.max() > 1e10:
                continue
                
            # <reason>chain: Check if baseline has meaningful motion</reason>
            if np.std(r_b) < 1e-10 and np.std(phi_b) < 1e-10:
                print(f"        WARNING: Baseline {baseline_name} has no motion!")
                continue
                
            # Convert to Cartesian
            x_b = r_b * np.cos(phi_b)
            y_b = r_b * np.sin(phi_b)
            z_b = t_b
            
            # <reason>chain: Limit z values to prevent baselines from going too high</reason>
            # Truncate baseline at main trajectory's max time
            max_z_main = z.max() if len(z) > 0 else 100.0
            z_b_truncated = z_b[z_b <= max_z_main * 1.1]  # Allow 10% extra
            if len(z_b_truncated) > 0:
                x_b = x_b[:len(z_b_truncated)]
                y_b = y_b[:len(z_b_truncated)]
                z_b = z_b_truncated
                all_z.append(z_b)
            
            # Find the right style
            style_info = None
            for key, (color, linestyle) in baseline_colors.items():
                if key in baseline_name:
                    style_info = (color, linestyle)
                    break
            
            if not style_info:
                style_info = ('gray', '--')
            
            color, linestyle = style_info
            
            # <reason>chain: Plot all baselines with labels</reason>
            # Plot baseline trajectory with lower alpha and dashed/dotted line - lower zorder
            ax.plot(x_b, y_b, z_b, color=color, linestyle=linestyle, 
                   linewidth=2, alpha=0.4, label=baseline_name, zorder=3)
            
            # <reason>chain: Add checkpoints only for Kerr and Kerr-Newman</reason>
            if 'Kerr' in baseline_name or 'Kerr-Newman' in baseline_name:
                print(f"        Adding checkpoints for {baseline_name}")
                
                # <reason>chain: Calculate checkpoint positions based on percentage of trajectory</reason>
                # Kerr: every 10% (10%, 20%, 30%, ..., 100%)
                # Kerr-Newman: every 5% but skip 10% marks (5%, 15%, 25%, ..., 95%)
                checkpoint_indices = []
                trajectory_length = len(t_b)
                
                if 'Kerr-Newman' in baseline_name:
                    # Kerr-Newman: 5%, 15%, 25%, 35%, 45%, 55%, 65%, 75%, 85%, 95%
                    percentages = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
                else:
                    # Kerr (not Kerr-Newman): 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
                    percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                
                for pct in percentages:
                    idx = int((pct / 100.0) * (trajectory_length - 1))
                    if idx < len(t_b):  # Make sure index is valid
                        checkpoint_indices.append((idx, pct))
                
                print(f"        Checkpoint percentages: {[pct for _, pct in checkpoint_indices]}")
                
                for i, pct in checkpoint_indices:
                    # Skip if beyond array length
                    if i >= len(x_b) or i >= len(y_b) or i >= len(z_b):
                        continue
                        
                    # <reason>chain: Create smaller checkpoint ring at baseline position</reason>
                    # Ring parameters - smaller radius for better visual clarity
                    ring_radius = 2.0  # Smaller radius (was 5.0)
                    tube_radius = 0.2  # Thinner tube (was 0.5)
                    
                    # Create ring at baseline position
                    u = np.linspace(0, 2 * np.pi, 40)  # More points for smoother ring
                    
                    # Ring center at baseline position
                    center_x, center_y, center_z = x_b[i], y_b[i], z_b[i]
                    
                    # <reason>chain: Fix checkpoint z-position to use baseline trajectory time</reason>
                    # Make sure the checkpoint is at the proper time/height
                    print(f"          Checkpoint at {pct}%: idx={i}, z={center_z:.3f}")
                    
                    # Create flat ring in XY plane at given Z height
                    ring_x = center_x + ring_radius * np.cos(u)
                    ring_y = center_y + ring_radius * np.sin(u) 
                    ring_z = np.full_like(u, center_z)
                    
                    # Plot the ring outline with baseline color - higher zorder
                    ring_color = 'yellow' if 'Kerr' in baseline_name and 'Newman' not in baseline_name else 'green'
                    ax.plot(ring_x, ring_y, ring_z, color=ring_color, linewidth=3, alpha=0.8, zorder=4)
                    
                    # Add inner and outer ring edges for 3D effect
                    inner_radius = ring_radius - tube_radius
                    outer_radius = ring_radius + tube_radius
                    
                    # Inner ring
                    inner_x = center_x + inner_radius * np.cos(u)
                    inner_y = center_y + inner_radius * np.sin(u)
                    ax.plot(inner_x, inner_y, ring_z, color=ring_color, linewidth=1.5, alpha=0.5, zorder=4)
                    
                    # Outer ring  
                    outer_x = center_x + outer_radius * np.cos(u)
                    outer_y = center_y + outer_radius * np.sin(u)
                    ax.plot(outer_x, outer_y, ring_z, color=ring_color, linewidth=1.5, alpha=0.5, zorder=4)
                    
                    # <reason>chain: Calculate loss and miss distance at this checkpoint</reason>
                    # Find closest point on main trajectory to this baseline point
                    # Use the same time index for fair comparison
                    if i < len(x) and i < len(y) and i < len(z):
                        # Calculate position differences
                        dx = x[i] - center_x
                        dy = y[i] - center_y
                        dz = z[i] - center_z
                        dr = r[i] - r_b[i]
                        dphi = phi[i] - phi_b[i]
                        
                        # Wrap phi difference to [-pi, pi]
                        while dphi > np.pi:
                            dphi -= 2 * np.pi
                        while dphi < -np.pi:
                            dphi += 2 * np.pi
                        
                        # Calculate various loss metrics
                        euclidean_loss_2d = np.sqrt(dx**2 + dy**2)
                        angular_loss = abs(dphi) * r[i]  # Convert to arc length
                        
                        # Total loss (weighted combination)
                        total_loss = euclidean_loss_2d + 0.1 * angular_loss
                        
                        # <reason>chain: Position labels away from trajectory lines</reason>
                        # Position labels at larger distance to avoid overlapping trajectory
                        
                        # Single loss label - positioned to the side
                        label_angle = -0.3  # Slightly offset from right
                        label_distance = ring_radius + 4.0  # Further away from ring
                        label_x = center_x + label_distance * np.cos(label_angle)
                        label_y = center_y + label_distance * np.sin(label_angle)
                        label_z = center_z + 2.0  # Above the ring plane
                        
                        # Format loss value
                        # <reason>chain: Add explanation that L is loss in Schwarzschild radii</reason>
                        if total_loss < 0.01:
                            loss_str = f'L={total_loss:.3e} Rs'
                        elif total_loss < 1:
                            loss_str = f'L={total_loss:.3f} Rs'
                        else:
                            loss_str = f'L={total_loss:.1f} Rs'
                        
                        # Add baseline name to first checkpoint's loss label
                        if i == checkpoint_indices[0][0]:
                            loss_str = f'{baseline_name}\n{loss_str}\n(L = Loss in Rs)'
                        
                        # Display loss label with baseline color for border
                        ax.text(label_x, label_y, label_z, loss_str, 
                               fontsize=6, color='white', weight='normal',
                               bbox=dict(boxstyle='round,pad=0.1', facecolor='black', 
                                       edgecolor=ring_color, alpha=0.6, linewidth=1),
                               zorder=8)
        
        # <reason>chain: Calculate plot bounds in geometric units - zoomed out for better Z-axis visibility</reason>
        # Focus on the actual trajectory region but zoom out more for visibility
        max_r = max(30.0, max(r.max() for r in [r] + [self._to_numpy(h[:,1])/self.engine.length_scale for h in baseline_results.values() if h is not None and np.isfinite(h[:,1]).all() and h[:,1].max() < 1e20]))
        
        # Set reasonable bounds with more margin for Z-axis visibility
        plot_range = max_r * 1.5  # Increased from 1.2 to 1.5 for better view
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        
        # Z-axis (time) bounds with extra margin
        z_min = min(0, min(arr.min() for arr in all_z if len(arr) > 0 and np.isfinite(arr).all()))
        z_max = max(arr.max() for arr in all_z if len(arr) > 0 and np.isfinite(arr).all())
        if not np.isfinite(z_max) or z_max > 1000:
            z_max = z[-1] * 1.1  # Use main trajectory's max time
        # Add extra margin to Z-axis for label visibility
        z_range = z_max - z_min
        ax.set_zlim(z_min - z_range * 0.1, z_max + z_range * 0.2)  # More space at top for Z-label
        
        # <reason>chain: Event Horizon with proper depth ordering - lower zorder</reason>
        # Draw event horizon at r = 2 (2M in geometric units)
        u_cyl = np.linspace(0, 2 * np.pi, 100)
        z_cyl_vals = np.linspace(z_min, z_max, 50)
        u_grid, z_grid = np.meshgrid(u_cyl, z_cyl_vals)
        x_cyl = 2.0 * np.cos(u_grid)  # r = 2 in geometric units
        y_cyl = 2.0 * np.sin(u_grid)
        
        # Plot cylinder surface with lower zorder for proper depth
        ax.plot_surface(x_cyl, y_cyl, z_grid, color='cyan', alpha=0.3, linewidth=0, 
                       antialiased=True, zorder=1)
        
        # Add event horizon boundary circles at top and bottom
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), np.full_like(theta, z_min), 
                'c-', linewidth=3, alpha=0.9, label='Event Horizon (r=2M)', zorder=2)
        ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), np.full_like(theta, z_max), 
                'c-', linewidth=3, alpha=0.9, zorder=2)
        
        # <reason>chain: Singularity with proper depth ordering - cleaner visualization</reason>
        ax.plot([0, 0], [0, 0], [z_min, z_max], 'y--', linewidth=3, alpha=0.8, 
                label='Singularity (r=0)', zorder=2)
        
        # Add a single central sphere at middle to indicate point singularity - not step counters
        if z_range > 0:
            sphere_z_center = z_min + z_range * 0.5  # Center of time range
            u_sph, v_sph = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            sphere_radius = 0.8  # Slightly larger single sphere
            x_sph = sphere_radius * np.cos(u_sph) * np.sin(v_sph)
            y_sph = sphere_radius * np.sin(u_sph) * np.sin(v_sph)
            z_sph = sphere_z_center + sphere_radius * np.cos(v_sph)
            ax.plot_surface(x_sph, y_sph, z_sph, color='yellow', alpha=0.4, zorder=2)
        
        # <reason>chain: Add initial orbit circle for reference</reason>
        # Show initial circular orbit
        r0_orbit = r[0]
        orbit_theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(r0_orbit * np.cos(orbit_theta), r0_orbit * np.sin(orbit_theta), 
                np.full_like(orbit_theta, z[0]), 
                'g--', linewidth=1, alpha=0.5, label=f'Initial orbit (r={r0_orbit:.1f}M)', zorder=2)
        
        # <reason>chain: Enhanced title with particle information</reason>
        # Build title with particle info if available
        title_parts = [f"Geodesic: {model.name}"]
        
        # <reason>chain: Add Lagrangian type to title</reason>
        if hasattr(model, 'complete_lagrangian') and model.complete_lagrangian is not None:
            if hasattr(model, 'enable_quantum') and model.enable_quantum:
                title_parts.append("[Quantum Lagrangian]")
            else:
                title_parts.append("[Quantum Lagrangian - Classical Mode]")
        elif hasattr(model, 'lagrangian') and model.lagrangian is not None:
            if model.category == 'quantum':
                title_parts.append("[Classical Lagrangian - WARNING: Quantum theory]")
            else:
                title_parts.append("[Classical Lagrangian]")
        
        if particle_info:
            particle = particle_info.get('particle')
            if particle:
                # Add particle type and properties
                particle_desc = f"({particle.name.capitalize()}"
                if particle.particle_type == 'massless':
                    particle_desc += ", massless"
                else:
                    particle_desc += f", m={particle.mass:.2e}kg"
                if particle.charge != 0:
                    particle_desc += f", q={particle.charge:.2e}C"
                particle_desc += f", spin={particle.spin})"
                title_parts.append(particle_desc)
            
            # Add solver type with clear 4D/6D designation
            solver_tag = particle_info.get('tag', '')
            # <reason>chain: Check for UGM and quantum theories first</reason>
            theory_category = getattr(model, 'category', 'unknown')
            if 'ugm' in solver_tag:
                # <reason>chain: UGM theories use special UGM solver</reason>
                title_parts.append("[UGM Solver (6D)]")
            elif 'quantum_unified' in solver_tag or 'quantum' in solver_tag:
                title_parts.append("[Quantum Path Integral (6D)]")
            elif theory_category == 'quantum':
                title_parts.append("[Quantum Theory (6D)]")
            elif 'null' in solver_tag:
                title_parts.append("[Null Geodesic Solver (4D)]")
            elif 'charged' in solver_tag:
                if 'symmetric' in solver_tag:
                    title_parts.append("[Charged Particle Solver (4D)]")
                else:
                    title_parts.append("[Charged Particle Solver (6D)]")
            elif 'symmetric' in solver_tag:
                title_parts.append("[Symmetric Spacetime Solver (4D)]")
            elif 'general' in solver_tag:
                title_parts.append("[General Geodesic Solver (6D)]")
            else:
                # Check if this is a UGM theory
                is_ugm = (hasattr(model, 'use_ugm_solver') and model.use_ugm_solver) or \
                         model.__class__.__name__ == 'UnifiedGaugeModel' or \
                         'ugm' in model.__class__.__name__.lower()
                
                if is_ugm:
                    title_parts.append("[UGM Solver (6D)]")
                else:
                    # Try to infer from particle and model properties
                    if particle and particle.particle_type == 'massless':
                        title_parts.append("[Null Geodesic Solver (4D)]")
                    elif particle and particle.charge != 0:
                        # Check if model is symmetric to determine 4D vs 6D
                        if hasattr(model, 'is_symmetric') and model.is_symmetric:
                            title_parts.append("[Charged Particle Solver (4D)]")
                        else:
                            title_parts.append("[Charged Particle Solver (6D)]")
                    else:
                        # Check model symmetry for default case
                        if hasattr(model, 'is_symmetric') and model.is_symmetric:
                            title_parts.append("[Geodesic Solver (4D)]")
                        else:
                            title_parts.append("[Geodesic Solver (6D)]")
        
        title = '\n'.join(title_parts)
        ax.set_title(title, fontsize=16, pad=20, color='white')
        
                # Fix axis labels with clear unit descriptions
        ax.set_xlabel('\nX (Schwarzschild radii)', fontsize=14, color='white')
        ax.set_ylabel('\nY (Schwarzschild radii)', fontsize=14, color='white')
        # <reason>chain: Make z-label more visible with better positioning and larger font</reason>
        dtau_value = 0.1  # Default time step in geometric units
        ax.set_zlabel(f'Time Units\n(Δτ = {dtau_value})', fontsize=16, color='white', labelpad=25)
        
        # <reason>chain: Removed CONSTRAINTS PASSED text - now in legend</reason>
        
        # <reason>chain: Create clean legend with particle properties</reason>
        # Add particle property box if available
        if particle_info and particle_info.get('particle'):
            particle = particle_info['particle']
            prop_text = [
                f"Particle: {particle.name}",
                f"Type: {particle.particle_type}",
                f"Mass: {particle.mass:.2e} kg",
                f"Charge: {particle.charge:.2e} C",
                f"Spin: {particle.spin}"
            ]
            prop_str = '\n'.join(prop_text)
            ax.text2D(0.98, 0.02, prop_str, transform=ax.transAxes,
                     fontsize=10, color='white', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                              edgecolor='white', alpha=0.8))
        
        # <reason>chain: Create main legend only - portals already visible on left</reason>
        # Get existing legend handles and labels for main trajectories
        handles, labels = ax.get_legend_handles_labels()
        
        # Add custom legend entry for step markers
        from matplotlib.lines import Line2D
        step_marker = Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor='white', markeredgecolor='red', 
                           markeredgewidth=0.5, markersize=8, linestyle='',
                           label='Step markers')
        handles.append(step_marker)
        labels.append('Step markers')
        
        # <reason>chain: Add constraints passed indicator to legend if applicable</reason>
        if validations_dict and 'validations' in validations_dict:
            constraints_passed = all(
                res['flags'].get('overall', 'FAIL') == 'PASS' 
                for res in validations_dict.get('validations', []) if res.get('type') == 'constraint'
            )
            if constraints_passed:
                constraint_marker = Line2D([0], [0], color='lime', linewidth=3, 
                                         label='✓ CONSTRAINTS PASSED')
                handles.append(constraint_marker)
                labels.append('✓ CONSTRAINTS PASSED')
        
        # Main legend for trajectories and basic elements
        main_legend = ax.legend(handles=handles, labels=labels,
                               loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                               fontsize=10, frameon=True, fancybox=True, shadow=True,
                               facecolor='black', edgecolor='white', framealpha=0.9)
        
        # Create separate portal legend
        portal_handles = []
        portal_labels = []
        
        # Add portal indicators (rings) to separate legend
        yellow_portal = Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor='none', markeredgecolor='yellow', 
                              markeredgewidth=3, markersize=12, linestyle='',
                              label='Kerr Checkpoints')
        green_portal = Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='none', markeredgecolor='green', 
                             markeredgewidth=3, markersize=12, linestyle='',
                             label='Kerr-Newman Checkpoints')
        
        portal_handles.extend([yellow_portal, green_portal])
        portal_labels.extend(['Kerr Checkpoints', 'Kerr-Newman Checkpoints'])
        
        # Add portal legend in a different location
        portal_legend = ax.legend(handles=portal_handles, labels=portal_labels,
                                 loc='upper left', bbox_to_anchor=(0.02, 0.98),
                                 fontsize=9, frameon=True, fancybox=True,
                                 facecolor='black', edgecolor='yellow', framealpha=0.8,
                                 title='Baseline Checkpoints', title_fontsize=10)
        portal_legend.get_title().set_color('white')
        
        # Add main legend back (matplotlib removes previous legend when creating new one)
        ax.add_artist(main_legend)
        
        # Set viewing angle for best perspective
        ax.view_init(elev=25, azim=45)
        
        # Grid styling
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', alpha=0.3)
        
        # Pane colors for dark theme
        ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        
        # Save the figure with proper spacing
        # <reason>chain: Adjust margins to ensure z-label is fully visible</reason>
        plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='black', edgecolor='none')
        plt.close(fig)
        
        print(f"  Plot saved to {plot_filename}")
        
    def generate_all_particles_comparison(self, model: GravitationalTheory, particle_trajectories: dict,
                                         baseline_results: dict, baseline_theories: dict,
                                         plot_filename: str, rs_val: float, validations_dict: dict = None):
        """
        Generate a single plot showing trajectories for all particle types from defaults folder.
        
        <reason>chain: Creates a combined visualization showing electron, photon, proton, and neutrino 
        trajectories all on the same plot with the same style as trajectory_comparison.png</reason>
        """
        print("  Generating all-particles comparison plot...")
        
        # Use dark background for better contrast
        plt.style.use('dark_background')
        
        # Create figure with 3D axes
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define particle colors
        # <reason>chain: Use gold for photon for better visibility</reason>
        particle_colors = {
            'electron': 'blue',
            'photon': '#FFD700',  # Gold - more visible than yellow
            'proton': 'red',
            'neutrino': 'green'
        }
        
        # Store all z values for axis limits
        all_z = []
        all_r = []
        
        # Process each particle
        for particle_name, particle_data in particle_trajectories.items():
            hist = particle_data['trajectory']
            particle = particle_data['particle']
            
            if hist is None or len(hist) < 2:
                print(f"    Skipping {particle_name} - no trajectory data")
                continue
                
            # Convert to geometric units
            t = self._to_numpy(hist[:, 0]) / self.engine.time_scale
            r = self._to_numpy(hist[:, 1]) / self.engine.length_scale
            phi = self._to_numpy(hist[:, 2])
            
            # <reason>chain: Check if trajectory has meaningful motion</reason>
            if np.std(r) < 1e-10:
                print(f"    WARNING: {particle_name} has no radial motion! Skipping...")
                continue
                
            # Convert to Cartesian
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = t
            
            all_z.append(z)
            all_r.append(r)
            
            # Get particle color
            color = particle_colors.get(particle_name, 'white')
            
            # Create particle label with properties and solver type
            label_parts = [particle.name]
            if particle.particle_type == 'massless':
                label_parts.append('(massless)')
            else:
                label_parts.append(f'(m={particle.mass:.2e})')
            if particle.charge != 0:
                label_parts.append(f'q={particle.charge:.2e}')
            
            # <reason>chain: Add solver type to clarify which theorem was used</reason>
            # Determine and add solver type with clear 4D/6D designation
            solver_tag = particle_data.get('tag', '')
            theory_category = getattr(model, 'category', 'unknown')
            
            if 'ugm' in solver_tag:
                # <reason>chain: UGM theories use special UGM solver</reason>
                label_parts.append('[UGM Solver (6D)]')
            elif 'quantum_unified' in solver_tag or 'quantum' in solver_tag:
                label_parts.append('[Quantum Path Integral (6D)]')
            elif theory_category == 'quantum':
                label_parts.append('[Quantum Theory (6D)]')
            elif particle.particle_type == 'massless' or 'null' in solver_tag:
                label_parts.append('[Null Geodesic Solver (4D)]')
            elif 'charged' in solver_tag:
                if 'symmetric' in solver_tag:
                    label_parts.append('[Charged Particle Solver (4D)]')
                else:
                    label_parts.append('[Charged Particle Solver (6D)]')
            elif 'symmetric' in solver_tag:
                label_parts.append('[Symmetric Spacetime Solver (4D)]')
            elif 'general' in solver_tag:
                label_parts.append('[General Geodesic Solver (6D)]')
            else:
                # Check if this is a UGM theory
                is_ugm = (hasattr(model, 'use_ugm_solver') and model.use_ugm_solver) or \
                         model.__class__.__name__ == 'UnifiedGaugeModel' or \
                         'ugm' in model.__class__.__name__.lower()
                
                if is_ugm:
                    label_parts.append('[UGM Solver (6D)]')
                else:
                    # Infer from particle properties and model
                    if particle.particle_type == 'massless':
                        label_parts.append('[Null Geodesic Solver (4D)]')
                    elif particle.charge != 0:
                        if hasattr(model, 'is_symmetric') and model.is_symmetric:
                            label_parts.append('[Charged Particle Solver (4D)]')
                        else:
                            label_parts.append('[Charged Particle Solver (6D)]')
                    else:
                        if hasattr(model, 'is_symmetric') and model.is_symmetric:
                            label_parts.append('[Symmetric Spacetime Solver (4D)]')
                        else:
                            label_parts.append('[General Geodesic Solver (6D)]')
            
            label = ' '.join(label_parts)
            
            # Plot trajectory
            ax.plot(x, y, z, color=color, linewidth=3, alpha=0.8, label=label, zorder=5)
            
            # Add start and end markers
            ax.scatter(x[0], y[0], z[0], c=color, s=50, marker='o', 
                      edgecolors='white', linewidth=1, alpha=0.9, zorder=6)
            ax.scatter(x[-1], y[-1], z[-1], c=color, s=50, marker='s', 
                      edgecolors='white', linewidth=1, alpha=0.9, zorder=6)
        
        # Add baseline trajectories (only show one set of portals)
        baseline_plotted = False
        for baseline_name, baseline_hist in baseline_results.items():
            if baseline_hist is None or len(baseline_hist) < 2:
                continue
                
            # Only plot first valid baseline to avoid clutter
            if baseline_plotted:
                break
                
            # Convert baseline trajectory
            t_b = self._to_numpy(baseline_hist[:, 0]) / self.engine.time_scale
            r_b = self._to_numpy(baseline_hist[:, 1]) / self.engine.length_scale
            phi_b = self._to_numpy(baseline_hist[:, 2])
            
            x_b = r_b * np.cos(phi_b)
            y_b = r_b * np.sin(phi_b)
            z_b = t_b
            
            # Plot baseline with dashed line
            ax.plot(x_b, y_b, z_b, color='white', linestyle='--', 
                   linewidth=1.5, alpha=0.4, label=baseline_name, zorder=3)
            
            baseline_plotted = True
        
        # Calculate plot bounds
        if all_r:
            max_r = max(30.0, max(r_arr.max() for r_arr in all_r if np.isfinite(r_arr).all()))
        else:
            max_r = 30.0
            
        plot_range = max_r * 1.2
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        
        # Z-axis bounds
        if all_z:
            z_min = min(0, min(z_arr.min() for z_arr in all_z if len(z_arr) > 0 and np.isfinite(z_arr).all()))
            z_max = max(z_arr.max() for z_arr in all_z if len(z_arr) > 0 and np.isfinite(z_arr).all())
        else:
            z_min, z_max = 0, 100
            
        ax.set_zlim(z_min, z_max)
        
        # Draw event horizon
        u_cyl = np.linspace(0, 2 * np.pi, 100)
        z_cyl_vals = np.linspace(z_min, z_max, 50)
        u_grid, z_grid = np.meshgrid(u_cyl, z_cyl_vals)
        x_cyl = 2.0 * np.cos(u_grid)
        y_cyl = 2.0 * np.sin(u_grid)
        
        ax.plot_surface(x_cyl, y_cyl, z_grid, color='cyan', alpha=0.2, linewidth=0, 
                       antialiased=True, zorder=1)
        
        # Add singularity
        ax.plot([0, 0], [0, 0], [z_min, z_max], 'y--', linewidth=3, alpha=0.8, 
                label='Singularity (r=0)', zorder=2)
        
        # Set title
        ax.set_title(f'Multi-Particle Geodesics: {model.name}', fontsize=18, pad=20, color='white')
        
        # Axis labels
        ax.set_xlabel('\nX (Schwarzschild radii)', fontsize=14)
        ax.set_ylabel('\nY (Schwarzschild radii)', fontsize=14)
        # <reason>chain: Make z-label more visible with better positioning and larger font</reason>
        dtau_value = 0.1  # Default time step in geometric units
        ax.set_zlabel(f'Time Units\n(Δτ = {dtau_value})', fontsize=16, labelpad=25)
        
        # Legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                 fontsize=10, frameon=True, fancybox=True,
                 facecolor='black', edgecolor='white', framealpha=0.9)
        
        # Grid and viewing angle
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=25, azim=45)
        
        # Pane colors
        ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        
        # Save
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.1, hspace=0.35, wspace=0.25)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"  All-particles plot saved to: {plot_filename}")
    
    def _get_validation_status(self, validations_dict: dict) -> str:
        """
        Get summary status of constraint validations.
        """
        if not validations_dict or 'validations' not in validations_dict:
            return 'N/A'
        
        constraints = [v for v in validations_dict['validations'] if v.get('type') == 'constraint']
        if not constraints:
            return 'No constraints'
        
        passed = all(v['flags']['overall'] == 'PASS' for v in constraints)
        return 'All Passed' if passed else 'Failed' 

    def generate_multi_particle_grid(self, model: GravitationalTheory, particle_results: dict, 
                                   baseline_results: dict, baseline_theories: dict,
                                   plot_filename: str, rs_val: float, validations_dict: dict = None):
        """
        <reason>chain: Generate a grid visualization showing trajectories for multiple particles</reason>
        Shows Lagrangian, solver type, and particle info for each subplot.
        """
        print("  Generating multi-particle grid visualization...")
        
        # Use dark background for better contrast
        plt.style.use('dark_background')
        
        # Determine grid size based on number of particles
        n_particles = len(particle_results)
        if n_particles <= 2:
            rows, cols = 1, 2
        elif n_particles <= 4:
            rows, cols = 2, 2
        else:
            rows = int(np.ceil(np.sqrt(n_particles)))
            cols = int(np.ceil(n_particles / rows))
        
        # Create figure with subplots - increased vertical spacing and figure size
        fig = plt.figure(figsize=(cols * 9, rows * 10), facecolor='black')
        
        # Main title - moved up with more padding from subplots
        fig.suptitle(f'Multi-Particle Trajectories: {model.name}', 
                    fontsize=24, color='white', y=0.98, weight='bold')
        
        # <reason>chain: First, generate baseline trajectory for each particle type</reason>
        # Get Schwarzschild baseline theory
        baseline_theory = None
        for name, theory in baseline_theories.items():
            if 'Schwarzschild' in name or 'General Relativity' in name:
                baseline_theory = theory
                break
        
        if baseline_theory is None:
            # Fallback: use first baseline
            baseline_theory = list(baseline_theories.values())[0] if baseline_theories else None
        
        # Generate particle-specific baselines
        particle_baselines = {}
        if baseline_theory is not None:
            print("    Generating particle-specific baselines...")
            for particle_name in particle_results.keys():
                if particle_results[particle_name]['trajectory'] is not None:
                    # Get initial conditions from the particle's trajectory
                    particle_hist = particle_results[particle_name]['trajectory']
                    r0 = particle_hist[0, 1]
                    
                    # Generate baseline for this particle
                    try:
                        baseline_hist, _, _ = self.engine.run_trajectory(
                            baseline_theory, r0, len(particle_hist), 
                            self.engine.dtau if hasattr(self.engine, 'dtau') else 0.001,
                            particle_name=particle_name,
                            no_cache=False
                        )
                        particle_baselines[particle_name] = baseline_hist
                    except:
                        # Fallback to regular baseline
                        particle_baselines[particle_name] = baseline_results.get('General Relativity (Schwarzschild)')
        
        # Process each particle
        for idx, (particle_name, result) in enumerate(particle_results.items()):
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            ax.set_facecolor('black')
            
            # Skip if trajectory failed
            if result['trajectory'] is None:
                ax.text(0.5, 0.5, 0.5, f'Failed: {particle_name}', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=16, color='red')
                ax.set_title(f'{particle_name.capitalize()} - FAILED', 
                           fontsize=14, color='red', pad=30)
                continue
            
            # Extract trajectory data
            hist = result['trajectory']
            t = self._to_numpy(hist[:, 0]) / self.engine.time_scale  # Convert to geometric units
            r = self._to_numpy(hist[:, 1]) / self.engine.length_scale
            phi = self._to_numpy(hist[:, 2])
            
            # Convert to Cartesian
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = t  # Time as z-axis
            
            # Plot trajectory with particle-specific color
            # <reason>chain: Use gold for photon for better visibility</reason>
            particle_colors = {
                'electron': 'blue',
                'photon': '#FFD700',  # Gold - more visible than yellow
                'proton': 'red',
                'neutrino': 'green'
            }
            color = particle_colors.get(particle_name, 'white')
            
            # <reason>chain: Extract particle object from result dictionary</reason>
            particle = result.get('particle')
            
            # <reason>chain: Build label with particle name and solver type for clarity</reason>
            # Determine solver type for the legend with clear 4D/6D designation
            solver_tag = result.get('tag', '')
            solver_label = ""
            theory_category = getattr(model, 'category', 'unknown')
            
            if 'ugm' in solver_tag:
                # <reason>chain: UGM theories use special UGM solver</reason>
                solver_label = " [UGM Solver (6D)]"
            elif 'quantum_unified' in solver_tag:
                solver_label = " [Quantum Path Integral + Classical Solver]"
            elif theory_category == 'quantum':
                solver_label = " [Quantum Theory (6D)]"
            elif particle and particle.particle_type == 'massless' or 'null' in solver_tag:
                solver_label = " [Null Geodesic Solver (4D)]"
            elif 'charged' in solver_tag:
                if 'symmetric' in solver_tag:
                    solver_label = " [Charged Particle Solver (4D)]"
                else:
                    solver_label = " [Charged Particle Solver (6D)]"
            elif 'symmetric' in solver_tag:
                solver_label = " [Symmetric Spacetime Solver (4D)]"
            elif 'general' in solver_tag:
                solver_label = " [General Geodesic Solver (6D)]"
            else:
                # Check if this is a UGM theory
                is_ugm = (hasattr(model, 'use_ugm_solver') and model.use_ugm_solver) or \
                         model.__class__.__name__ == 'UnifiedGaugeModel' or \
                         'ugm' in model.__class__.__name__.lower()
                
                if is_ugm:
                    solver_label = " [UGM Solver (6D)]"
                else:
                    # Infer from particle properties and model
                    if particle and particle.particle_type == 'massless':
                        solver_label = " [Null Geodesic Solver (4D)]"
                    elif particle and particle.charge != 0:
                        if hasattr(model, 'is_symmetric') and model.is_symmetric:
                            solver_label = " [Charged Particle Solver (4D)]"
                        else:
                            solver_label = " [Charged Particle Solver (6D)]"
                    else:
                        if hasattr(model, 'is_symmetric') and model.is_symmetric:
                            solver_label = " [Symmetric Spacetime Solver (4D)]"
                        else:
                            solver_label = " [General Geodesic Solver (6D)]"
            
            # Main trajectory with thicker line
            ax.plot(x, y, z, color=color, linewidth=3.5, alpha=0.9, 
                   label=f'{particle_name.capitalize()}{solver_label}')
            
            # Add particle-specific baseline trajectory
            baseline_hist = particle_baselines.get(particle_name)
            if baseline_hist is None:
                # Fallback to general baseline
                baseline_hist = baseline_results.get('General Relativity (Schwarzschild)')
            
            if baseline_hist is not None and len(baseline_hist) > 0:
                t_b = self._to_numpy(baseline_hist[:, 0]) / self.engine.time_scale
                r_b = self._to_numpy(baseline_hist[:, 1]) / self.engine.length_scale
                phi_b = self._to_numpy(baseline_hist[:, 2])
                
                # Ensure same length for comparison
                min_len = min(len(t), len(t_b))
                t_b = t_b[:min_len]
                r_b = r_b[:min_len]
                phi_b = phi_b[:min_len]
                
                x_b = r_b * np.cos(phi_b)
                y_b = r_b * np.sin(phi_b)
                z_b = t_b
                
                # Plot baseline with distinctive style
                # <reason>chain: Use actual baseline name instead of generic "GR Baseline"</reason>
                ax.plot(x_b, y_b, z_b, color='white', linewidth=2, alpha=0.7,
                       linestyle='--', label=baseline_theory.name if hasattr(baseline_theory, 'name') else 'Schwarzschild')
            
            # Add event horizon
            u_cyl = np.linspace(0, 2 * np.pi, 50)
            z_cyl_vals = np.linspace(z.min(), z.max(), 30)
            u_grid, z_grid = np.meshgrid(u_cyl, z_cyl_vals)
            x_cyl = 2.0 * np.cos(u_grid)
            y_cyl = 2.0 * np.sin(u_grid)
            ax.plot_surface(x_cyl, y_cyl, z_grid, color='cyan', alpha=0.15, linewidth=0)
            
            # Add singularity
            ax.plot([0, 0], [0, 0], [z.min(), z.max()], 'y--', linewidth=2, alpha=0.6)
            
            # Set axis limits
            max_r = max(30.0, r.max() * 1.2)
            ax.set_xlim(-max_r, max_r)
            ax.set_ylim(-max_r, max_r)
            ax.set_zlim(z.min(), z.max())
            
            # Particle-specific title with better spacing
            particle = result.get('particle')
            if particle:
                title_parts = [f'{particle.name.capitalize()}']
                if particle.particle_type == 'massless':
                    title_parts.append('(massless)')
                else:
                    title_parts.append(f'(m={particle.mass:.2e} kg)')
                
                # Add charge info
                if particle.charge != 0:
                    title_parts.append(f'q={particle.charge:.2e} C')
                
                title = ' '.join(title_parts)
            else:
                title = f'{particle_name.capitalize()}'
            # Increased padding to avoid overlap with main title
            ax.set_title(title, fontsize=14, color='white', pad=35)
            
            # Add legend for this subplot - moved to avoid overlap
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
                     bbox_to_anchor=(1.0, 0.95), frameon=True,
                     facecolor='black', edgecolor='white')
            
            # Add info text box with better formatting
            info_text = []
            
            # <reason>chain: Clearly indicate if theory uses quantum or classical Lagrangian</reason>
            # Add theory Lagrangian type
            if hasattr(model, 'complete_lagrangian') and model.complete_lagrangian is not None:
                info_text.append('Lagrangian: Quantum')
                # Check if theory is using quantum corrections
                if hasattr(model, 'enable_quantum') and model.enable_quantum:
                    info_text.append('Quantum: Enabled')
                else:
                    info_text.append('Quantum: Disabled')
            else:
                info_text.append('Lagrangian: Classical')
            
            # Add particle type
            if particle:
                if particle.particle_type == 'massless':
                    info_text.append('Particle Type: Massless')
                elif particle.charge != 0:
                    info_text.append('Particle Type: Charged')
                else:
                    info_text.append('Particle Type: Massive')
            
            # Add solver type (without brackets since already in label)
            if solver_label:
                info_text.append(f'Solver: {solver_label.strip()[1:-1]}')
            
            # Add spin info if available
            if particle and particle.spin:
                info_text.append(f'Spin: {particle.spin}')
            
            # Add validation status
            if validations_dict:
                status = self._get_validation_status(validations_dict)
                info_text.append(f'Validation: {status}')
            
            # Create text box with better positioning
            props = dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white')
            ax.text2D(0.02, 0.02, '\n'.join(info_text), transform=ax.transAxes, fontsize=8,
                     verticalalignment='bottom', bbox=props, color='white')
            
            # Plot main trajectory
            ax.plot(x, y, z, color=color, linewidth=3.5, alpha=0.9, 
                   label=f'{particle_name.capitalize()}{solver_label}')
            
            # Plot particle-specific baseline if available
            baseline_hist = particle_baselines.get(particle_name)
            if baseline_hist is not None:
                t_b = self._to_numpy(baseline_hist[:, 0]) / self.engine.time_scale
                r_b = self._to_numpy(baseline_hist[:, 1]) / self.engine.length_scale
                phi_b = self._to_numpy(baseline_hist[:, 2])
                
                x_b = r_b * np.cos(phi_b)
                y_b = r_b * np.sin(phi_b)
                z_b = t_b
                
                # <reason>chain: Use actual baseline name from theory</reason>
                # Get the baseline theory object
                baseline_theory_obj = baseline_theories.get('Schwarzschild') or baseline_theories.get('General Relativity')
                if baseline_theory_obj and hasattr(baseline_theory_obj, 'name'):
                    baseline_name = baseline_theory_obj.name
                else:
                    baseline_name = 'Schwarzschild'
                ax.plot(x_b, y_b, z_b, color='gray', linewidth=1.5, alpha=0.5,
                       label=baseline_name, linestyle='--')
            
            # Plot event horizon
            horizon_radius = 2.0  # Geometric units
            u = np.linspace(0, 2*np.pi, 100)
            v = np.linspace(0, np.max(z)*1.1, 50)  # Extend to max z
            U, V = np.meshgrid(u, v)
            X_h = horizon_radius * np.cos(U)
            Y_h = horizon_radius * np.sin(U)
            Z_h = V
            ax.plot_surface(X_h, Y_h, Z_h, color='white', alpha=0.3, rstride=5, cstride=5)
            
            # Set axis limits
            r_max = max(np.max(r), horizon_radius * 3) * 1.1
            ax.set_xlim(-r_max, r_max)
            ax.set_ylim(-r_max, r_max)
            ax.set_zlim(0, np.max(z) * 1.1)
            
            # Labels
            ax.set_xlabel('X (M)', color='white', fontsize=10, labelpad=15)
            ax.set_ylabel('Y (M)', color='white', fontsize=10, labelpad=15)
            ax.set_zlabel('Time (M)', color='white', fontsize=10, labelpad=15)
            
            # Customize view
            ax.view_init(elev=20, azim=-60)
            ax.dist = 8  # Adjust zoom
            
            # Custom tick formatting
            ax.tick_params(axis='x', colors='white', pad=5)
            ax.tick_params(axis='y', colors='white', pad=5)
            ax.tick_params(axis='z', colors='white', pad=5)
            
            # Hide grid and axes background
            ax.grid(False)
            ax.xaxis.pane.set_edgecolor('black')
            ax.yaxis.pane.set_edgecolor('black')
            ax.zaxis.pane.set_edgecolor('black')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            # Add subtle grid lines
            ax.xaxis._axinfo['grid'].update({'linewidth':0.2, 'color':'gray'})
            ax.yaxis._axinfo['grid'].update({'linewidth':0.2, 'color':'gray'})
            ax.zaxis._axinfo['grid'].update({'linewidth':0.2, 'color':'gray'})
       
        # Add main title with theory info
        fig.suptitle(f'{model.name} - Particle Trajectories (Geometric Units)', fontsize=24, color='white', y=0.96)
       
        # Add subtitle with validation info
        if validations_dict:
            valid_status = self._get_validation_status(validations_dict)
            fig.text(0.5, 0.92, f'Validation Status: {valid_status}', ha='center', fontsize=16, color='white')
       
        # Adjust layout
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
       
        # Save figure
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close(fig)
       
        print(f"    Generated multi-particle grid: {plot_filename}")

    def generate_unified_multi_particle_plot(self, model: GravitationalTheory, particle_trajectories: dict,
                                           baseline_results: dict, baseline_theories: dict,
                                           plot_filename: str, rs_val: float, validations_dict: dict = None):
        """
        Generate a unified 3D plot with all particle trajectories in one figure.
        
        <reason>chain: Create a single plot showing all particles for easier comparison</reason>
        <reason>chain: Use time as z-axis and Cartesian coordinates for clarity</reason>
        <reason>chain: Add particle-specific labels with properties</reason>
        <reason>chain: Include solver type in labels</reason>
        <reason>chain: Add validation status to subtitle</reason>
        <reason>chain: Use particle-specific colors for distinction</reason>
        <reason>chain: Add event horizon cylinder</reason>
        <reason>chain: Add info box with theory details</reason>
        """
        print(f"    Generating unified multi-particle plot for {model.name}...")
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(16, 12), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        
        # Define particle colors matching the standard scheme
        # <reason>chain: Change photon color to gold/orange for better visibility on dark background</reason>
        particle_colors = {
            'electron': 'blue',      # Negative charge
            'photon': '#FFD700',     # Gold color - more visible than yellow on black
            'proton': 'red',         # Positive charge
            'neutrino': 'green'      # Nearly massless, neutral
        }
        
        # Store all z values and radii for axis limits
        all_z = []
        all_r = []
        successful_particles = []
        failed_particles = []
        
        # <reason>chain: First check what data we have</reason>
        print(f"    Particle trajectories available: {list(particle_trajectories.keys())}")
        
        # Process each particle trajectory
        for particle_name, particle_data in particle_trajectories.items():
            print(f"    Processing {particle_name}...")
            
            # Check if particle_data is properly structured
            if not isinstance(particle_data, dict):
                print(f"      ERROR: Invalid data structure for {particle_name}")
                failed_particles.append(particle_name)
                continue
                
            hist = particle_data.get('trajectory')
            particle = particle_data.get('particle')
            
            if hist is None:
                print(f"      WARNING: {particle_name} has no trajectory (computation may have failed)")
                failed_particles.append(particle_name)
                continue
                
            if len(hist) < 2:
                print(f"      WARNING: {particle_name} trajectory too short ({len(hist)} points)")
                failed_particles.append(particle_name)
                continue
                
            if particle is None:
                print(f"      ERROR: {particle_name} has no particle object")
                failed_particles.append(particle_name)
                continue
            
            # Convert to geometric units
            t = self._to_numpy(hist[:, 0]) / self.engine.time_scale
            r = self._to_numpy(hist[:, 1]) / self.engine.length_scale
            phi = self._to_numpy(hist[:, 2])
            
            # <reason>chain: Diagnostic info for troubleshooting</reason>
            print(f"      {particle_name} trajectory: {len(t)} points")
            print(f"      r range: [{r.min():.3f}, {r.max():.3f}] (geometric units)")
            print(f"      phi range: [{phi.min():.3f}, {phi.max():.3f}] rad")
            print(f"      t range: [{t.min():.3f}, {t.max():.3f}] (geometric units)")
            
            # <reason>chain: Check if the trajectory has meaningful motion</reason>
            if np.std(r) < 1e-6:  # Very small radial variation
                print(f"      WARNING: {particle_name} has minimal radial motion (std={np.std(r):.3e})")
                # Still plot it to show the issue
            
            # Convert to Cartesian coordinates
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = t  # Use time as z-axis
            
            all_z.append(z)
            all_r.append(r)
            successful_particles.append(particle_name)
            
            # Get particle color
            color = particle_colors.get(particle_name, 'white')
            
            # Create detailed particle label with exact properties from defaults
            label_parts = [f"{particle.name}"]
            if particle.particle_type == 'massless':
                label_parts.append("(massless)")
            else:
                # Show exact mass from defaults file
                if particle.mass < 1e-30:
                    label_parts.append(f"(m={particle.mass:.1e}kg)")
                else:
                    label_parts.append(f"(m={particle.mass:.2e}kg)")
            
            # Add charge if non-zero
            if particle.charge != 0:
                label_parts.append(f"q={particle.charge:.2e}C")
            
            # Add spin
            label_parts.append(f"s={particle.spin}")
            
            # <reason>chain: Add solver type to make it clear which theorem was used</reason>
            # Determine and add solver type with clear 4D/6D designation
            solver_tag = particle_data.get('tag', '')
            solver_label = ""
            theory_category = getattr(model, 'category', 'unknown')
            
            if 'ugm' in solver_tag:
                # <reason>chain: UGM theories use special UGM solver</reason>
                solver_label = " [UGM Solver (6D)]"
            elif 'quantum_unified' in solver_tag:
                solver_label = " [Quantum Path Integral + Classical Solver]"
            elif theory_category == 'quantum':
                solver_label = " [Quantum Theory (6D)]"
            elif particle.particle_type == 'massless' or 'null' in solver_tag:
                solver_label = " [Null Geodesic Solver (4D)]"
            elif 'charged' in solver_tag:
                if 'symmetric' in solver_tag:
                    solver_label = " [Charged Particle Solver (4D)]"
                else:
                    solver_label = " [Charged Particle Solver (6D)]"
            elif 'symmetric' in solver_tag:
                solver_label = " [Symmetric Spacetime Solver (4D)]"
            elif 'general' in solver_tag:
                solver_label = " [General Geodesic Solver (6D)]"
            else:
                # Check if this is a UGM theory
                is_ugm = (hasattr(model, 'use_ugm_solver') and model.use_ugm_solver) or \
                         model.__class__.__name__ == 'UnifiedGaugeModel' or \
                         'ugm' in model.__class__.__name__.lower()
                
                if is_ugm:
                    solver_label = " [UGM Solver (6D)]"
                else:
                    # Infer from particle properties and model
                    if particle.particle_type == 'massless':
                        solver_label = " [Null Geodesic Solver (4D)]"
                    elif particle.charge != 0:
                        if hasattr(model, 'is_symmetric') and model.is_symmetric:
                            solver_label = " [Charged Particle Solver (4D)]"
                        else:
                            solver_label = " [Charged Particle Solver (6D)]"
                    else:
                        if hasattr(model, 'is_symmetric') and model.is_symmetric:
                            solver_label = " [Symmetric Spacetime Solver (4D)]"
                        else:
                            solver_label = " [General Geodesic Solver (6D)]"
            
            # Add solver to label
            label_parts.append(solver_label.strip())
            
            # Create final label
            label = ' '.join(label_parts)
            
            # Plot main trajectory with thicker line and particle-specific color
            ax.plot(x, y, z, color=color, linewidth=3.0, alpha=0.9, 
                   label=label, zorder=10)  # Higher zorder to ensure visibility
            
            # Add subtle glow effect with thinner line
            ax.plot(x, y, z, color=color, linewidth=5.0, alpha=0.3, zorder=9)
            
            print(f"      Successfully plotted {particle_name} trajectory")
        
        # <reason>chain: Add a reference baseline for comparison</reason>
        # Find the Schwarzschild baseline for reference
        schwarzschild_baseline = None
        for baseline_name, baseline_hist in baseline_results.items():
            if 'Schwarzschild' in baseline_name and baseline_hist is not None and len(baseline_hist) > 2:
                schwarzschild_baseline = baseline_hist
                schwarzschild_name = baseline_name
                break
                
        if schwarzschild_baseline is not None:
            # Convert baseline trajectory
            t_b = self._to_numpy(schwarzschild_baseline[:, 0]) / self.engine.time_scale
            r_b = self._to_numpy(schwarzschild_baseline[:, 1]) / self.engine.length_scale
            phi_b = self._to_numpy(schwarzschild_baseline[:, 2])
            
            # Limit to same time range as particles
            if all_z:
                max_particle_time = max(z_arr.max() for z_arr in all_z)
                time_mask = t_b <= max_particle_time * 1.1
                t_b = t_b[time_mask]
                r_b = r_b[time_mask]
                phi_b = phi_b[time_mask]
            
            x_b = r_b * np.cos(phi_b)
            y_b = r_b * np.sin(phi_b)
            z_b = t_b
            
            # Plot baseline trajectory more subtly
            ax.plot(x_b, y_b, z_b, color='gray', linewidth=1.0, alpha=0.4, 
                   linestyle='--', label=schwarzschild_name, zorder=2)
            
            print(f"    Added {schwarzschild_name} baseline for reference")
        
        # Calculate plot bounds
        if all_r:
            max_r = max(30.0, max(r_arr.max() for r_arr in all_r if np.isfinite(r_arr).all()))
            min_r = min(r_arr.min() for r_arr in all_r if np.isfinite(r_arr).all())
            print(f"    Radial range across all particles: [{min_r:.3f}, {max_r:.3f}]")
        else:
            max_r = 30.0
            print("    WARNING: No valid radial data found!")
        
        plot_range = max_r * 1.2
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        
        # Z-axis (time) bounds
        if all_z:
            z_min = min(0, min(z_arr.min() for z_arr in all_z if len(z_arr) > 0 and np.isfinite(z_arr).all()))
            z_max = max(z_arr.max() for z_arr in all_z if len(z_arr) > 0 and np.isfinite(z_arr).all())
            if not np.isfinite(z_max) or z_max > 1000:
                z_max = 100  # Default max time
            print(f"    Time range across all particles: [{z_min:.3f}, {z_max:.3f}]")
        else:
            z_min, z_max = 0, 100
            print("    WARNING: No valid time data found!")
        
        ax.set_zlim(z_min, z_max)
        
        # Event Horizon
        u_cyl = np.linspace(0, 2 * np.pi, 100)
        z_cyl_vals = np.linspace(z_min, z_max, 50)
        u_grid, z_grid = np.meshgrid(u_cyl, z_cyl_vals)
        x_cyl = 2.0 * np.cos(u_grid)  # r = 2M
        y_cyl = 2.0 * np.sin(u_grid)
        
        ax.plot_surface(x_cyl, y_cyl, z_grid, color='cyan', alpha=0.25, linewidth=0, 
                       antialiased=True, zorder=1)
        
        # Add event horizon boundary circles
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), np.full_like(theta, z_min), 
                'c-', linewidth=2, alpha=0.8, zorder=2)
        ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), np.full_like(theta, z_max), 
                'c-', linewidth=2, alpha=0.8, zorder=2)
        
        # Singularity - thinner with 3D spheres
        ax.plot([0, 0], [0, 0], [z_min, z_max], 'y--', linewidth=1.5, alpha=0.6, zorder=2)
        
        # Add small spheres along singularity
        if z_max - z_min > 0:
            sphere_interval = (z_max - z_min) / 8
            for sphere_z in np.arange(z_min, z_max + sphere_interval/2, sphere_interval):
                u_sph, v_sph = np.mgrid[0:2*np.pi:15j, 0:np.pi:8j]
                sphere_radius = 0.3
                x_sph = sphere_radius * np.cos(u_sph) * np.sin(v_sph)
                y_sph = sphere_radius * np.sin(u_sph) * np.sin(v_sph)
                z_sph = sphere_z + sphere_radius * np.cos(v_sph)
                ax.plot_surface(x_sph, y_sph, z_sph, color='yellow', alpha=0.25, zorder=2)
        
        # Title with theory name and particle status
        title_lines = [f'Multi-Particle Geodesics: {model.name}']
        
        # <reason>chain: Add Lagrangian type to title for clarity</reason>
        if hasattr(model, 'complete_lagrangian') and model.complete_lagrangian is not None:
            if hasattr(model, 'enable_quantum') and model.enable_quantum:
                title_lines.append('[Quantum Lagrangian - Quantum Path Integral Active]')
            else:
                title_lines.append('[Quantum Lagrangian - Classical Path Only]')
        elif hasattr(model, 'lagrangian') and model.lagrangian is not None:
            if model.category == 'quantum':
                title_lines.append('[WARNING: Quantum Theory Using Classical Lagrangian]')
            else:
                title_lines.append('[Classical Lagrangian]')
        else:
            title_lines.append('[No Lagrangian Defined]')
        
        # Report on particle success/failure
        if successful_particles:
            title_lines.append(f'Successfully plotted: {", ".join(successful_particles)}')
        if failed_particles:
            title_lines.append(f'Failed to plot: {", ".join(failed_particles)}')
        title_lines.append('All Standard Model Particles with Exact Default Values')
        
        ax.set_title('\n'.join(title_lines),
                    fontsize=18, pad=20, color='white', weight='bold')
        
        # Axis labels
        ax.set_xlabel('\nX (Schwarzschild radii)', fontsize=14, color='white')
        ax.set_ylabel('\nY (Schwarzschild radii)', fontsize=14, color='white')
        # <reason>chain: Make z-label more visible with better positioning and larger font</reason>
        dtau_value = 0.1  # Default time step in geometric units
        ax.set_zlabel(f'Time Units\n(Δτ = {dtau_value})', fontsize=16, color='white', labelpad=25)
        
        # Custom legend with additional labels
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        
        # Add event horizon and singularity to legend if not already there
        from matplotlib.lines import Line2D
        if 'Event Horizon' not in legend_labels:
            legend_handles.append(Line2D([0], [0], color='cyan', linewidth=2, label='Event Horizon (r=2M)'))
        if 'Singularity' not in legend_labels:
            legend_handles.append(Line2D([0], [0], color='yellow', linestyle='--', linewidth=2, label='Singularity (r=0)'))
        
        # <reason>chain: Fix legend to properly match handles and labels</reason>
        ax.legend(handles=legend_handles, labels=None,
                 loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                 fontsize=9, frameon=True, fancybox=True,
                 facecolor='black', edgecolor='white', framealpha=0.9,
                 ncol=1)  # Single column for cleaner look
        
        # Add particle summary box with status
        summary_lines = [
            "Standard Model Particles:",
            f"• Electron: m={9.11e-31}kg, q=-1.60e-19C" + (" ✓" if "electron" in successful_particles else " ✗"),
            f"• Photon: massless, q=0, spin=1" + (" ✓" if "photon" in successful_particles else " ✗"),
            f"• Proton: m={1.67e-27}kg, q=+1.60e-19C" + (" ✓" if "proton" in successful_particles else " ✗"), 
            f"• Neutrino: m={3.2e-37}kg, q=0" + (" ✓" if "neutrino" in successful_particles else " ✗")
        ]
        summary_text = '\n'.join(summary_lines)
        
        ax.text2D(0.98, 0.02, summary_text, transform=ax.transAxes,
                 fontsize=9, color='white', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                          edgecolor='white', alpha=0.9))
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Grid styling
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', alpha=0.3)
        
        # Pane colors
        ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        
        # Save with high quality
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
        plt.close(fig)
        
        print(f"\n  Unified multi-particle plot saved to: {plot_filename}")
        
        # <reason>chain: Print summary of what was plotted</reason>
        print(f"  Summary: {len(successful_particles)} particles plotted successfully")
        if failed_particles:
            print(f"  Failed particles: {', '.join(failed_particles)}")
    
    def _to_numpy(self, tensor):
        """Helper to convert tensors to numpy arrays"""
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)
        
    def _add_quantum_uncertainty_cloud(self, ax, r, phi, t, model):
        """
        <reason>chain: Add visualization of quantum uncertainty around trajectory</reason>
        
        Shows the quantum mechanical position uncertainty as a translucent cloud
        around the classical path, with size proportional to the de Broglie wavelength.
        """
        import matplotlib.patches as patches
        from mpl_toolkits.mplot3d import art3d
        
        # <reason>chain: Debug logging to diagnose trajectory issues</reason>
        print(f"  DEBUG: Quantum uncertainty cloud - r range: [{r.min():.3f}, {r.max():.3f}]")
        print(f"  DEBUG: Quantum uncertainty cloud - phi range: [{phi.min():.3f}, {phi.max():.3f}]")
        print(f"  DEBUG: Quantum uncertainty cloud - t range: [{t.min():.3f}, {t.max():.3f}]")
        
        # Get particle properties
        particle_mass = 9.109e-31  # Default to electron mass
        if hasattr(model, '_last_particle') and model._last_particle is not None:
            particle_mass = model._last_particle.mass
            
        # <reason>chain: Compute de Broglie wavelength at each point</reason>
        # λ = h/(mv) where v is the orbital velocity
        h = 6.626e-34  # Planck constant
        c = 3e8  # Speed of light
        
        # Sample points along trajectory for uncertainty visualization
        num_samples = min(20, len(r))  # Don't oversample
        indices = np.linspace(0, len(r)-1, num_samples, dtype=int)
        
        for idx in indices:
            # Estimate velocity from trajectory
            if idx > 0 and idx < len(r) - 1:
                dt = (t[idx+1] - t[idx-1]) * self.engine.time_scale
                dr = (r[idx+1] - r[idx-1]) * self.engine.length_scale
                dphi = phi[idx+1] - phi[idx-1]
                
                # Orbital velocity components
                v_r = dr / dt if dt > 0 else 0
                v_phi = r[idx] * self.engine.length_scale * dphi / dt if dt > 0 else 0
                v = np.sqrt(v_r**2 + v_phi**2)
                
                # Clamp velocity to reasonable bounds
                v = max(v, 0.001 * c)  # At least 0.1% speed of light
                v = min(v, 0.99 * c)   # Less than speed of light
            else:
                v = 0.1 * c  # Default 10% speed of light
                
            # de Broglie wavelength
            lambda_db = h / (particle_mass * v)
            
            # <reason>chain: Scale uncertainty for visualization</reason>
            # In geometric units, typical orbit r ~ 10-100, so scale appropriately
            uncertainty_radius = lambda_db / self.engine.length_scale * 1e6  # Scale for visibility
            uncertainty_radius = min(uncertainty_radius, r[idx] * 0.1)  # Cap at 10% of radius
            
            # Convert to Cartesian coordinates
            x = r[idx] * np.cos(phi[idx])
            y = r[idx] * np.sin(phi[idx])
            z = t[idx]
            
            # <reason>chain: Create uncertainty sphere at this point</reason>
            # Use low-resolution sphere for performance
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x_sphere = x + uncertainty_radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = y + uncertainty_radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = z + uncertainty_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Plot uncertainty cloud
            ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                          color='cyan', alpha=0.1, linewidth=0, 
                          antialiased=False, zorder=1)
                          
        # Add legend entry for quantum uncertainty
        from matplotlib.lines import Line2D
        uncertainty_legend = Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='cyan', alpha=0.3, 
                                  markersize=10, label='Quantum Uncertainty')
        ax.add_artist(uncertainty_legend)
