"""
Simplified Theory Visualizer for gravitational theory simulations.
Focuses on creating a single, publication-quality 3D trajectory plot.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.constants import G as const_G, c as const_c
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# Conditional import for sympy for displaying Lagrangian
try:
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

from physics_agent.base_theory import GravitationalTheory, Tensor, QuantumMixin

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
        
        # <reason>chain: Create larger figure for better visibility
        fig = plt.figure(figsize=(18, 14))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert main theory trajectory - CONVERT TO GEOMETRIC UNITS
        t = self._to_numpy(hist[:, 0]) / self.engine.time_scale
        r = self._to_numpy(hist[:, 1]) / self.engine.length_scale
        phi = self._to_numpy(hist[:, 2])
        
        # Smooth the end of the trajectory to avoid abrupt overlaps
        if len(t) > 2:
            # Interpolate last segment if points overlap
            if np.linalg.norm([r[-1] - r[-2], phi[-1] - phi[-2], t[-1] - t[-2]]) < 1e-2:
                t_interp = np.linspace(t[-2], t[-1], 5)
                r_interp = np.interp(t_interp, t[-2:], r[-2:])
                phi_interp = np.interp(t_interp, t[-2:], phi[-2:])
                t = np.concatenate([t[:-2], t_interp])
                r = np.concatenate([r[:-2], r_interp])
                phi = np.concatenate([phi[:-2], phi_interp])
        
        # <reason>chain: Ensure we have meaningful trajectory data before plotting
        if len(t) < 2:
            print(f"  Warning: Trajectory too short ({len(t)} points), skipping visualization")
            return
        
        # <reason>chain: Check if r is changing - if not, the trajectory has issues
        if np.std(r) < 1e-10:  # If r doesn't change at all
            print(f"  WARNING: Trajectory has no radial motion! r_mean={np.mean(r):.3f}, r_std={np.std(r):.3e}")
            print(f"    Initial r={r[0]:.3f}, final r={r[-1]:.3f}")
            print(f"    This suggests a problem with the geodesic solver or initial conditions")
        
        # <reason>chain: Check if this is a quantum theory and add uncertainty visualization
        theory_category = getattr(model, 'category', 'unknown')
        is_quantum = theory_category == 'quantum' and hasattr(model, 'quantum_integrator')
        
        # Convert to Cartesian coordinates
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = t  # Use time as z-axis
        
        # <reason>chain: Collect all trajectory data for proper bounds calculation
        all_x = [x]
        all_y = [y]
        all_z = [z]
        all_r = [r]
        
        # <reason>chain: Calculate average quantum uncertainty to display in the legend, instead of plotting the cloud.
        avg_uncertainty_radius = None
        if is_quantum and hasattr(model, '_get_average_quantum_uncertainty'):
            avg_uncertainty_radius = model._get_average_quantum_uncertainty(r, phi, t, self.engine)
        
        # <reason>chain: Auto-calculate proper marker intervals based on trajectory length
        # If we have a full trajectory, ensure we show markers that end nicely at the end
        total_steps = len(t)
        desired_markers = 30  # Reduced from 50 for cleaner look
        
        # Calculate step interval to get close to desired markers AND end at the last point
        step_interval = max(1, (total_steps - 1) // desired_markers)
            
        print(f"  Trajectory has {total_steps} points, using step interval {step_interval} for markers")
        print(f"  Trajectory range: r=[{r.min():.3f}, {r.max():.3f}], phi=[{phi.min():.3f}, {phi.max():.3f}]")
        
        # <reason>chain: Warn about problematic trajectories
        r_range = r.max() - r.min()
        phi_range = phi.max() - phi.min()
        if r_range < 1e-6 and phi_range < 1e-6:
            print(f"  WARNING: Theory '{model.name}' produces no motion (stationary trajectory)")
            print(f"           This may indicate a problem with the theory's equations of motion")
        elif r_range < 1e-6:
            print(f"  WARNING: Theory '{model.name}' has no radial motion (circular orbit)")
        elif phi_range < 1e-6:
            print(f"  WARNING: Theory '{model.name}' has no angular motion (radial plunge)")
        
        # <reason>chain: Plot main theory trajectory with thinner line for better cloud visibility
        # Make the main trajectory thinner - reduced from 4 to 2
        main_particle_name = "Primary Particle"
        if particle_info and particle_info.get('particle'):
            particle = particle_info['particle']
            solver_tag = particle_info.get('tag', '')
            
            # Build label with particle name and solver type
            label_parts = [f"{particle.name.capitalize()} ({model.name})"]
            
            # Add solver type to label - always include "Solver" and 4D/6D designation
            # <reason>chain: Check for UGM and quantum theories first before defaulting to symmetric/general
            if 'ugm' in solver_tag:
                # <reason>chain: UGM theories use special UGM solver that handles gauge fields
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
                # <reason>chain: Add explicit UGM and quantum theory check for proper labeling
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
                        # <reason>chain: Quantum theories now use proper path integral solver
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
        
        # <reason>chain: Reduced line width from 4 to 2 for better cloud visibility
        # <reason>chain: Use a dotted line style for the main theory as requested.
        ax.plot(x, y, z, color='red', linestyle=':', linewidth=2, alpha=0.9, label=main_particle_name, zorder=10)
        
        # Add step markers along the main trajectory - make them smaller
        marker_indices = list(range(0, total_steps, step_interval))
        # Ensure the last point is included
        if marker_indices[-1] != total_steps - 1:
            marker_indices.append(total_steps - 1)
            
        for i in marker_indices:
            # <reason>chain: Use consistent marker style regardless of charge
            ax.scatter(x[i], y[i], z[i], c='white', s=15, marker='o',
                      edgecolors='red', linewidth=0.5, alpha=0.7, zorder=11)
        
        # <reason>chain: Initialize list to store z values for axis limits
        all_z = [z]
        
        # <reason>chain: Dynamically build a list of legend elements for baselines that are actually plotted.
        baseline_legend_elements = []
        
        # <reason>chain: Define a monochrome color palette for all baselines for visual consistency.
        baseline_colors = {
            'Schwarzschild': '#A9A9A9',
            'Kerr': '#D3D3D3',
            'Kerr-Newman': '#FFFFFF',
        }
        
        # <reason>chain: Process baselines
        print(f"    Processing {len(baseline_results)} baselines...")
        for baseline_name, baseline_data in baseline_results.items():
            print(f"      Baseline: {baseline_name}")
            
            # <reason>chain: Handle new structure where baselines contain per-particle trajectories</reason>
            # Extract the trajectory for the current particle being visualized
            if isinstance(baseline_data, dict):
                # New structure: baseline_data is a dict of particle trajectories
                # Get particle name and normalize case
                current_particle_name = particle_info.get('particle_properties', {}).get('name', 'electron') if particle_info else 'electron'
                # <reason>chain: Normalize particle name to lowercase for consistency</reason>
                current_particle_name_lower = current_particle_name.lower()
                baseline_hist = baseline_data.get(current_particle_name_lower)
                if baseline_hist is None:
                    # Try exact match first
                    baseline_hist = baseline_data.get(current_particle_name)
                if baseline_hist is None:
                    print(f"        Skipping - no data for particle {current_particle_name}")
                    continue
            else:
                # Old structure: baseline_data is the trajectory itself
                baseline_hist = baseline_data
            
            if baseline_hist is None or len(baseline_hist) < 2:
                print(f"        Skipping - no data or too short")
                continue
                
            # <reason>chain: Skip baseline if it's the same as the main theory
            # Don't plot the main theory as its own baseline
            if baseline_name == model.name or (baseline_name in model.name) or (model.name in baseline_name):
                print(f"        Skipping - same as main theory")
                continue
                
            # Convert baseline trajectory to geometric units
            t_b = self._to_numpy(baseline_hist[:, 0]) / self.engine.time_scale
            r_b = self._to_numpy(baseline_hist[:, 1]) / self.engine.length_scale
            phi_b = self._to_numpy(baseline_hist[:, 2])
            
            # <reason>chain: Fix baselines 'falling off' the plot by ensuring time is monotonic
            # Find the first point where time decreases and truncate the baseline there.
            if len(t_b) > 1:
                monotonic_time_mask = np.concatenate(([True], t_b[1:] >= t_b[:-1]))
                if not np.all(monotonic_time_mask):
                    first_non_monotonic_idx = np.where(~monotonic_time_mask)[0][0]
                    if first_non_monotonic_idx > 0:
                        print(f"        WARNING: Baseline '{baseline_name}' has non-monotonic time at index {first_non_monotonic_idx}. Truncating.")
                        t_b = t_b[:first_non_monotonic_idx]
                        r_b = r_b[:first_non_monotonic_idx]
                        phi_b = phi_b[:first_non_monotonic_idx]
            
            # Clip baseline to match main trajectory time range
            max_t = np.max(t)
            time_mask = t_b <= max_t
            t_b = t_b[time_mask]
            r_b = r_b[time_mask]
            phi_b = phi_b[time_mask]
            
            # Skip if invalid data
            if not np.isfinite(r_b).all() or r_b.max() > 1e10:
                continue
                
            # <reason>chain: Check if baseline has meaningful motion
            if np.std(r_b) < 1e-10 and np.std(phi_b) < 1e-10:
                print(f"        WARNING: Baseline {baseline_name} has no motion!")
                continue
                
            # Convert to Cartesian
            x_b = r_b * np.cos(phi_b)
            y_b = r_b * np.sin(phi_b)
            z_b = t_b
            
            # <reason>chain: Limit z values to prevent baselines from going too high
            # Truncate baseline at main trajectory's max time
            max_z_main = z.max() if len(z) > 0 else 100.0
            z_b_truncated = z_b[z_b <= max_z_main * 1.1]  # Allow 10% extra
            if len(z_b_truncated) > 0:
                x_b = x_b[:len(z_b_truncated)]
                y_b = y_b[:len(z_b_truncated)]
                z_b = z_b_truncated
                # <reason>chain: Add baseline data to bounds calculation
                all_x.append(x_b)
                all_y.append(y_b)
                all_z.append(z_b)
                all_r.append(r_b[:len(z_b_truncated)])
            
            # <reason>chain: Retrieve the baseline theory object to inspect its properties for styling.
            baseline_theory = None
            for theory_name, theory_obj in baseline_theories.items():
                if theory_name in baseline_name:
                    baseline_theory = theory_obj
                    break
                    
            if not baseline_theory:
                print(f"        WARNING: Could not find baseline theory object for '{baseline_name}'. Skipping.")
                continue
            
            # <reason>chain: Determine the baseline color from the monochrome palette.
            color = 'gray' # default
            if 'Kerr-Newman' in baseline_name:
                color = baseline_colors['Kerr-Newman']
            elif 'Kerr' in baseline_name:
                color = baseline_colors['Kerr']
            elif 'Schwarzschild' in baseline_name:
                color = baseline_colors['Schwarzschild']
            
            # <reason>chain: Plot baselines with styles based on whether the black hole is charged.
            # A baseline is considered charged if its class name includes "Newman" or "Reissner".
            is_charged_baseline = 'Newman' in baseline_theory.__class__.__name__ or 'Reissner' in baseline_theory.__class__.__name__

            if is_charged_baseline:
                # Use '+' markers for charged baselines to represent '++' style.
                ax.plot(x_b, y_b, z_b, color=color, linestyle='None', marker='+', markersize=5,
                       alpha=0.6, label=baseline_name, zorder=3)
            else:
                # Use '--' for uncharged baselines.
                ax.plot(x_b, y_b, z_b, color=color, linestyle='--', 
                       linewidth=1, alpha=0.4, label=baseline_name, zorder=3)
            
            # <reason>chain: Dynamically add legend entry for the plotted baseline to ensure accuracy
            # The legend now indicates whether the baseline theory describes a charged black hole.
            if 'Schwarzschild' in baseline_name:
                baseline_legend_elements.append(Line2D([0], [0], color=color, lw=2, linestyle='--', 
                                            label='-- Schwarzschild (uncharged)', alpha=0.7))
            elif 'Kerr' in baseline_name and 'Newman' not in baseline_name:
                baseline_legend_elements.append(Line2D([0], [0], color=color, lw=2, linestyle='--', 
                                            label='-- Kerr (uncharged, rotating)', alpha=0.7))
            elif 'Kerr-Newman' in baseline_name:
                baseline_legend_elements.append(Line2D([0], [0], color=color, lw=0, marker='+', markersize=8,
                                            label='+++ Kerr-Newman (charged, rotating)', alpha=0.7))
            

        
        # <reason>chain: Calculate proper bounds including all trajectories
        # Calculate bounds that include all trajectories
        x_min = min(arr.min() for arr in all_x if len(arr) > 0)
        x_max = max(arr.max() for arr in all_x if len(arr) > 0)
        y_min = min(arr.min() for arr in all_y if len(arr) > 0)
        y_max = max(arr.max() for arr in all_y if len(arr) > 0)
        
        # Calculate ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # <reason>chain: Handle edge case of stationary trajectories
        # If main trajectory has no motion, ensure we can still see baselines
        main_r_range = r.max() - r.min()
        main_phi_range = phi.max() - phi.min()
        
        if main_r_range < 1e-6 and main_phi_range < 1e-6:
            # Stationary trajectory - ensure we can see baselines
            min_range = max(20.0, r.mean())  # At least 20 Rs or the stationary radius
        else:
            min_range = 10.0  # Normal minimum
        
        # <reason>chain: Always include event horizon and origin in view
        # Ensure we can see from origin (singularity) to the orbit
        x_abs_max = max(abs(x_min), abs(x_max))
        y_abs_max = max(abs(y_min), abs(y_max))
        plot_radius = max(x_abs_max, y_abs_max, 5.0) * 1.2  # At least 5 Rs to show event horizon, 20% padding
        
        # Center plot at origin to show black hole
        ax.set_xlim(-plot_radius, plot_radius)
        ax.set_ylim(-plot_radius, plot_radius)
        
        # Z-axis (time) bounds with extra margin
        z_min = min(0, min(arr.min() for arr in all_z if len(arr) > 0 and np.isfinite(arr).all()))
        z_max = max(arr.max() for arr in all_z if len(arr) > 0 and np.isfinite(arr).all())
        if not np.isfinite(z_max):
            z_max = z[-1] * 1.1  # Use main trajectory's max time
        # Add extra margin to Z-axis for label visibility
        z_range = z_max - z_min
        ax.set_zlim(z_min - z_range * 0.05, z_max + z_range * 0.1)
        
        # <reason>chain: Always show event horizon since we center at origin
        # Draw event horizon at r = 2 (2M in geometric units)
        if True:  # Always show event horizon
            u_cyl = np.linspace(0, 2 * np.pi, 100)
            z_cyl_vals = np.linspace(z_min, z_max, 50)
            u_grid, z_grid = np.meshgrid(u_cyl, z_cyl_vals)
            x_cyl = 2.0 * np.cos(u_grid)  # r = 2 in geometric units
            y_cyl = 2.0 * np.sin(u_grid)
            
            # Plot cylinder surface with lower zorder for proper depth
            ax.plot_surface(x_cyl, y_cyl, z_grid, color='cyan', alpha=0.15, linewidth=0, 
                           antialiased=True, zorder=1)
            
            # Add event horizon boundary circles at top and bottom
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), np.full_like(theta, z_min), 
                    'c-', linewidth=2, alpha=0.9, label='Event Horizon (r=2M)', zorder=2)
            ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), np.full_like(theta, z_max), 
                    'c-', linewidth=2, alpha=0.9, zorder=2)
        
        # <reason>chain: Always show singularity at origin
        if True:
            ax.plot([0, 0], [0, 0], [z_min, z_max], 'y--', linewidth=2, alpha=0.8, 
                    label='Singularity (r=0)', zorder=2)
        
        # <reason>chain: Always show initial orbit circle for reference
        # Show initial circular orbit
        r0_orbit = r[0]
        if True:
            orbit_theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(r0_orbit * np.cos(orbit_theta), r0_orbit * np.sin(orbit_theta), 
                    np.full_like(orbit_theta, z[0]), 
                    'g--', linewidth=1, alpha=0.5, label=f'Initial orbit (r={r0_orbit:.1f}M)', zorder=2)
        
        # <reason>chain: Enhanced title with particle information
        # Build title with particle info if available
        title_parts = [f"Geodesic: {model.name}"]
        
        # <reason>chain: Add Lagrangian type to title
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
            # <reason>chain: Check for UGM and quantum theories first
            theory_category = getattr(model, 'category', 'unknown')
            if 'ugm' in solver_tag:
                # <reason>chain: UGM theories use special UGM solver
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
        ax.set_title(title, fontsize=20, pad=25, color='white')
        
        # Fix axis labels with clear unit descriptions
        ax.set_xlabel('\nX (Schwarzschild radii)', fontsize=16, color='white')
        ax.set_ylabel('\nY (Schwarzschild radii)', fontsize=16, color='white')
        # <reason>chain: Make z-label more visible with better positioning and larger font
        dtau_value = 0.1  # Default time step in geometric units
        ax.set_zlabel(f'Time Units\n(Δτ = {dtau_value})', fontsize=18, color='white', labelpad=30)
        
        # <reason>chain: Removed CONSTRAINTS PASSED text - now in legend
        
        # <reason>chain: Remove legend from plot - will be added below
        # Don't add legend to the plot itself
        
        # Set viewing angle for best perspective
        ax.view_init(elev=25, azim=45)
        
        # Grid styling
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', alpha=0.3)
        
        # Pane colors for dark theme
        ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 0.8))
        
        # Save the figure with proper spacing
        # <reason>chain: Adjust margins for cleaner look with simplified legend
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92)
        
        # <reason>chain: Create beautiful, crystal-clear visual legend
        from matplotlib.patches import Patch, Circle
        
        # Create legend elements with perfect visual clarity
        legend_elements = []
        
        # === TRAJECTORIES SECTION ===
        # Main trajectory with particle name
        legend_elements.append(Line2D([0], [0], color='red', lw=2, linestyle=':',
                                    label=f'··· {main_particle_name.capitalize()} trajectory',
                                    solid_capstyle='round'))
        
        # <reason>chain: Add average quantum uncertainty to the legend instead of plotting the cloud.
        if is_quantum:
            label_text = 'Quantum Uncertainty'
            if avg_uncertainty_radius:
                label_text = f'Avg. Quantum Uncertainty: {avg_uncertainty_radius:.2e} Rs'
            legend_elements.append(Patch(facecolor='cyan', alpha=0.5, 
                                       label=f'⬤ {label_text}'))
        
        # Add spacing
        legend_elements.append(Line2D([0], [0], color='none', label=''))
        
        # === BASELINE COMPARISONS ===
        legend_elements.append(Line2D([0], [0], color='none', 
                                    label='BASELINE THEORIES:'))
        
        legend_elements.extend(baseline_legend_elements)
        
        # Add spacing
        legend_elements.append(Line2D([0], [0], color='none', label=''))
        
        # === KEY FEATURES ===
        legend_elements.append(Line2D([0], [0], color='none', 
                                    label='SPACETIME FEATURES:'))
        
        if 2.0 < plot_radius:
            # Event horizon with cylinder symbol
            legend_elements.append(Line2D([0], [0], color='cyan', lw=4, alpha=0.5,
                                        label='▐ Event horizon (r=2M)', 
                                        solid_capstyle='butt'))
        
        # Step markers
        legend_elements.append(Line2D([0], [0], color='white', lw=0, marker='o', 
                                    markersize=6, label='• Progress markers (o: uncharged, +: charged)'))
        
        # Create the legend with enhanced styling
        legend = ax.legend(handles=legend_elements, 
                          loc='upper left', 
                          bbox_to_anchor=(0.01, 0.99),
                          fancybox=True, 
                          shadow=True, 
                          ncol=1, 
                          fontsize=11,
                          facecolor='#0a0a0a',  # Very dark background
                          edgecolor='#333333',  # Dark gray border
                          framealpha=0.95,
                          labelcolor='white',
                          borderpad=1.0,
                          columnspacing=1.5,
                          handlelength=3.0,  # Longer lines for clarity
                          handletextpad=1.0)  # More space between symbol and text
        
        # Style the legend title and text
        legend.get_frame().set_linewidth(1.5)
        
        # Make section headers bold
        for i, text in enumerate(legend.get_texts()):
            label = text.get_text()
            if 'BASELINE THEORIES:' in label or 'SPACETIME FEATURES:' in label:
                text.set_weight('bold')
                text.set_color('#88ccff')  # Light blue for headers
            elif label == '':  # Spacer lines
                text.set_fontsize(6)
            else:
                text.set_weight('normal')
        
        # <reason>chain: Add minimal key information below plot
        # Add quantum scale info if needed
        if is_quantum:
            scale_text = "Quantum uncertainty scaled for visibility"
            fig.text(0.5, 0.02, scale_text, fontsize=10, ha='center', style='italic',
                    color='cyan', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='black', alpha=0.7))
        
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
        # <reason>chain: Use gold for photon for better visibility
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
            
            # <reason>chain: Check if trajectory has meaningful motion
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
            
            # <reason>chain: Add solver type to clarify which theorem was used
            # Determine and add solver type with clear 4D/6D designation
            solver_tag = particle_data.get('tag', '')
            theory_category = getattr(model, 'category', 'unknown')
            
            if 'ugm' in solver_tag:
                # <reason>chain: UGM theories use special UGM solver
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
        # <reason>chain: Use proper time τ in geometric units for relativistic accuracy
        ax.set_zlabel('Proper Time (τ)', fontsize=16, labelpad=15)
        
        # Legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                 fontsize=10, frameon=True, fancybox=True,
                 facecolor='black', edgecolor='white', framealpha=0.9)
        
        # Grid and viewing angle
        ax.grid(True, alpha=0.3, color='gray')
        ax.view_init(elev=25, azim=45)
        
        # Set tick colors to white and ensure labels are visible
        ax.tick_params(axis='x', colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=10)
        ax.tick_params(axis='z', colors='white', labelsize=10)
        
        # Force all axis properties to be white
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        
        # Set tick label colors explicitly
        ax.xaxis.set_tick_params(labelcolor='white')
        ax.yaxis.set_tick_params(labelcolor='white')
        ax.zaxis.set_tick_params(labelcolor='white')
        
        # Force update the 3D axis info
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo['tick']['color'] = 'white'
            axis._axinfo['label']['color'] = 'white'
            axis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.5)
            # Explicitly set tick label color
            axis.set_tick_params(which='both', labelcolor='white')
            # Force redraw ticks
            for tick in axis.get_major_ticks():
                tick.label1.set_color('white')
                tick.label2.set_color('white')
        
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
        <reason>chain: Generate a grid visualization showing trajectories for multiple particles
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
        
        # <reason>chain: First, generate baseline trajectory for each particle type
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
            # <reason>chain: Use gold for photon for better visibility
            particle_colors = {
                'electron': 'blue',
                'photon': '#FFD700',  # Gold - more visible than yellow
                'proton': 'red',
                'neutrino': 'green'
            }
            color = particle_colors.get(particle_name, 'white')
            
            # <reason>chain: Extract particle object from result dictionary
            particle = result.get('particle')
            
            # <reason>chain: Build label with particle name and solver type for clarity
            # Determine solver type for the legend with clear 4D/6D designation
            solver_tag = result.get('tag', '')
            solver_label = ""
            theory_category = getattr(model, 'category', 'unknown')
            
            if 'ugm' in solver_tag:
                # <reason>chain: UGM theories use special UGM solver
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
                # <reason>chain: Use actual baseline name instead of generic "GR Baseline"
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
            
            # <reason>chain: Clearly indicate if theory uses quantum or classical Lagrangian
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
                
                # <reason>chain: Use actual baseline name from theory
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
        
        <reason>chain: Create a single plot showing all particles for easier comparison
        <reason>chain: Use time as z-axis and Cartesian coordinates for clarity
        <reason>chain: Add particle-specific labels with properties
        <reason>chain: Include solver type in labels
        <reason>chain: Add validation status to subtitle
        <reason>chain: Use particle-specific colors for distinction
        <reason>chain: Add event horizon cylinder
        <reason>chain: Add info box with theory details
        """
        print(f"    Generating unified multi-particle plot for {model.name}...")
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(16, 12), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        
        # Define particle colors matching the standard scheme
        # <reason>chain: Change photon color to gold/orange for better visibility on dark background
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
        
        # <reason>chain: First check what data we have
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
            
            # <reason>chain: Diagnostic info for troubleshooting
            print(f"      {particle_name} trajectory: {len(t)} points")
            print(f"      r range: [{r.min():.3f}, {r.max():.3f}] (geometric units)")
            print(f"      phi range: [{phi.min():.3f}, {phi.max():.3f}] rad")
            print(f"      t range: [{t.min():.3f}, {t.max():.3f}] (geometric units)")
            
            # <reason>chain: Check if the trajectory has meaningful motion
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
            
            # <reason>chain: Add solver type to make it clear which theorem was used
            # Determine and add solver type with clear 4D/6D designation
            solver_tag = particle_data.get('tag', '')
            solver_label = ""
            theory_category = getattr(model, 'category', 'unknown')
            
            if 'ugm' in solver_tag:
                # <reason>chain: UGM theories use special UGM solver
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
        
        # <reason>chain: Add a reference baseline for comparison
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
            
            # Plot baseline trajectory with better visibility
            ax.plot(x_b, y_b, z_b, color='white', linewidth=2.0, alpha=0.6, 
                   linestyle='--', label='Schwarzschild Baseline', zorder=3)
            
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
            if not np.isfinite(z_max):
                z_max = 100  # Default max time if no valid data
            print(f"    Time range across all particles: [{z_min:.3f}, {z_max:.3f}]")
        else:
            z_min, z_max = 0, 100
            print("    WARNING: No valid time data found!")
        
        # <reason>chain: Ensure proper z-axis range even if time data is constant
        if abs(z_max - z_min) < 1e-6:  # Essentially the same time
            print("    WARNING: Time range too small, expanding for visualization")
            z_max = max(z_min + 10.0, 10.0)  # Expand range
        
        ax.set_zlim(z_min, z_max)
        
        # Event Horizon - make cylinder more visible
        u_cyl = np.linspace(0, 2 * np.pi, 50)
        z_cyl_vals = np.linspace(z_min, z_max, 20)
        u_grid, z_grid = np.meshgrid(u_cyl, z_cyl_vals)
        x_cyl = 2.0 * np.cos(u_grid)  # r = 2M
        y_cyl = 2.0 * np.sin(u_grid)
        
        # Use wireframe for better visibility
        ax.plot_wireframe(x_cyl, y_cyl, z_grid, color='cyan', alpha=0.5, linewidth=0.5,
                         rstride=5, cstride=5, zorder=1)
        
        # Add event horizon boundary circles at top and bottom
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), [z_min]*len(theta), 
                'c-', linewidth=2, alpha=0.8, zorder=2)
        ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), [z_max]*len(theta), 
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
        
        # <reason>chain: Add Lagrangian type to title for clarity
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
        # <reason>chain: Use proper time τ in geometric units for relativistic accuracy
        ax.set_zlabel('Proper Time (τ)', fontsize=16, color='white', labelpad=15)
        
        # Custom legend with additional labels
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        
        # Add event horizon and singularity to legend if not already there
        from matplotlib.lines import Line2D
        if 'Event Horizon' not in legend_labels:
            legend_handles.append(Line2D([0], [0], color='cyan', linewidth=2, label='Event Horizon (r=2M)'))
        if 'Singularity' not in legend_labels:
            legend_handles.append(Line2D([0], [0], color='yellow', linestyle='--', linewidth=2, label='Singularity (r=0)'))
        
        # <reason>chain: Fix legend to properly show all trajectories with white text
        # Get the actual labels from the handles
        labels_for_handles = [h.get_label() for h in legend_handles]
        legend = ax.legend(handles=legend_handles, labels=labels_for_handles,
                          loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                          fontsize=10, frameon=True, fancybox=True,
                          facecolor='black', edgecolor='white', framealpha=0.9,
                          labelcolor='white',  # White text for visibility
                          ncol=1)  # Single column for cleaner look
        
        # Ensure all text is white
        for text in legend.get_texts():
            text.set_color('white')
        
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
        
        # <reason>chain: Print summary of what was plotted
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
        Shows the quantum mechanical position uncertainty as a translucent cloud
        around the trajectory, with both scaled visualization and true microscopic scale inset.
        
        <reason>chain: Show both a scaled version for visibility and true scale in inset
        """
        if not isinstance(model, QuantumMixin):
            return
            
        print(f"  DEBUG: Quantum uncertainty cloud - r range: [{r.min():.3f}, {r.max():.3f}]")
        print(f"  DEBUG: Quantum uncertainty cloud - phi range: [{phi.min():.3f}, {phi.max():.3f}]")
        print(f"  DEBUG: Quantum uncertainty cloud - t range: [{t.min():.3f}, {t.max():.3f}]")
        
        # Physical constants (SI units)
        h = 6.626e-34  # Planck constant
        c = 3e8  # Speed of light
        
        # Get particle mass
        particle_mass = model._last_particle.mass if hasattr(model, '_last_particle') else 9.109e-31  # Electron default
        
        # <reason>chain: Use more samples for the cloud on longer trajectories to ensure it follows the path to the end
        # Use more samples, scaling with trajectory length, to prevent the cloud from stopping short.
        num_samples = min(max(20, len(r) // 8), 80) # Scale with length, min 20, max 80 samples.
        indices = np.linspace(0, len(r)-1, num_samples, dtype=int)
        
        # Create cloud visualization
        cloud_points = []
        cloud_radii_actual = []
        cloud_radii_scaled = []
        actual_wavelengths = []
        
        # Handle both stationary and moving trajectories
        r_range = r.max() - r.min()
        phi_range = phi.max() - phi.min()
        is_stationary = r_range < 1e-3 and phi_range < 1e-3
        
        if is_stationary:
            print(f"  Trajectory appears stationary - showing quantum uncertainty at fixed position")
            # For stationary case, create a cloud at the position
            r_pos = r.mean()
            phi_pos = phi.mean()
            
            # For stationary particle, use thermal de Broglie wavelength
            T = 300  # Room temperature assumption
            v_thermal = (3 * 1.381e-23 * T / particle_mass)**0.5
            lambda_db = h / (particle_mass * v_thermal)
            actual_wavelengths.append(lambda_db)
            
            # <reason>chain: Calculate actual and scaled uncertainties
            uncertainty_radius_actual = lambda_db / self.engine.length_scale  # In geometric units
            uncertainty_radius_scaled = 0.1 * r_pos  # 10% of orbital radius for visibility
            
            # For main plot, show scaled version as translucent sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x_sphere = r_pos * np.cos(phi_pos) + uncertainty_radius_scaled * np.outer(np.cos(u), np.sin(v))
            y_sphere = r_pos * np.sin(phi_pos) + uncertainty_radius_scaled * np.outer(np.sin(u), np.sin(v))
            z_sphere = t.mean() + uncertainty_radius_scaled * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color='cyan', alpha=0.2, zorder=6)
            
            # Add inset for true scale
            self._add_uncertainty_inset(ax, r_pos * np.cos(phi_pos), r_pos * np.sin(phi_pos), 
                                       t.mean(), uncertainty_radius_actual, uncertainty_radius_scaled)
            
        else:
            # Moving trajectories - create translucent tube
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
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
                actual_wavelengths.append(lambda_db)
                
                # <reason>chain: Calculate both actual and scaled uncertainties
                uncertainty_radius_actual = (lambda_db / 2) / self.engine.length_scale  # Geometric units
                # Scale up for visibility - proportional to orbital radius
                uncertainty_radius_scaled = 0.02 * r[idx]  # 2% of orbital radius
                
                # Convert to Cartesian
                x_pt = r[idx] * np.cos(phi[idx])
                y_pt = r[idx] * np.sin(phi[idx])
                z_pt = t[idx]
                
                cloud_points.append([x_pt, y_pt, z_pt])
                cloud_radii_actual.append(uncertainty_radius_actual)
                cloud_radii_scaled.append(uncertainty_radius_scaled)
            
            # <reason>chain: Create translucent tube for scaled quantum uncertainty
            if len(cloud_points) > 2:
                verts = []
                for i in range(len(cloud_points) - 1):
                    p1 = cloud_points[i]
                    p2 = cloud_points[i + 1]
                    r1 = cloud_radii_scaled[i]
                    r2 = cloud_radii_scaled[i + 1]
                    
                    # Create cylinder segment
                    theta = np.linspace(0, 2 * np.pi, 8)  # 8-sided for performance
                    
                    # Bottom circle
                    x1 = p1[0] + r1 * np.cos(theta)
                    y1 = p1[1] + r1 * np.sin(theta)
                    z1 = np.full_like(theta, p1[2])
                    
                    # Top circle  
                    x2 = p2[0] + r2 * np.cos(theta)
                    y2 = p2[1] + r2 * np.sin(theta)
                    z2 = np.full_like(theta, p2[2])
                    
                    # Create faces
                    for j in range(len(theta) - 1):
                        verts.append([
                            [x1[j], y1[j], z1[j]],
                            [x1[j+1], y1[j+1], z1[j+1]],
                            [x2[j+1], y2[j+1], z2[j+1]],
                            [x2[j], y2[j], z2[j]]
                        ])
                
                # Add the tube
                tube = Poly3DCollection(verts, alpha=0.2, facecolor='cyan', 
                                      edgecolor='none', zorder=6)
                ax.add_collection3d(tube)
        
        # Calculate scale factor for display
        if actual_wavelengths and cloud_radii_scaled:
            avg_wavelength = np.mean(actual_wavelengths)
            avg_radius_actual = np.mean(cloud_radii_actual) if cloud_radii_actual else 0
            avg_radius_scaled = np.mean(cloud_radii_scaled)
            rs_meters = 2 * const_G * 1.989e30 / const_c**2  # Solar Schwarzschild radius ~3km
            actual_size_meters = avg_wavelength
            
            # Calculate display scale factor
            if avg_radius_actual > 0:
                display_scale = avg_radius_scaled / avg_radius_actual
            else:
                display_scale = 1e15  # Typical scale factor
        
        # Add inset for one representative point to show actual uncertainty scale
        if cloud_points and cloud_radii_actual:
            # Choose middle point for inset
            mid_idx = len(cloud_points) // 2
            self._add_uncertainty_inset(ax, cloud_points[mid_idx][0], cloud_points[mid_idx][1], 
                                       cloud_points[mid_idx][2], cloud_radii_actual[mid_idx],
                                       cloud_radii_scaled[mid_idx])
        
        # <reason>chain: Add text to explain the scaling
        if actual_wavelengths:
            scale_text = f"Quantum uncertainty scaled {display_scale:.0e}× for visibility"
            ax.text2D(0.02, 0.98, scale_text, transform=ax.transAxes, 
                     fontsize=9, color='cyan', alpha=0.8, va='top')
        
        print(f"  Added quantum uncertainty visualization with {len(indices)} sample points")
        if len(cloud_radii_actual) > 0:
            print(f"  Actual uncertainty radius: ~{np.mean(cloud_radii_actual):.3e} Rs")
            print(f"  Scaled uncertainty radius: ~{np.mean(cloud_radii_scaled):.3f} Rs")
            if actual_wavelengths:
                print(f"  Actual de Broglie wavelength: ~{actual_size_meters:.2e} m")
                print(f"  Display scale factor: {display_scale:.2e}×")
    
    def _add_uncertainty_inset(self, ax, x0, y0, z0, radius_actual, radius_scaled):
        """
        Add zoomed-in inset showing actual quantum uncertainty scale.
        
        <reason>chain: Creates a small inset axes to visualize the microscopic uncertainty
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        # Create 2D inset (matplotlib 3D insets have issues)
        inset_ax = inset_axes(ax, width="20%", height="20%", 
                             loc='lower right', borderpad=4,
                             bbox_to_anchor=(0.95, 0.05, 1, 1),
                             bbox_transform=ax.transAxes)
        
        # Set inset background
        inset_ax.patch.set_facecolor('#0a0a0a')
        inset_ax.patch.set_alpha(0.9)
        
        # Draw the trajectory point as a red dot
        inset_ax.scatter(0, 0, c='red', s=50, zorder=10)
        
        # Draw the actual uncertainty as a circle
        circle_actual = plt.Circle((0, 0), radius_actual, 
                                  color='cyan', fill=False, linewidth=2,
                                  linestyle='--', label='Actual scale')
        inset_ax.add_patch(circle_actual)
        
        # Set limits to show the actual scale
        max_range = radius_actual * 5
        inset_ax.set_xlim(-max_range, max_range)
        inset_ax.set_ylim(-max_range, max_range)
        inset_ax.set_aspect('equal')
        
        # Remove ticks
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        
        # Add title to inset
        inset_ax.set_title(f'Actual λ/2 = {radius_actual:.2e} Rs', 
                          fontsize=8, color='white', pad=5)
        
        # Add scale bar if radius is very small
        if radius_actual < 1e-10:
            # Add a scale bar showing the size
            scale_length = radius_actual * 2
            inset_ax.plot([-scale_length/2, scale_length/2], [-max_range*0.8, -max_range*0.8],
                         'w-', linewidth=2)
            inset_ax.text(0, -max_range*0.9, f'{scale_length:.1e} Rs',
                         ha='center', va='top', fontsize=7, color='white')
        
        # Add border to inset
        for spine in inset_ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1)
