"""
Modern 3D Theory Visualizer for gravitational theory simulations.
Creates beautiful 3D trajectory visualizations showing particles approaching black holes.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from scipy.constants import G as const_G, c as const_c
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation

# Conditional import for sympy for displaying Lagrangian
try:
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

from physics_agent.base_theory import GravitationalTheory, Tensor, QuantumMixin

class TheoryVisualizer:
    """
    Modern 3D visualizer for gravitational theories.
    Creates publication-quality 3D plots showing particle trajectories near black holes.
    """
    
    def __init__(self, engine):
        """
        Initialize the visualizer with a reference to the engine.
        
        Args:
            engine: TheoryEngine instance for accessing physics computations
        """
        self.engine = engine
        # Custom color scheme for beautiful visualizations
        self.colors = {
            'electron': '#4287f5',    # Bright blue
            'photon': '#f5d442',      # Bright yellow  
            'proton': '#f54242',      # Bright red
            'neutrino': '#42f554',    # Bright green
            'theory': '#ffffff',      # White for main theory
            'kerr': '#00ffff',        # Cyan for Kerr baseline (better visibility)
            'horizon': '#000000',     # Black for event horizon
            'singularity': '#ff0000'  # Red for singularity
        }
        
    def _to_numpy(self, tensor):
        """Convert torch tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy()
        return np.array(tensor)
    
    def _cartesian_from_polar(self, r, theta, phi):
        """Convert spherical coordinates to Cartesian."""
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z
    
    def _draw_event_horizon(self, ax, rs, alpha=0.3):
        """Draw a semi-transparent sphere representing the event horizon."""
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = rs * np.outer(np.cos(u), np.sin(v))
        y = rs * np.outer(np.sin(u), np.sin(v))
        z = rs * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Draw wireframe sphere for better visibility
        ax.plot_wireframe(x, y, z, color='red', alpha=0.2, linewidth=0.5)
        
        # Add a prominent equatorial circle
        theta_eq = np.linspace(0, 2 * np.pi, 100)
        x_eq = rs * np.cos(theta_eq)
        y_eq = rs * np.sin(theta_eq)
        z_eq = np.zeros_like(theta_eq)
        ax.plot(x_eq, y_eq, z_eq, color='red', linewidth=2, alpha=0.8, label='Event Horizon')
        
    def _draw_singularity(self, ax):
        """Draw the singularity at the center."""
        ax.scatter([0], [0], [0], color='red', s=200, marker='*', 
                  edgecolors='yellow', linewidth=2, label='Singularity', zorder=1000)
    
    def _create_3d_trajectory_plot(self, ax, hist, color, label, linewidth=2, alpha=1.0, solver_info=None):
        """Create 3D trajectory plot with solver information and improved labels"""
        hist_np = self._to_numpy(hist)
        
        # Extract coordinates based on solver type
        # 4D solver: [t, r, phi, dr/dtau]
        # 6D solver: [t, r, theta, phi, u_t, u_r, u_phi]
        t = hist_np[:, 0]
        r = hist_np[:, 1]
        
        if hist_np.shape[1] == 4:
            # 4D symmetric solver output
            theta = np.full_like(r, np.pi/2)  # Equatorial plane
            phi = hist_np[:, 2]
        else:
            # 6D general solver output
            theta = hist_np[:, 2] if hist_np.shape[1] > 2 else np.full_like(r, np.pi/2)
            phi = hist_np[:, 3] if hist_np.shape[1] > 3 else np.zeros_like(r)
        
        # Convert to Cartesian coordinates
        x, y, z = self._cartesian_from_polar(r, theta, phi)
        
        # Create gradient effect by varying alpha along trajectory
        segments = len(x) - 1
        for i in range(segments):
            segment_alpha = alpha * (1 - 0.5 * i / segments)  # Fade out towards the end
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                   color=color, linewidth=linewidth, alpha=segment_alpha)
        
        # Add time markers along the trajectory
        # Only show a few key time markers for maximum legibility
        if len(t) > 10:  # Only add markers if we have enough points
            # Show markers at: start (0%), 25%, 50%, 75%, and end (100%)
            percentages = [0, 0.25, 0.5, 0.75, 1.0]
            marker_indices = []
            
            for pct in percentages:
                idx = int(pct * (len(t) - 1))
                marker_indices.append(idx)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in marker_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            
            for i, idx in enumerate(unique_indices):
                # Add small sphere at this point
                ax.scatter(x[idx], y[idx], z[idx], color=color, s=5, alpha=0.6, 
                          edgecolors='white', linewidth=0.3, zorder=100+i)
                
                # Add time label with better formatting and positioning
                time_val = t[idx]
                time_label = f"t={time_val:.1f}"
                
                # Calculate offset direction based on trajectory direction at this point
                if idx < len(x) - 1:
                    # Direction vector to next point
                    dx = x[idx+1] - x[idx]
                    dy = y[idx+1] - y[idx] 
                    dz = z[idx+1] - z[idx]
                else:
                    # For last point, use direction from previous
                    dx = x[idx] - x[idx-1]
                    dy = y[idx] - y[idx-1]
                    dz = z[idx] - z[idx-1]
                
                # Normalize and create perpendicular offset
                norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-10
                # Offset perpendicular to trajectory - increased offset distance
                offset_scale = 0.5 * np.sqrt((x.max()-x.min())**2 + (y.max()-y.min())**2 + (z.max()-z.min())**2)
                offset_x = -dy/norm * offset_scale
                offset_y = dx/norm * offset_scale
                offset_z = 0.2 * offset_scale  # Small z offset
                
                # Position for label
                label_x = x[idx] + offset_x
                label_y = y[idx] + offset_y
                label_z = z[idx] + offset_z
                
                # Draw connecting line from trajectory to label
                ax.plot([x[idx], label_x], [y[idx], label_y], [z[idx], label_z],
                       color='white', linewidth=0.3, alpha=0.3, linestyle=':')
                
                # Add the label - much smaller fontsize (10x smaller: 10 -> 1)
                ax.text(label_x, label_y, label_z, time_label, 
                       fontsize=1, color='white', alpha=0.7,
                       ha='center', va='center', zorder=200+i)
        
        # Add glow effect for the current position
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=30, alpha=1.0, 
                  edgecolors='white', linewidth=1)
        
        # Return line for legend
        return Line2D([0], [0], color=color, linewidth=linewidth, label=label)
    
    def generate_comparison_plot(self, model: GravitationalTheory, hist: Tensor, baseline_results: dict, 
                                baseline_theories: dict, plot_filename: str, rs_val: float, 
                                validations_dict: dict = None, particle_info: dict = None,
                                solver_info: dict = None):
        """
        Generate a single particle 3D trajectory comparison plot.
        """
        # Use multi-particle visualization with single particle
        particle_name = 'default'
        if particle_info and 'particle' in particle_info:
            particle_name = particle_info['particle'].name.lower()
        
        particle_trajectories = {particle_name: hist}
        
        self.generate_all_particles_comparison(
            model, particle_trajectories, baseline_results, baseline_theories,
            plot_filename, rs_val, validations_dict, solver_info=solver_info
        )
    
    def generate_all_particles_comparison(self, model: GravitationalTheory, particle_trajectories: dict,
                                             baseline_results: dict, baseline_theories: dict,
                                             plot_filename: str, rs_val: float, validations_dict: dict = None,
                                             solver_info: dict = None):
        """
        Generate beautiful 3D comparison plots for all particles.
        Creates one plot per particle showing theory vs Kerr baseline.
        """
        # Set up the figure with dark background for aesthetics
        
        # Filter out None trajectories
        valid_trajectories = {}
        for name, traj in particle_trajectories.items():
            if traj is not None and hasattr(traj, '__len__') and len(traj) > 0:
                valid_trajectories[name] = traj
            else:
                print(f"    Skipping {name} - no valid trajectory data")
        
        if not valid_trajectories:
            print("    No valid trajectories to plot!")
            return
        
        # Get particle names and sort them
        particle_names = sorted(list(valid_trajectories.keys()))
        n_particles = len(particle_names)
        
        # Create subplots - 2x2 grid for up to 4 particles
        if n_particles == 1:
            fig = plt.figure(figsize=(12, 10))
            subplot_grid = (1, 1)
        elif n_particles == 2:
            fig = plt.figure(figsize=(20, 10))
            subplot_grid = (1, 2)
        else:
            fig = plt.figure(figsize=(20, 20))
            subplot_grid = (2, 2)
            
        fig.patch.set_facecolor('#0a0a0a')
        
        for idx, particle_name in enumerate(particle_names):
            ax = fig.add_subplot(subplot_grid[0], subplot_grid[1], idx + 1, projection='3d')
            ax.set_facecolor('#0a0a0a')
            
            # Get particle color
            particle = self.engine.particle_loader.get_particle(particle_name)
            particle_color = particle.color if hasattr(particle, 'color') else self.colors.get(particle_name, 'white')
            
            # Draw event horizon
            self._draw_event_horizon(ax, rs_val, alpha=0.2)
            
            # Draw singularity
            self._draw_singularity(ax)
            
            # Plot main theory trajectory
            hist = valid_trajectories[particle_name]
            theory_line = self._create_3d_trajectory_plot(
                ax, hist, particle_color, f'{model.name} - {particle_name.capitalize()}',
                linewidth=3, alpha=0.9, solver_info=solver_info
            )
            
            # Plot Kerr baseline if available
            kerr_line = None
            for baseline_name, baseline_hist in baseline_results.items():
                # Look for Kerr baseline for this specific particle
                if 'kerr' in baseline_name.lower() and 'newman' not in baseline_name.lower() and particle_name in baseline_name.lower():
                    kerr_line = self._create_3d_trajectory_plot(
                        ax, baseline_hist, self.colors['kerr'], f'Kerr - {particle_name.capitalize()}',
                        linewidth=2, alpha=0.6
                    )
                    break
            
            # Set labels and title
            ax.set_xlabel('X [Schwarzschild radii]', fontsize=12, labelpad=10)
            ax.set_ylabel('Y [Schwarzschild radii]', fontsize=12, labelpad=10)
            ax.set_zlabel('Z [Schwarzschild radii]', fontsize=12, labelpad=10)
            ax.set_title(f'{particle_name.capitalize()} Trajectory Near Black Hole', 
                        fontsize=16, pad=20, color='white')
            
            # Set viewing angle for best perspective
            ax.view_init(elev=20, azim=45)
            
            # Set axis limits dynamically to focus on the trajectory
            hist_np = self._to_numpy(hist)
            
            # Extract coordinates based on solver type
            if hist_np.shape[1] == 4:
                # 4D symmetric solver output
                theta_vals = np.pi/2
                phi_vals = hist_np[:, 2]
            else:
                # 6D general solver output
                theta_vals = hist_np[:, 2] if hist_np.shape[1] > 2 else np.pi/2
                phi_vals = hist_np[:, 3] if hist_np.shape[1] > 3 else 0
                
            r_vals = hist_np[:, 1]
            x, y, z = self._cartesian_from_polar(r_vals, theta_vals, phi_vals)
            
            # Set limits
            x_range = max(abs(x.max()), abs(x.min()))
            y_range = max(abs(y.max()), abs(y.min()))
            z_range = max(abs(z.max()), abs(z.min()))
            max_range = max(x_range, y_range, z_range, 3 * rs_val) * 1.2
            
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            
            # Grid styling
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            # Legend with computational methods in first subplot
            if idx == 0:
                legend_elements = []
                if theory_line:
                    legend_elements.append(theory_line)
                if kerr_line:
                    legend_elements.append(kerr_line)
                # Add event horizon and singularity to legend
                legend_elements.extend([
                    Line2D([0], [0], color='none', marker='o', markersize=8, 
                          markerfacecolor='none', markeredgecolor='cyan', label='Event Horizon'),
                    Line2D([0], [0], color='none', marker='*', markersize=10, 
                          markerfacecolor='red', markeredgecolor='yellow', label='Singularity')
                ])
                
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                         frameon=True, fancybox=True, shadow=True,
                         facecolor='#1a1a1a', edgecolor='white')
                
                # Add computational methods box
                methods_text = "Computational Methods:\n"
                if hasattr(model, 'theory_type'):
                    methods_text += f"Theory Solver:\n"
                    if 'quantum' in model.theory_type.lower():
                        methods_text += "• Quantum Geodesic Solver\n"
                        methods_text += "• Path integral formulation\n"
                        methods_text += "• WKB/Semiclassical approx.\n"
                        methods_text += "• QED corrections enabled\n"
                    else:
                        if model.is_symmetric:
                            methods_text += "• 4D symmetric spacetime\n"
                            methods_text += "• Conserved: E, L_z\n"
                        else:
                            methods_text += "• 6D general spacetime\n"
                
                methods_text += "\nKerr Baseline:\n"
                methods_text += "• Kerr Geodesic Solver\n"
                methods_text += "• Boyer-Lindquist coords\n"
                methods_text += "• Spin parameter a=0.50\n"
                
                # Add solver info if provided
                if solver_info:
                    if 'solver_type' in solver_info:
                        methods_text += f"\nActive: {solver_info['solver_type']}"
                    if 'integration_method' in solver_info:
                        methods_text += f"\nMethod: {solver_info['integration_method']}"
                
                # Position the methods box in the lower left
                ax.text2D(0.02, 0.02, methods_text, transform=ax.transAxes,
                         fontsize=8, color='white', alpha=0.8,
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', 
                                  edgecolor='white', alpha=0.7),
                         verticalalignment='bottom', horizontalalignment='left',
                         family='monospace')
            
            # Particle type label
            if hasattr(particle, 'particle_type'):
                type_label = f"Type: {particle.particle_type}"
                if particle.particle_type == 'massless':
                    type_label += "\nOrbit: circular_fast"
                ax.text2D(0.02, 0.98, type_label, transform=ax.transAxes,
                         fontsize=8, color='white', alpha=0.7,
                         verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', 
                                  edgecolor=particle_color, alpha=0.5))
        
        # Add main title
        main_title = f"{model.name} - Particle Trajectories Near Black Hole"
        fig.suptitle(main_title, fontsize=20, color='white', y=0.98)
        
        # Add theory description at bottom
        theory_text = f"Theory: {model.name}"
        if hasattr(model, 'short_description'):
            theory_text += f"\n{model.short_description}"
        
        fig.text(0.5, 0.02, theory_text, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                         edgecolor='white', alpha=0.8))
        
        # Save with high quality
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, facecolor='#0a0a0a', edgecolor='none',
                   bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        print(f"    Generated 3D particle comparison plot: {plot_filename}")
    
    def generate_multi_particle_grid(self, model: GravitationalTheory, particle_results: dict, 
                                   baseline_results: dict, baseline_theories: dict,
                                   plot_filename: str, rs_val: float, validations_dict: dict = None):
        """
        Generate a grid of 3D trajectory plots for multiple particles.
        Delegates to generate_all_particles_comparison for consistency.
        """
        # Extract trajectories from particle results
        particle_trajectories = {}
        for particle_name, result in particle_results.items():
            if isinstance(result, dict) and 'trajectory' in result:
                particle_trajectories[particle_name] = result['trajectory']
            elif isinstance(result, torch.Tensor):
                particle_trajectories[particle_name] = result
            else:
                # Try to extract tensor from tuple
                if isinstance(result, tuple) and len(result) > 0:
                    particle_trajectories[particle_name] = result[0]
        
        self.generate_all_particles_comparison(
            model, particle_trajectories, baseline_results, baseline_theories,
            plot_filename, rs_val, validations_dict
        )
    
    def generate_unified_multi_particle_plot(self, model: GravitationalTheory, particle_trajectories: dict,
                                           baseline_results: dict, baseline_theories: dict,
                                           plot_filename: str, rs_val: float, validations_dict: dict = None):
        """
        Generate a unified 3D plot showing all particle trajectories together.
        This creates a single 3D view with all particles visible at once.
        """
        # Set up the figure with dark background
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor('#0a0a0a')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#0a0a0a')
        
        # Draw event horizon
        self._draw_event_horizon(ax, rs_val, alpha=0.15)
        
        # Draw singularity
        self._draw_singularity(ax)
        
        legend_elements = []
        
        # Plot trajectories for each particle
        for particle_name, hist in particle_trajectories.items():
            particle = self.engine.particle_loader.get_particle(particle_name)
            particle_color = particle.color if hasattr(particle, 'color') else self.colors.get(particle_name, 'white')
            
            line = self._create_3d_trajectory_plot(
                ax, hist, particle_color, particle_name.capitalize(),
                linewidth=2.5, alpha=0.8
            )
            legend_elements.append(line)
        
        # Set labels and title
        ax.set_xlabel('X [Schwarzschild radii]', fontsize=14, labelpad=10)
        ax.set_ylabel('Y [Schwarzschild radii]', fontsize=14, labelpad=10)
        ax.set_zlabel('Z [Schwarzschild radii]', fontsize=14, labelpad=10)
        ax.set_title(f'{model.name} - All Particle Trajectories', fontsize=20, pad=20)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Set axis limits based on all trajectories
        all_x, all_y, all_z = [], [], []
        for hist in particle_trajectories.values():
            if hist is not None and len(hist) > 0:
                hist_np = self._to_numpy(hist)
                r = hist_np[:, 1]
                
                # Extract coordinates based on solver type
                if hist_np.shape[1] == 4:
                    # 4D symmetric solver output
                    theta = np.full_like(r, np.pi/2)
                    phi = hist_np[:, 2]
                else:
                    # 6D general solver output
                    theta = hist_np[:, 2] if hist_np.shape[1] > 2 else np.full_like(r, np.pi/2)
                    phi = hist_np[:, 3] if hist_np.shape[1] > 3 else np.zeros_like(r)
                
                x, y, z = self._cartesian_from_polar(r, theta, phi)
                all_x.extend(x)
                all_y.extend(y)
                all_z.extend(z)

        if not all_x: # If no valid trajectories
            lim = 10 * rs_val
            center_x, center_y, center_z = 0, 0, 0
        else:
            all_x, all_y, all_z = np.array(all_x), np.array(all_y), np.array(all_z)
            # Set limits to include all trajectories and ensure origin/horizon are visible
            if all_x:
                x_range = max(all_x) - min(all_x)
                y_range = max(all_y) - min(all_y)
                z_range = max(all_z) - min(all_z)
                max_range = max(x_range, y_range, z_range) / 2
                
                center_x = (max(all_x) + min(all_x)) / 2
                center_y = (max(all_y) + min(all_y)) / 2
                center_z = (max(all_z) + min(all_z)) / 2
                
                buffer = max_range * 0.2
                
                # Always include the origin (singularity) and event horizon
                lim = max(max_range + buffer,
                         abs(center_x) + max_range + buffer,
                         abs(center_y) + max_range + buffer,
                         abs(center_z) + max_range + buffer,
                         3 * rs_val)
                
                # Center view on origin
                ax.set_xlim([-lim, lim])
                ax.set_ylim([-lim, lim])
                ax.set_zlim([-lim, lim])
        
        # Grid styling
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Add legend
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                  markersize=10, label='Event Horizon', alpha=0.5),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                  markersize=15, label='Singularity')
        ])
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
                 facecolor='#1a1a1a', edgecolor='white', framealpha=0.8)
        
        # Save with high quality
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, facecolor='#0a0a0a',
                   bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        print(f"    Generated unified 3D plot: {plot_filename}")
    
    def _get_validation_status(self, validations_dict: dict) -> str:
        """Extract validation status string from validation results."""
        if not validations_dict:
            return "No validation data"
        
        passed_tests = []
        failed_tests = []
        
        for validator_name, result in validations_dict.items():
            if isinstance(result, dict) and 'passed' in result:
                if result['passed']:
                    passed_tests.append(validator_name)
                else:
                    failed_tests.append(validator_name)
        
        if failed_tests:
            return f"Failed: {', '.join(failed_tests)}"
        elif passed_tests:
            return f"Passed: {len(passed_tests)} tests"
        else:
            return "No test results"
    
    def _add_quantum_uncertainty_cloud(self, ax, r, phi, t, model):
        """Add quantum uncertainty visualization if applicable."""
        # Placeholder for quantum effects - can be enhanced
        pass
    
    def _add_uncertainty_inset(self, ax, x0, y0, z0, radius_actual, radius_scaled):
        """Add an inset showing uncertainty scale."""
        # Placeholder for uncertainty visualization
        pass
