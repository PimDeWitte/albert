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
            'kerr': '#ff00ff',        # Magenta for Kerr baseline
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
        
        # Draw semi-transparent black sphere
        ax.plot_surface(x, y, z, color='black', alpha=alpha, shade=True)
        
    def _draw_singularity(self, ax):
        """Draw the singularity at the center."""
        ax.scatter([0], [0], [0], color='red', s=100, marker='*', 
                  edgecolors='white', linewidth=2, label='Singularity')
    
    def _create_3d_trajectory_plot(self, ax, hist, color, label, linewidth=2, alpha=1.0):
        """Create a 3D trajectory plot from history tensor."""
        hist_np = self._to_numpy(hist)
        
        # Extract coordinates
        t = hist_np[:, 0]
        r = hist_np[:, 1]
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
        
        # Add glow effect for the current position
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=50, alpha=1.0, 
                  edgecolors='white', linewidth=1)
        
        # Return line for legend
        return Line2D([0], [0], color=color, linewidth=linewidth, label=label)
    
    def generate_comparison_plot(self, model: GravitationalTheory, hist: Tensor, baseline_results: dict, 
                                baseline_theories: dict, plot_filename: str, rs_val: float, 
                                validations_dict: dict = None, particle_info: dict = None):
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
            plot_filename, rs_val, validations_dict
        )
    
    def generate_all_particles_comparison(self, model: GravitationalTheory, particle_trajectories: dict,
                                         baseline_results: dict, baseline_theories: dict,
                                         plot_filename: str, rs_val: float, validations_dict: dict = None):
        """
        Generate beautiful 3D comparison plots for all particles.
        Creates one plot per particle showing theory vs Kerr baseline.
        """
        # Set up the figure with dark background for aesthetics
        plt.style.use('dark_background')
        
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
                linewidth=3, alpha=0.9
            )
            
            # Plot Kerr baseline if available
            kerr_line = None
            for baseline_name, baseline_hist in baseline_results.items():
                if 'kerr' in baseline_name.lower() and 'newman' not in baseline_name.lower():
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
            
            # Set axis limits
            hist_np = self._to_numpy(hist)
            r_max = np.max(hist_np[:, 1]) * 1.2
            lim = max(r_max, 10 * rs_val)
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            ax.set_zlim([-lim, lim])
            
            # Grid styling
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#333333')
            ax.yaxis.pane.set_edgecolor('#333333')
            ax.zaxis.pane.set_edgecolor('#333333')
            
            # Add legend
            legend_elements = []
            if theory_line:
                legend_elements.append(theory_line)
            if kerr_line:
                legend_elements.append(kerr_line)
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='black', markersize=10,
                                        label='Event Horizon', alpha=0.5))
            legend_elements.append(Line2D([0], [0], marker='*', color='w',
                                        markerfacecolor='red', markersize=15,
                                        label='Singularity'))
            
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                     facecolor='#1a1a1a', edgecolor='white', framealpha=0.8)
            
            # Add particle properties text
            props_text = f"Type: {particle.particle_type}\\n"
            props_text += f"Orbit: {particle.orbital_parameters.get('orbit_type', 'unknown')}"
            ax.text2D(0.02, 0.98, props_text, transform=ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                              edgecolor='white', alpha=0.8))
        
        # Add main title
        fig.suptitle(f'{model.name} - Particle Trajectories Near Black Hole',
                    fontsize=24, y=0.98, color='white')
        
        # Add theory information
        theory_text = f"Theory: {model.name}\\n"
        if hasattr(model, 'get_parameters'):
            params = model.get_parameters()
            if params:
                param_str = ', '.join([f"{k}={v:.3g}" for k, v in params.items()])
                theory_text += f"Parameters: {param_str}"
        
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
        all_r = []
        for hist in particle_trajectories.values():
            hist_np = self._to_numpy(hist)
            all_r.extend(hist_np[:, 1])
        r_max = max(all_r) * 1.2
        lim = max(r_max, 10 * rs_val)
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
