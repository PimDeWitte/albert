#!/usr/bin/env python3
"""
Clean Theory Visualizer for 2D trajectory plots.
Focuses on generating clean 2D charts for trajectory analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class TheoryVisualizer:
    """Generate clean 2D visualizations for theory trajectories."""
    
    def __init__(self, engine):
        """Initialize the visualizer with physics engine."""
        self.engine = engine
        self.bh_mass = engine.M_SI if hasattr(engine, 'M_SI') else engine.bh_preset.mass_kg
        self.bh_radius = engine.M if hasattr(engine, 'M') else engine.length_scale
        
        # Set up matplotlib parameters
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        
        # Color palette
        self.colors = {
            'theory': '#3498db',      # Blue
            'baseline': '#2c3e50',    # Dark gray
            'kerr': '#e74c3c',        # Red
            'kerr_newman': '#e67e22', # Orange
            'horizon': '#34495e',     # Dark blue-gray
            'photon_sphere': '#f39c12', # Yellow
            'isco': '#27ae60'         # Green
        }
        
    def generate_trajectory_comparison(self, 
                                     theory_name: str,
                                     theory_traj: torch.Tensor,
                                     baseline_trajs: Dict[str, torch.Tensor],
                                     output_path: str) -> str:
        """Generate a comprehensive trajectory comparison plot."""
        
        # Convert tensors to numpy
        theory_hist = theory_traj.cpu().numpy() if isinstance(theory_traj, torch.Tensor) else theory_traj
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'{theory_name} - Trajectory Analysis', fontsize=16, y=0.98)
        
        # 1. 3D trajectory projection (2D view)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_trajectory_xy(ax1, theory_hist, baseline_trajs)
        
        # 2. Radial evolution
        ax2 = plt.subplot(2, 3, 2)
        self._plot_radial_evolution(ax2, theory_hist, baseline_trajs)
        
        # 3. Phase space (r vs dr/dt)
        ax3 = plt.subplot(2, 3, 3)
        self._plot_phase_space(ax3, theory_hist, baseline_trajs)
        
        # 4. Angular momentum
        ax4 = plt.subplot(2, 3, 4)
        self._plot_angular_momentum(ax4, theory_hist, baseline_trajs)
        
        # 5. Energy evolution
        ax5 = plt.subplot(2, 3, 5)
        self._plot_energy_evolution(ax5, theory_hist, baseline_trajs)
        
        # 6. Trajectory statistics
        ax6 = plt.subplot(2, 3, 6)
        self._plot_statistics(ax6, theory_hist, baseline_trajs, theory_name)
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _plot_trajectory_xy(self, ax, theory_hist, baseline_trajs):
        """Plot X-Y projection of trajectories."""
        # Extract coordinates
        t, r, theta, phi = theory_hist[:, 0], theory_hist[:, 1], theory_hist[:, 2], theory_hist[:, 3]
        
        # Convert to Cartesian
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        
        # Normalize by M
        x_M = x / self.bh_radius
        y_M = y / self.bh_radius
        
        # Plot theory trajectory
        ax.plot(x_M, y_M, color=self.colors['theory'], linewidth=1.5, label='Theory', alpha=0.8)
        
        # Plot baselines
        for name, traj in baseline_trajs.items():
            if traj is not None:
                r_b, theta_b, phi_b = traj[:, 1], traj[:, 2], traj[:, 3]
                x_b = r_b * np.sin(theta_b) * np.cos(phi_b) / self.bh_radius
                y_b = r_b * np.sin(theta_b) * np.sin(phi_b) / self.bh_radius
                color = self.colors.get(name.lower().replace(' ', '_'), '#95a5a6')
                ax.plot(x_b, y_b, '--', color=color, linewidth=1, label=name, alpha=0.6)
        
        # Add horizon and key radii
        horizon = plt.Circle((0, 0), 2, fill=True, color=self.colors['horizon'], alpha=0.8)
        photon_sphere = plt.Circle((0, 0), 3, fill=False, color=self.colors['photon_sphere'], 
                                  linestyle='--', linewidth=1.5)
        isco = plt.Circle((0, 0), 6, fill=False, color=self.colors['isco'], 
                         linestyle=':', linewidth=1.5)
        
        ax.add_patch(horizon)
        ax.add_patch(photon_sphere)
        ax.add_patch(isco)
        
        # Styling
        ax.set_xlabel('X/M')
        ax.set_ylabel('Y/M')
        ax.set_title('Orbital Trajectory (X-Y Plane)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Set limits
        max_r = max(15, np.max(np.sqrt(x_M**2 + y_M**2)) * 1.1)
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
    
    def _plot_radial_evolution(self, ax, theory_hist, baseline_trajs):
        """Plot radial coordinate over time."""
        t = theory_hist[:, 0]
        r = theory_hist[:, 1]
        
        # Normalize
        t_M = t / self.engine.time_scale * self.engine.c_si
        r_M = r / self.bh_radius
        
        # Plot theory
        ax.plot(t_M, r_M, color=self.colors['theory'], linewidth=1.5, label='Theory')
        
        # Plot baselines
        for name, traj in baseline_trajs.items():
            if traj is not None:
                t_b = traj[:, 0] / self.engine.time_scale * self.engine.c_si
                r_b = traj[:, 1] / self.bh_radius
                color = self.colors.get(name.lower().replace(' ', '_'), '#95a5a6')
                ax.plot(t_b, r_b, '--', color=color, linewidth=1, label=name, alpha=0.6)
        
        # Add key radii
        ax.axhline(y=2, color=self.colors['horizon'], linestyle='-', alpha=0.5, label='Horizon')
        ax.axhline(y=3, color=self.colors['photon_sphere'], linestyle='--', alpha=0.5, label='Photon sphere')
        ax.axhline(y=6, color=self.colors['isco'], linestyle=':', alpha=0.5, label='ISCO')
        
        ax.set_xlabel('t/M')
        ax.set_ylabel('r/M')
        ax.set_title('Radial Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    def _plot_phase_space(self, ax, theory_hist, baseline_trajs):
        """Plot phase space diagram (r vs dr/dt)."""
        t = theory_hist[:, 0]
        r = theory_hist[:, 1]
        
        # Calculate dr/dt safely
        if len(t) > 1:
            dt = np.diff(t)
            if np.any(dt > 0):
                dr_dt = np.zeros_like(r)
                dr_dt[0] = (r[1] - r[0]) / dt[0] if dt[0] > 0 else 0
                for i in range(1, len(r) - 1):
                    if t[i+1] - t[i-1] > 0:
                        dr_dt[i] = (r[i+1] - r[i-1]) / (t[i+1] - t[i-1])
                if len(dt) > 0 and dt[-1] > 0:
                    dr_dt[-1] = (r[-1] - r[-2]) / dt[-1]
            else:
                dr_dt = np.zeros_like(r)
        else:
            dr_dt = np.zeros_like(r)
        
        # Normalize
        r_M = r / self.bh_radius
        dr_dt_c = dr_dt / self.engine.c_si
        
        # Plot
        ax.plot(r_M[1:], dr_dt_c[1:], color=self.colors['theory'], linewidth=1.5, label='Theory')
        ax.scatter(r_M[0], dr_dt_c[0], color='green', s=100, zorder=5, label='Start')
        ax.scatter(r_M[-1], dr_dt_c[-1], color='red', s=100, zorder=5, label='End')
        
        ax.set_xlabel('r/M')
        ax.set_ylabel('(dr/dt)/c')
        ax.set_title('Phase Space')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    def _plot_angular_momentum(self, ax, theory_hist, baseline_trajs):
        """Plot angular momentum evolution."""
        t = theory_hist[:, 0]
        r = theory_hist[:, 1]
        
        # For circular orbits, L = r * v_phi
        if theory_hist.shape[1] >= 6:
            # If we have velocity components
            u_phi = theory_hist[:, 5]
            L = r * u_phi
        else:
            # Estimate from phi evolution
            phi = theory_hist[:, 3]
            dphi_dt = np.gradient(phi, t)
            L = r**2 * dphi_dt
        
        # Normalize
        t_M = t / self.engine.time_scale * self.engine.c_si
        L_norm = L / (self.engine.G_SI * self.bh_mass / self.engine.c_si)
        
        ax.plot(t_M, L_norm, color=self.colors['theory'], linewidth=1.5)
        ax.set_xlabel('t/M')
        ax.set_ylabel('L/(GM/c)')
        ax.set_title('Angular Momentum')
        ax.grid(True, alpha=0.3)
    
    def _plot_energy_evolution(self, ax, theory_hist, baseline_trajs):
        """Plot energy evolution."""
        t = theory_hist[:, 0]
        r = theory_hist[:, 1]
        
        # For test particle in Schwarzschild: E = sqrt(1 - 2M/r + L²/(r²))
        # Simplified version for circular orbits
        E_approx = np.sqrt(1 - 2 * self.bh_radius / r)
        
        t_M = t / self.engine.time_scale * self.engine.c_si
        
        ax.plot(t_M, E_approx, color=self.colors['theory'], linewidth=1.5)
        ax.set_xlabel('t/M')
        ax.set_ylabel('E/mc²')
        ax.set_title('Energy per unit mass')
        ax.grid(True, alpha=0.3)
    
    def _plot_statistics(self, ax, theory_hist, baseline_trajs, theory_name):
        """Plot trajectory statistics panel."""
        ax.axis('off')
        
        # Calculate statistics
        r = theory_hist[:, 1]
        phi = theory_hist[:, 3]
        t = theory_hist[:, 0]
        
        r_M = r / self.bh_radius
        
        # Basic stats
        r_min = np.min(r_M)
        r_max = np.max(r_M)
        r_mean = np.mean(r_M)
        
        # Number of orbits
        n_orbits = np.abs(phi[-1] - phi[0]) / (2 * np.pi)
        
        # Total distance traveled
        if len(r) > 1:
            dr = np.diff(r)
            dphi = np.diff(phi)
            dtheta = np.diff(theory_hist[:, 2])
            
            # Spherical distance elements
            ds = np.sqrt(dr**2 + r[:-1]**2 * dtheta**2 + r[:-1]**2 * np.sin(theory_hist[:-1, 2])**2 * dphi**2)
            total_distance = np.sum(ds) / self.bh_radius
        else:
            total_distance = 0
        
        # Duration
        duration = (t[-1] - t[0]) / self.engine.time_scale * self.engine.c_si
        
        # Create text
        stats_text = f"""
{theory_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━
Trajectory Statistics:

Radial Range: {r_min:.2f} - {r_max:.2f} M
Mean Radius: {r_mean:.2f} M
Min approach: {r_min:.2f} M

Orbits completed: {n_orbits:.2f}
Total distance: {total_distance:.1f} M
Duration: {duration:.2f} M

Initial r: {r_M[0]:.2f} M
Final r: {r_M[-1]:.2f} M
Steps: {len(r)}
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    def generate_multi_particle_comparison(self,
                                         theory_name: str,
                                         particle_trajs: Dict[str, torch.Tensor],
                                         output_path: str) -> str:
        """Generate comparison plot for multiple particles."""
        
        n_particles = len(particle_trajs)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{theory_name} - Multi-Particle Trajectories', fontsize=16)
        
        axes = axes.flatten()
        
        for idx, (particle_name, traj) in enumerate(particle_trajs.items()):
            if idx >= 4:
                break
                
            ax = axes[idx]
            hist = traj.cpu().numpy() if isinstance(traj, torch.Tensor) else traj
            
            # Plot X-Y trajectory
            r, theta, phi = hist[:, 1], hist[:, 2], hist[:, 3]
            x = r * np.sin(theta) * np.cos(phi) / self.bh_radius
            y = r * np.sin(theta) * np.sin(phi) / self.bh_radius
            
            ax.plot(x, y, linewidth=1.5, label=particle_name)
            
            # Add black hole
            horizon = plt.Circle((0, 0), 2, fill=True, color='black', alpha=0.8)
            ax.add_patch(horizon)
            
            ax.set_xlabel('X/M')
            ax.set_ylabel('Y/M')
            ax.set_title(f'{particle_name.capitalize()} Trajectory')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            max_r = max(10, np.max(np.sqrt(x**2 + y**2)) * 1.1)
            ax.set_xlim(-max_r, max_r)
            ax.set_ylim(-max_r, max_r)
        
        # Hide unused subplots
        for idx in range(n_particles, 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def calculate_trajectory_loss(self, theory_traj: torch.Tensor, baseline_traj: torch.Tensor) -> float:
        """Calculate trajectory loss between theory and baseline."""
        if theory_traj is None or baseline_traj is None:
            return float('inf')
        
        # Ensure same length
        min_len = min(len(theory_traj), len(baseline_traj))
        theory_traj = theory_traj[:min_len]
        baseline_traj = baseline_traj[:min_len]
        
        # Extract r, theta, phi
        r1, theta1, phi1 = theory_traj[:, 1:4].T
        r2, theta2, phi2 = baseline_traj[:, 1:4].T
        
        # Convert to Cartesian for comparison
        x1 = r1 * torch.sin(theta1) * torch.cos(phi1)
        y1 = r1 * torch.sin(theta1) * torch.sin(phi1)
        z1 = r1 * torch.cos(theta1)
        
        x2 = r2 * torch.sin(theta2) * torch.cos(phi2)
        y2 = r2 * torch.sin(theta2) * torch.sin(phi2)
        z2 = r2 * torch.cos(theta2)
        
        # Calculate MSE
        mse = torch.mean((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        
        return float(mse.item())