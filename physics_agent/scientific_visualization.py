#!/usr/bin/env python3
"""
Scientific Visualization Suite for Gravitational Theories.

This module provides a `ScientificVisualizer` class that creates publication-quality
plots for analyzing and comparing gravitational theories, focusing on metrics
relevant to field theory research.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Assuming base_theory and theory_loader are in this path
from physics_agent.base_theory import GravitationalTheory

class ScientificVisualizer:
    """
    Creates scientific-standard visualizations for gravitational theories.
    """
    def __init__(self, M, c, G, device, dtype, figsize=(12, 9), dpi=150):
        self.M = M
        self.c = c
        self.G = G
        self.RS = 2 * G * M / c**2
        self.device = device
        self.dtype = dtype
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use('dark_background')

    def _to_numpy(self, tensor):
        """Helper to convert tensor to numpy array, detaching from graph."""
        return tensor.cpu().detach().numpy()

    def create_essential_plots(self, model: GravitationalTheory, baseline_theories: dict, hist: torch.Tensor,
                             plot_filename: str, energy: float, angular_momentum: float):
        """
        Creates an essential 4-panel plot including:
        1. Effective Potential
        2. Trajectory (r vs phi)
        3. Radial Time Evolution (r vs t)
        4. Gravitational Waveform (h_plus)
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

        # 1. Effective Potential
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_effective_potential(ax1, model, baseline_theories, energy, angular_momentum)

        # 2. Trajectory
        ax2 = fig.add_subplot(gs[0, 1], polar=True)
        self._plot_trajectory(ax2, hist)

        # 3. Radial Time Evolution
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_radial_evolution(ax3, hist)

        # 4. Gravitational Waveform
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_gravitational_waveform(ax4, hist)

        fig.suptitle(f"Essential Analysis: {model.name}", fontsize=18)
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=self.dpi)
        plt.close(fig)

    def create_parameter_space_plot(self, model: GravitationalTheory, baseline_theories: dict, plot_filename: str):
        """
        Creates a plot showing constraints on the theory's parameter space.
        (This is a placeholder and will need a more sophisticated implementation)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(f"Parameter Space Constraints: {model.name}")
        ax.set_xlabel("Parameter 1 (e.g., alpha)")
        ax.set_ylabel("Parameter 2 (e.g., beta)")
        ax.text(0.5, 0.5, "Constraint analysis not yet implemented.\nThis plot will show regions ruled out by\n"
                         "observations like PPN constraints, GW events, etc.",
                ha='center', va='center', transform=ax.transAxes, color='yellow')
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=self.dpi)
        plt.close(fig)

    def _plot_effective_potential(self, ax, model, baselines, E, L):
        r_vals = torch.logspace(np.log10(self.RS.item() * 0.9), np.log10(self.RS.item() * 50), 500,
                              device=self.device, dtype=self.dtype)
        
        # Plot for main theory
        V_eff = self._calculate_V_eff(model, r_vals, L)
        V_eff_np = self._to_numpy(V_eff)
        
        # Filter out non-finite values for plotting
        finite_mask = np.isfinite(V_eff_np)
        
        ax.plot(self._to_numpy(r_vals / self.RS)[finite_mask], V_eff_np[finite_mask], 'r-', label=model.name, zorder=10)

        # Plot for baselines
        for name, baseline_model in baselines.items():
            V_eff_base = self._calculate_V_eff(baseline_model, r_vals, L)
            ax.plot(self._to_numpy(r_vals / self.RS), self._to_numpy(V_eff_base), '--', label=name, alpha=0.7)

        ax.axhline(E, color='cyan', linestyle=':', label='Particle Energy')
        ax.set_title("Effective Potential")
        ax.set_xlabel("r/rs")
        ax.set_ylabel("V_eff(r)")
        
        # Set y-limits safely
        min_V = V_eff_np[finite_mask].min() if np.any(finite_mask) else -1
        max_V = V_eff_np[finite_mask].max() if np.any(finite_mask) else 1
        ax.set_ylim(min(min_V, E) - 0.1, max(max_V, E) + 0.1)

        ax.legend()

    def _calculate_V_eff(self, model, r, L):
        g_tt, _, g_pp, g_tp = model.get_metric(r, self.M, self.c, self.G)
        return torch.sqrt(-g_tt * (1 + L**2 / g_pp))

    def _plot_trajectory(self, ax, hist):
        r = self._to_numpy(hist[:, 1]) / self.RS.item()
        phi = self._to_numpy(hist[:, 2])
        ax.plot(phi, r)
        ax.set_title("Trajectory (r vs. phi)")
        ax.set_xlabel("Angle (rad)")
        ax.set_ylabel("Radius (r/rs)")

    def _plot_radial_evolution(self, ax, hist):
        t = self._to_numpy(hist[:, 0])
        r = self._to_numpy(hist[:, 1]) / self.RS.item()
        ax.plot(t, r)
        ax.set_title("Radial Evolution (r vs. t)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Radius (r/rs)")

    def _plot_gravitational_waveform(self, ax, hist):
        # Simplified quadrupole formula for h_plus
        t = hist[:, 0]
        r = hist[:, 1]
        phi = hist[:, 2]
        
        # Second derivative of quadrupole moment (approximate)
        I_xx = r**2 * torch.cos(phi)**2
        I_yy = r**2 * torch.sin(phi)**2
        Q = I_xx - I_yy
        
        # Use numpy gradient for derivatives on numpy arrays
        t_np = self._to_numpy(t)
        Q_np = self._to_numpy(Q)
        dt = np.mean(np.diff(t_np))
        d2Q_dt2 = np.gradient(np.gradient(Q_np, dt), dt)
        
        # Assume observer is at a distance D on the z-axis
        h_plus = (1 / 1e21) * d2Q_dt2 # Arbitrary distance scaling
        
        ax.plot(t_np, h_plus)
        ax.set_title("Gravitational Waveform (h+)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strain (h+)") 