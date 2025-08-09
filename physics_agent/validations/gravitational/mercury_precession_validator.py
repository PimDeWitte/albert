#!/usr/bin/env python3
"""
Mercury perihelion precession validator.
Tests gravitational theories against the observed precession of Mercury's orbit.
"""

import torch
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import os
from datetime import datetime

from ..base_validation import ObservationalValidator, ValidationResult
from physics_agent.base_theory import GravitationalTheory

# <reason>chain: Import observational data from centralized constants</reason>
from physics_agent.constants import MERCURY_PERIHELION_ADVANCE


class MercuryPrecessionValidator(ObservationalValidator):
    """
    Validates theories against observed perihelion advance of Mercury.
    
    The observed excess precession (after accounting for Newtonian perturbations
    from other planets) is 42.98 ± 0.04 arcseconds per century.
    
    This was one of the first major successes of General Relativity.
    """
    
    def __init__(self, engine=None):
        """Initialize the validator with a theory engine."""
        super().__init__(engine)
    
    def get_observational_data(self) -> Dict[str, Any]:
        """Get Mercury perihelion observational data"""
        # <reason>chain: Use centralized experimental data for consistency</reason>
        return {
            'object': 'Mercury',
            'measurement': 'Perihelion advance',
            'value': MERCURY_PERIHELION_ADVANCE['value'],
            'uncertainty': MERCURY_PERIHELION_ADVANCE['uncertainty'],
            'units': 'arcsec/century',
            'reference': MERCURY_PERIHELION_ADVANCE['reference'],
            'notes': MERCURY_PERIHELION_ADVANCE['notes']
        }
    
    def validate(self, theory: GravitationalTheory, verbose: bool = False) -> ValidationResult:
        """
        Validate theory against Mercury's precession.
        
        The calculation follows the formula for perihelion advance per orbit:
        Δφ = 6πGM/(c²a(1-e²))
        
        Then converts to arcseconds per century.
        """
        result = ValidationResult(self.name, theory.name)
        obs_data = self.get_observational_data()
        
        # Mercury orbital parameters
        a = 5.7909e10  # Semi-major axis (m)
        e = 0.2056     # Eccentricity
        T = 87.969     # Orbital period (days)
        
        # Physical constants
        M_sun = 1.989e30  # Solar mass (kg)
        c = 2.998e8       # Speed of light (m/s)
        G = 6.674e-11     # Gravitational constant
        
        if verbose:
            print(f"\nCalculating Mercury precession for {theory.name}...")
            print(f"  Semi-major axis: {a/1e9:.3f} Gm")
            print(f"  Eccentricity: {e:.4f}")
            print(f"  Orbital period: {T:.3f} days")
        
        # Convert to tensors on the engine's device
        r_peri = torch.tensor(a * (1 - e), device=self.engine.device, dtype=self.engine.dtype)  # Perihelion distance
        r_apo = torch.tensor(a * (1 + e), device=self.engine.device, dtype=self.engine.dtype)   # Aphelion distance
        M = torch.tensor(M_sun, device=self.engine.device, dtype=self.engine.dtype)
        
        # Sample the metric at several points along the orbit
        n_points = 100
        r_values = torch.linspace(r_peri.item(), r_apo.item(), n_points, 
                                 device=self.engine.device, dtype=self.engine.dtype)
        
        # Calculate average metric deviation from Newtonian
        total_deviation = 0.0
        
        for r in r_values:
            # Get metric components
            g_tt, g_rr, g_pp, g_tp = theory.get_metric(r.unsqueeze(0), M, c, G)
            
            # For weak field, g_tt ≈ -(1 + 2Φ/c²) where Φ = -GM/r
            # Deviation from Newtonian is approximately:
            # δ = g_tt + (1 - 2GM/(rc²))
            
            rs = 2 * G * M / c**2  # Schwarzschild radius
            g_tt_newton = -(1 - rs/r)
            
            deviation = (g_tt - g_tt_newton).abs().item()
            total_deviation += deviation
            
        avg_deviation = total_deviation / n_points
        
        # Classical GR formula for precession
        # Δφ = 6πGM/(c²a(1-e²)) radians per orbit
        delta_phi_per_orbit = 6 * np.pi * G * M_sun / (c**2 * a * (1 - e**2))
        
        # Apply theory-specific correction based on metric deviation
        # This is a simplified approach - a full calculation would integrate geodesics
        correction_factor = 1.0 + avg_deviation
        delta_phi_corrected = delta_phi_per_orbit * correction_factor
        
        # Convert to arcseconds per century
        orbits_per_century = 365.25 * 100 / T  # Number of orbits in a century
        precession_per_century_rad = delta_phi_corrected * orbits_per_century
        precession_per_century_arcsec = precession_per_century_rad * (180 / np.pi) * 3600
        
        # Store results
        result.observed_value = obs_data['value']
        result.predicted_value = float(precession_per_century_arcsec)
        result.units = obs_data['units']
        
        # Compute error
        result.error, result.error_percent = self.compute_error(
            result.observed_value, result.predicted_value
        )
        
        # Check if within observational uncertainty (3-sigma)
        tolerance = 3 * obs_data['uncertainty']  # 3-sigma = 0.12 arcsec/century
        relative_tolerance = tolerance / obs_data['value']  # ~0.28%
        
        result.passed = result.error_percent < (relative_tolerance * 100)
        
        if verbose:
            print(f"\nResults:")
            print(f"  Observed: {result.observed_value:.2f} ± {obs_data['uncertainty']:.2f} {result.units}")
            print(f"  Predicted: {result.predicted_value:.2f} {result.units}")
            print(f"  Error: {result.error:.3f} {result.units} ({result.error_percent:.2f}%)")
            print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
            
        result.notes = f"Tolerance: {relative_tolerance*100:.2f}% (3-sigma)"
        
        # <reason>chain: Solar oblateness correction is small and optional for this test</reason>
        # For Schwarzschild metric, we already have the prediction above
        
        # Generate trajectory plot if engine is available
        if self.engine is not None:
            self._create_orbit_plot(theory, M_sun, a, e, precession_per_century_arcsec, verbose)
        
        return result
    
    def _create_orbit_plot(self, theory: GravitationalTheory, M_sun: float, a: float, e: float, 
                          precession_arcsec: float, verbose: bool = False):
        """Create a plot of Mercury's orbit showing precession with detailed information"""
        try:
            if verbose:
                print("\n  Generating orbit trajectory plot...")
            
            # Create figure with two panels
            fig = plt.figure(figsize=(16, 8))
            ax1 = plt.subplot(121)  # Orbit visualization
            ax2 = plt.subplot(122)  # Comparison panel
            
            # Run a short trajectory simulation
            r_peri = a * (1 - e)  # Perihelion distance
            r_apo = a * (1 + e)   # Aphelion distance
            
            # Set up initial conditions for Mercury at perihelion
            # At perihelion, velocity is purely tangential
            v_peri = np.sqrt(self.engine.G_si * M_sun / a * (1 + e) / (1 - e))
            
            # Run for one full orbit to show precession
            T_orbit = 87.969 * 24 * 3600  # Orbital period in seconds
            n_orbits_display = 1  # Show one complete orbit
            
            # Use a reasonable timestep
            dt = T_orbit / 1000  # 1000 steps per orbit
            n_steps = int(n_orbits_display * T_orbit / dt)
            
            if verbose:
                print(f"    Generating {n_orbits_display} orbit visualization...")
            
            # Run trajectory (this is a simplified version - full implementation would use geodesic integrator)
            # For now, create an approximate elliptical orbit with precession
            times = np.linspace(0, n_orbits_display * 2 * np.pi, n_steps)
            
            # Orbital parameters
            p = a * (1 - e**2)  # Semi-latus rectum
            
            # Create arrays for the orbit
            r_vals = []
            phi_vals = []
            
            # Precession rate per orbit (in radians)
            # Convert from arcsec/century to radians/orbit
            # Mercury has ~415.2 orbits per century
            orbits_per_century = 415.2
            delta_phi_per_orbit = precession_arcsec * np.pi / (180 * 3600) / orbits_per_century
            
            # Also calculate GR prediction
            gr_precession_arcsec = 42.98  # Known GR value
            gr_delta_phi_per_orbit = gr_precession_arcsec * np.pi / (180 * 3600) / orbits_per_century
            
            # First orbit (no precession)
            for i, nu in enumerate(times):
                # True anomaly
                phi = nu
                # Radius from orbital equation
                r = p / (1 + e * np.cos(nu))
                
                r_vals.append(r)
                phi_vals.append(phi)
            
            r_vals_1 = np.array(r_vals)
            phi_vals_1 = np.array(phi_vals)
            
            # Second orbit (with precession) - for visualization
            r_vals_2 = []
            phi_vals_2 = []
            
            for i, nu in enumerate(times):
                # True anomaly with precession
                phi = nu + delta_phi_per_orbit
                # Radius from orbital equation
                r = p / (1 + e * np.cos(nu))
                
                r_vals_2.append(r)
                phi_vals_2.append(phi)
            
            r_vals_2 = np.array(r_vals_2)
            phi_vals_2 = np.array(phi_vals_2)
            
            # Convert to Cartesian for plotting
            x1 = r_vals_1 * np.cos(phi_vals_1) / 1e9  # Convert to Gm
            y1 = r_vals_1 * np.sin(phi_vals_1) / 1e9
            x2 = r_vals_2 * np.cos(phi_vals_2) / 1e9
            y2 = r_vals_2 * np.sin(phi_vals_2) / 1e9
            
            # LEFT PANEL: Orbit visualization
            # Plot first orbit
            ax1.plot(x1, y1, 'gray', linewidth=2, alpha=0.5, label='First orbit', linestyle='--')
            
            # Plot second orbit (with precession)
            ax1.plot(x2, y2, 'b-', linewidth=3, alpha=0.8, label=f'After 1 orbit ({theory.name})')
            
            # Mark perihelion points
            perihelion_idx = 0  # Start of orbit is perihelion
            ax1.plot(x1[perihelion_idx], y1[perihelion_idx], 'ro', markersize=12, 
                    label='Initial perihelion', zorder=5)
            ax1.plot(x2[perihelion_idx], y2[perihelion_idx], 'bs', markersize=12, 
                    label='Precessed perihelion', zorder=5)
            
            # Draw lines from Sun to perihelion points
            ax1.plot([0, x1[perihelion_idx]], [0, y1[perihelion_idx]], 'r-', 
                    alpha=0.7, linewidth=2, zorder=4)
            ax1.plot([0, x2[perihelion_idx]], [0, y2[perihelion_idx]], 'b-', 
                    alpha=0.7, linewidth=2, zorder=4)
            
            # Add precession angle visualization
            from matplotlib.patches import Wedge, FancyArrowPatch
            # Calculate angle in degrees for visualization (exaggerated)
            angle_deg = delta_phi_per_orbit * 180 / np.pi * 100  # Exaggerate by 100x for visibility
            
            # Draw wedge to show precession angle
            wedge = Wedge((0, 0), 20, 0, angle_deg, 
                         facecolor='red', alpha=0.2, edgecolor='red', linewidth=2)
            ax1.add_patch(wedge)
            
            # Add curved arrow to show direction
            arrow = FancyArrowPatch((15, 0), 
                                   (15 * np.cos(angle_deg * np.pi / 180), 
                                    15 * np.sin(angle_deg * np.pi / 180)),
                                   connectionstyle="arc3,rad=0.5", 
                                   arrowstyle='->', 
                                   mutation_scale=30, 
                                   linewidth=3, 
                                   color='red')
            ax1.add_patch(arrow)
            
            # Add the Sun
            sun = plt.Circle((0, 0), 2, color='#FDB813', edgecolor='#FD7E00', linewidth=3)
            ax1.add_patch(sun)
            ax1.text(0, 0, '☉', fontsize=30, ha='center', va='center', color='white')
            
            # Add precession angle label
            ax1.text(10, 5, f'Δφ = {delta_phi_per_orbit * 180 / np.pi * 3600:.3f}"\n(per orbit)', 
                    fontsize=14, color='red', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            # Add century accumulation note
            ax1.text(0.5, 0.02, f'After 100 years: {precession_arcsec:.2f}" precession', 
                    transform=ax1.transAxes, fontsize=12, ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            
            # Styling for left panel
            ax1.set_xlim(-80, 80)
            ax1.set_ylim(-60, 60)
            ax1.set_aspect('equal')
            ax1.set_xlabel('X (Gm)', fontsize=14)
            ax1.set_ylabel('Y (Gm)', fontsize=14)
            ax1.set_title('Mercury Orbit Precession', fontsize=16, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # RIGHT PANEL: Comparison chart
            # Bar chart comparing precessions
            observed_value = 42.98
            observed_error = 0.04
            
            categories = ['Observed', 'GR prediction', theory.name]
            values = [observed_value, gr_precession_arcsec, precession_arcsec]
            colors = ['green', 'gray', 'blue']
            
            bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            
            # Add error bar to observed value
            ax2.errorbar(0, observed_value, yerr=observed_error, fmt='none', 
                        ecolor='black', capsize=10, capthick=2)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{val:.2f}"', ha='center', va='bottom', fontsize=12, weight='bold')
            
            # Add pass/fail indicator
            error = abs(precession_arcsec - observed_value)
            tolerance = 3 * observed_error  # 3-sigma
            
            if error <= tolerance:
                status_color = 'green'
                status_text = '✓ PASS'
            else:
                status_color = 'red'
                status_text = '✗ FAIL'
            
            ax2.text(0.5, 0.95, status_text, transform=ax2.transAxes, 
                    fontsize=24, weight='bold', color=status_color,
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=status_color, alpha=0.2))
            
            # Error information
            error_percent = error / observed_value * 100
            tolerance_percent = tolerance / observed_value * 100
            ax2.text(0.5, 0.85, f'Error: {error:.2f}" ({error_percent:.1f}%)\n(Tolerance: ±{tolerance:.2f}" or {tolerance_percent:.1f}%)',
                    transform=ax2.transAxes, fontsize=12, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
            
            # Add Mercury info
            ax2.text(0.5, 0.70, f'Mercury orbit: {orbits_per_century:.0f} orbits/century\nEccentricity: {e:.4f}',
                    transform=ax2.transAxes, fontsize=11, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
            
            # Chart styling
            ax2.set_ylabel('Precession (arcsec/century)', fontsize=14)
            ax2.set_title('Perihelion Precession Comparison', fontsize=16, fontweight='bold')
            ax2.set_ylim(0, max(values) * 1.3)
            ax2.grid(True, axis='y', alpha=0.3)
            
            # Add reference line at observed value
            ax2.axhline(y=observed_value, color='green', linestyle='--', alpha=0.5, linewidth=2)
            ax2.axhspan(observed_value - tolerance, observed_value + tolerance, 
                       alpha=0.2, color='green', label='3σ range')
            
            # Final adjustments
            plt.tight_layout()
            
            # Save the plot
            output_dir = "physics_agent/latest_run"
            os.makedirs(output_dir, exist_ok=True)
            
            clean_name = theory.name.replace(' ', '_').replace('/', '_')
            filename = f"{output_dir}/mercury_orbit_{clean_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save copies to various directories
            import shutil
            
            # Save to docs directory
            docs_dir = "docs/latest_run"
            os.makedirs(docs_dir, exist_ok=True)
            docs_filename = f"{docs_dir}/mercury_precession_visualization.png"
            shutil.copy2(filename, docs_filename)
            
            # Save to validator_plots directory
            validator_plots_dir = "docs/latest_run/validator_plots"
            os.makedirs(validator_plots_dir, exist_ok=True)
            shutil.copy2(filename, f"{validator_plots_dir}/mercury_orbit.png")
            
            if verbose:
                print(f"    Mercury precession plot saved to {filename}")
            
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not create orbit plot: {e}")
            # Don't fail validation if plotting fails
            pass 