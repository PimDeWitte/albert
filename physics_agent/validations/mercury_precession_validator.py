#!/usr/bin/env python3
"""
Mercury perihelion precession validator.
Tests gravitational theories against the observed precession of Mercury's orbit.
"""

import torch
import numpy as np
from typing import Dict, Any

from .base_validation import ObservationalValidator, ValidationResult
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
        
        return result 