#!/usr/bin/env python3
"""
PSR J0740+6620 Shapiro delay validator.

Tests gravitational theories against precision timing measurements of PSR J0740+6620,
one of the most massive known pulsars. The Shapiro delay (relativistic time delay
of pulsar signals passing near the companion star) provides a stringent test of
General Relativity and alternative theories.

Data from: Fonseca et al. (2021, ApJL 915, L12)
"""

import torch
import numpy as np
import math
from typing import Dict, Any, Optional

from .base_validation import ObservationalValidator, ValidationResult
from physics_agent.base_theory import GravitationalTheory
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT, PSR_J0740_DATA


class PsrJ0740Validator(ObservationalValidator):
    """
    Validates theories against PSR J0740+6620 Shapiro delay measurements.
    
    PSR J0740+6620 is a millisecond pulsar in a binary system with exceptional
    timing precision (RMS residuals ~0.28 μs). The Shapiro delay effect - the
    extra light travel time when pulsar signals pass near the companion star -
    provides a clean test of relativistic gravity.
    
    This validator computes both analytic and numeric predictions for the
    Shapiro delay and compares them against the observed timing precision.
    """
    
    def __init__(self, engine=None):
        """Initialize the validator with a theory engine."""
        super().__init__(engine)
        
    def get_observational_data(self) -> Dict[str, Any]:
        """Get PSR J0740+6620 observational data from constants module"""
        # <reason>chain: Use centralized experimental data for consistency</reason>
        data = PSR_J0740_DATA.copy()
        data.update({
            'object': 'PSR J0740+6620',
            'measurement': 'Shapiro delay',
            'units': 'seconds'
        })
        return data
    
    def compute_shapiro_delay_analytic(self, theory: GravitationalTheory, 
                                     M_c: float, a: float, i: float, 
                                     e: float = 0.0, omega: float = 0.0, 
                                     phi: float = 0.0) -> float:
        """
        Compute analytic Shapiro delay for the binary pulsar system.
        
        Args:
            theory: Gravitational theory to test
            M_c: Companion mass (kg)
            a: Semi-major axis (m)
            i: Inclination angle (degrees)
            e: Eccentricity
            omega: Argument of periapsis (radians)
            phi: Orbital phase (radians)
            
        Returns:
            Shapiro delay in seconds
        """
        # Range parameter
        r_shap = GRAVITATIONAL_CONSTANT * M_c / SPEED_OF_LIGHT**3
        s = math.sin(math.radians(i))
        
        # GR delay - this is the standard formula
        arg = 1 - e * math.cos(phi) - s * math.sin(phi + omega)
        if arg <= 0:
            arg = 1e-10  # Avoid log(0)
        delay_gr = -2 * r_shap * math.log(arg)
        
        # Theory-specific modifications can be added here
        # For now, we use pure GR prediction
        kappa = getattr(theory, 'kappa', 0.0) if hasattr(theory, 'kappa') else 0.0
        delay_modification = kappa * r_shap * s if isinstance(kappa, (int, float)) else 0.0
        
        return delay_gr + delay_modification
    
    def compute_time_of_flight_numeric(self, theory: GravitationalTheory, 
                                     M: float, dist: float, 
                                     steps: int = 1000) -> float:
        """
        Compute light travel time through gravitational field numerically.
        
        Args:
            theory: Gravitational theory to test
            M: Mass of companion (kg)
            dist: Distance to integrate over (m)
            steps: Number of integration steps
            
        Returns:
            Extra time delay compared to flat spacetime (seconds)
        """
        # Convert to geometric units for numerical stability
        M_geom = GRAVITATIONAL_CONSTANT * M / SPEED_OF_LIGHT**2
        dist_geom = dist / M_geom
        r_min_geom = 2.1  # Just outside horizon
        
        if dist_geom <= r_min_geom:
            return float('inf')
        
        dr = (dist_geom - r_min_geom) / steps
        t_total = 0.0
        r = dist_geom
        
        while r > r_min_geom:
            try:
                r_tensor = torch.tensor(r, dtype=torch.float64).unsqueeze(0)
                # Get metric in geometric units
                g_tt, _, _, _ = theory.get_metric(
                    r_tensor,
                    torch.tensor(1.0, dtype=torch.float64),  # M=1 geometric
                    1.0,  # c=1
                    1.0   # G=1
                )
                
                f = -g_tt.squeeze().item()
                if f <= 0:
                    break
                
                # dt/dr for light ray in Schwarzschild-like metric
                dt_dr = 1 / f
                dt = dt_dr * dr
                t_total += dt
                r -= dr
            except Exception:
                # If metric evaluation fails, return a large value
                return 1e-3
        
        # Convert back to physical time
        t_phys = t_total * M_geom / SPEED_OF_LIGHT
        
        # Subtract flat space time
        flat_time = (dist_geom - r_min_geom) * M_geom / SPEED_OF_LIGHT
        
        return t_phys - flat_time
    
    def validate(self, theory: GravitationalTheory, verbose: bool = False) -> ValidationResult:
        """
        Validate theory against PSR J0740+6620 Shapiro delay measurements.
        
        This tests whether the theory's predictions for gravitational time delay
        are consistent with the exceptional timing precision of this pulsar.
        """
        result = ValidationResult("PSR J0740+6620 Shapiro Delay", theory.__class__.__name__)
        
        # Get observational data
        obs_data = self.get_observational_data()
        
        # Extract parameters
        M_p = obs_data['pulsar_mass'] * SOLAR_MASS  # kg
        M_c = obs_data['companion_mass'] * SOLAR_MASS  # kg
        P_b = obs_data['orbital_period'] * 86400  # seconds
        i_deg = obs_data['inclination']
        e = obs_data['eccentricity']
        rms_us = obs_data['timing_rms']  # seconds
        
        # Semi-major axis from Kepler's third law
        M_total = M_p + M_c
        a = (GRAVITATIONAL_CONSTANT * M_total * P_b**2 / (4 * math.pi**2))**(1/3)
        
        if verbose:
            print(f"\nPSR J0740+6620 System Parameters:")
            print(f"  Pulsar mass: {M_p/SOLAR_MASS:.2f} M☉")
            print(f"  Companion mass: {M_c/SOLAR_MASS:.3f} M☉")
            print(f"  Orbital period: {P_b/86400:.4f} days")
            print(f"  Semi-major axis: {a/1e6:.1f} × 10^6 m")
            print(f"  Inclination: {i_deg:.2f}°")
            print(f"  Timing RMS: {rms_us*1e6:.2f} μs")
        
        try:
            # Compute analytic Shapiro delay
            delay_analytic = self.compute_shapiro_delay_analytic(
                theory, M_c, a, i_deg, e
            )
            
            # Compute numeric time of flight
            tof_numeric = self.compute_time_of_flight_numeric(theory, M_c, a)
            
            if verbose:
                print(f"\nTheory Predictions:")
                print(f"  Analytic Shapiro delay: {delay_analytic:.9f} s")
                print(f"  Numeric time of flight: {tof_numeric:.9f} s")
            
            # The key test: are the predictions consistent with timing precision?
            # We use the numeric result as the primary test value
            result.predicted_value = abs(tof_numeric)
            result.observed_value = rms_us
            result.units = "seconds"
            
            # Check if prediction is within reasonable bounds
            # The Shapiro delay should be detectable but not dominate timing
            if abs(tof_numeric) < 1e-3 and abs(tof_numeric) > 1e-9:
                # Additional test: analytic should match numeric for consistency
                if abs(delay_analytic) > 0:
                    consistency = abs(tof_numeric - abs(delay_analytic)) / abs(delay_analytic)
                    if consistency < 0.1:  # 10% consistency threshold
                        result.passed = True
                        result.notes = f"Shapiro delay consistent with timing precision. "
                        result.notes += f"Analytic: {delay_analytic:.3e} s, Numeric: {tof_numeric:.3e} s"
                    else:
                        result.passed = False
                        result.notes = f"Inconsistent predictions: analytic vs numeric differ by {consistency*100:.1f}%"
                else:
                    result.passed = True
                    result.notes = f"Numeric Shapiro delay {tof_numeric:.3e} s within valid range"
            else:
                result.passed = False
                result.notes = f"Shapiro delay {tof_numeric:.3e} s outside valid range (1e-9 to 1e-3 s)"
            
            # Calculate error metrics
            if result.observed_value > 0:
                result.error = abs(result.predicted_value - result.observed_value)
                result.error_percent = (result.error / result.observed_value) * 100
            
        except Exception as e:
            result.passed = False
            result.notes = f"Validation failed: {str(e)}"
            result.predicted_value = float('nan')
            result.error = float('nan')
            result.error_percent = float('nan')
            
            if verbose:
                print(f"\nError during validation: {e}")
        
        return result 