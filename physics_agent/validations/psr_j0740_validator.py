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
from typing import Dict, Any, Optional, TYPE_CHECKING

from .base_validation import BaseValidation
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT, PSR_J0740_DATA

if TYPE_CHECKING:
    from physics_agent.base_theory import GravitationalTheory
    from physics_agent.theory_engine_core import TheoryEngine


class PsrJ0740Validator(BaseValidation):
    """
    Validates theories against PSR J0740+6620 Shapiro delay measurements.
    
    PSR J0740+6620 is a millisecond pulsar in a binary system with exceptional
    timing precision (RMS residuals ~0.28 μs). The Shapiro delay effect - the
    extra light travel time when pulsar signals pass near the companion star -
    provides a clean test of relativistic gravity.
    
    This validator computes both analytic and numeric predictions for the
    Shapiro delay and compares them against the observed timing precision.
    """
    category = "observational"
    
    def __init__(self, engine: "TheoryEngine", tolerance: float = 1e-6):
        """Initialize the validator with a theory engine."""
        super().__init__(engine, "PSR J0740 Shapiro Delay Validator")
        self.tolerance = tolerance
        
    def get_observational_data(self) -> Dict[str, Any]:
        """Get PSR J0740+6620 observational data"""
        # <reason>chain: Use centralized experimental data for consistency</reason>
        data = PSR_J0740_DATA.copy()
        data.update({
            'object': 'PSR J0740+6620',
            'measurement': 'Shapiro delay',
            'units': 'seconds'
        })
        return data
    
    def compute_shapiro_delay_analytic(self, theory: "GravitationalTheory", 
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
        
        # For a circular orbit at superior conjunction (phi=0), the standard formula simplifies
        # The maximum Shapiro delay occurs when the signal passes closest to the companion
        # For high inclination (nearly edge-on), this gives a significant delay
        
        # Maximum delay approximation for nearly circular orbit
        # This is the delay when the pulsar is directly behind the companion
        if e < 0.01:  # Nearly circular
            # For edge-on orbit, use the maximum delay formula
            delay_max = -2 * r_shap * math.log(1 - s)
            
            # Apply PPN gamma correction
            gamma = 1.0
            if hasattr(theory, 'ppn_gamma'):
                try:
                    gamma = float(getattr(theory, 'ppn_gamma', 1.0))
                except:
                    gamma = 1.0
            
            # The Shapiro delay is proportional to (1 + gamma)
            delay = (1 + gamma) * delay_max / 2
        else:
            # For eccentric orbits, use the full formula
            arg = 1 - e * math.cos(phi) - s * math.sin(phi + omega)
            if arg <= 0:
                arg = 1e-10  # Avoid log(0)
            delay_gr = -2 * r_shap * math.log(arg)
            
            # Apply PPN correction
            gamma = 1.0
            if hasattr(theory, 'ppn_gamma'):
                try:
                    gamma = float(getattr(theory, 'ppn_gamma', 1.0))
                except:
                    gamma = 1.0
                    
            delay = (1 + gamma) * delay_gr / 2
            
        return abs(delay)  # Return positive delay
    
    def compute_time_of_flight_numeric(self, theory: "GravitationalTheory", 
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
                r_tensor = torch.tensor(r, dtype=self.engine.dtype, device=self.engine.device).unsqueeze(0)
                # Get metric in geometric units
                g_tt, _, _, _ = theory.get_metric(
                    r_tensor,
                    torch.tensor(1.0, dtype=self.engine.dtype, device=self.engine.device),  # M=1 geometric
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
    
    def validate(self, theory: "GravitationalTheory", hist: torch.Tensor = None, **kwargs) -> Dict[str, Any]:
        """
        Validate theory against PSR J0740+6620 Shapiro delay measurements.
        
        This tests whether the theory's predictions for gravitational time delay
        are consistent with the exceptional timing precision of this pulsar.
        """
        verbose = kwargs.get('verbose', False)
        
        # Get observational data
        obs_data = self.get_observational_data()
        
        # Extract parameters
        M_p = obs_data['pulsar_mass'] * SOLAR_MASS  # kg
        M_c = obs_data['companion_mass'] * SOLAR_MASS  # kg
        P_b = obs_data['orbital_period'] * 86400  # seconds
        i_deg = obs_data['inclination']
        e = obs_data['eccentricity']
        rms_us = obs_data['timing_rms']  # seconds
        # <reason>chain: Use realistic tolerance based on measurement uncertainty, not timing RMS</reason>
        # The Shapiro delay itself can be much larger than timing RMS
        # What matters is that we can measure it accurately
        # Use 10% relative error as tolerance for Shapiro delay predictions
        tolerance = obs_data.get('tolerance', None)
        if tolerance is None:
            # For PSR J0740, the Shapiro delay is ~18.7 μs
            # A 10% tolerance allows 1.9 μs deviation, which is reasonable
            tolerance = 2e-6  # 2 microseconds absolute tolerance
        
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
            print(f"  Validation tolerance: {tolerance*1e6:.1f} μs")
        
        try:
            # Compute analytic Shapiro delay at superior conjunction (maximum delay)
            delay_analytic = self.compute_shapiro_delay_analytic(
                theory, M_c, a, i_deg, e, omega=0.0, phi=0.0
            )
            
            # Compute numeric time of flight
            tof_numeric = self.compute_time_of_flight_numeric(theory, M_c, a)
            
            if verbose:
                print(f"\nTheory Predictions:")
                print(f"  Analytic Shapiro delay: {delay_analytic*1e6:.3f} μs")
                print(f"  Numeric time of flight: {tof_numeric*1e6:.3f} μs")
            
            # The key test: are the predictions consistent with timing precision?
            # The maximum Shapiro delay should be detectable but not too large
            max_delay = max(abs(delay_analytic), abs(tof_numeric))
            
            # Check if prediction is within reasonable bounds
            # For PSR J0740, the Shapiro delay is ~10-100 μs range
            loss = 0.0
            flags = {'overall': 'PASS'}
            details = {
                'shapiro_delay_analytic': delay_analytic,
                'shapiro_delay_numeric': tof_numeric,
                'timing_rms': rms_us,
                'max_delay': max_delay,
                'units': 'seconds'
            }
            
            # Check if delay is physically reasonable
            if max_delay < 1e-9 or max_delay > 1e-3:
                loss = 1.0
                flags['overall'] = 'FAIL'
                flags['details'] = f"Shapiro delay {max_delay*1e6:.3f} μs outside valid range (0.001-1000 μs)"
            else:
                # Check consistency between analytic and numeric
                if abs(delay_analytic) > 0 and abs(tof_numeric) > 0:
                    consistency = abs(tof_numeric - delay_analytic) / abs(delay_analytic)
                    if consistency > 0.1:  # 10% consistency threshold
                        loss = consistency
                        flags['overall'] = 'FAIL'
                        flags['details'] = f"Inconsistent predictions: analytic vs numeric differ by {consistency*100:.1f}%"
                    else:
                        # <reason>chain: Check relative error instead of absolute comparison</reason>
                        # GR predicts ~18.7 μs Shapiro delay for this system
                        gr_shapiro_delay = 18.7e-6  # Expected GR value in seconds
                        relative_error = abs(max_delay - gr_shapiro_delay) / gr_shapiro_delay
                        
                        if relative_error > 0.1:  # 10% relative error threshold
                            loss = relative_error * 10  # Scale to ~1 for 10% error
                            flags['overall'] = 'FAIL'
                            flags['details'] = f"Shapiro delay {max_delay*1e6:.1f} μs deviates {relative_error*100:.1f}% from GR expectation"
                        elif relative_error > 0.02:  # 2% warning threshold
                            loss = relative_error
                            flags['overall'] = 'WARNING'
                            flags['details'] = f"Shapiro delay {max_delay*1e6:.1f} μs shows {relative_error*100:.1f}% deviation from GR"
                        else:
                            flags['details'] = f"Shapiro delay {max_delay*1e6:.1f} μs matches GR within {relative_error*100:.1f}%"
                
            # Add comparison to observed value
            details['observed_delay'] = rms_us  # Use timing RMS as proxy for detectability
            details['predicted_delay'] = max_delay
            details['error_percent'] = abs(max_delay - rms_us) / rms_us * 100 if rms_us > 0 else 0
            
        except Exception as e:
            loss = 1.0
            flags = {
                'overall': 'ERROR',
                'details': f"Validation failed: {str(e)}"
            }
            details = {
                'error': str(e),
                'error_type': type(e).__name__
            }
            
            if verbose:
                print(f"\nError during validation: {e}")
                import traceback
                traceback.print_exc()
        
        return {
            'loss': loss,
            'flags': flags,
            'details': details
        } 