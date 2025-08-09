#!/usr/bin/env python3
"""
Cassini PPN-γ parameter validator.
Tests theories against the Cassini spacecraft's measurement of the PPN γ parameter.
"""

import numpy as np
from typing import Dict, Any

from physics_agent.validations import ObservationalValidator, ValidationResult
from physics_agent.base_theory import GravitationalTheory


class CassiniPPNValidator(ObservationalValidator):
    """
    Validates theories against Cassini's precise measurement of the PPN γ parameter.
    
    The Parameterized Post-Newtonian (PPN) formalism characterizes deviations
    from General Relativity. The γ parameter specifically measures how much
    space curvature is produced by unit rest mass.
    
    GR predicts γ = 1 exactly.
    """
    
    def get_observational_data(self) -> Dict[str, Any]:
        """Get Cassini PPN observational data"""
        return {
            'object': 'Solar System',
            'measurement': 'PPN γ parameter',
            'value': 1.0,  # GR prediction
            'uncertainty': 2.3e-5,  # Cassini measurement precision
            'units': 'dimensionless',
            'reference': 'Bertotti et al. (2003), Nature 425, 374',
            'notes': 'γ = (1 + γ_measured)/2 = 1 + (2.1 ± 2.3) × 10^-5'
        }
    
    def validate(self, theory: GravitationalTheory, verbose: bool = False) -> ValidationResult:
        """
        Validate theory's PPN γ parameter.
        
        For a spherically symmetric metric in isotropic coordinates:
        ds² = -A(r)c²dt² + B(r)dr² + r²dΩ²
        
        The PPN γ parameter can be extracted from the weak-field expansion:
        B(r) ≈ 1 + 2γGM/(rc²) + O((GM/rc²)²)
        """
        result = ValidationResult(self.name, theory.name)
        obs_data = self.get_observational_data()
        
        if verbose:
            print(f"\nCalculating PPN γ parameter for {theory.name}...")
        
        # Test at multiple distances to ensure we're in weak field
        test_distances = [1e11, 1e12, 1e13]  # meters (beyond Earth's orbit)
        M_sun = 1.989e30  # kg
        c = 2.998e8       # m/s
        G = 6.674e-11     # m³/kg/s²
        
        gamma_values = []
        
        for r_test in test_distances:
            r = self.engine.tensor(r_test)
            M = self.engine.tensor(M_sun)
            
            # Get metric components
            g_tt, g_rr, g_pp, g_tp = theory.get_metric(r.unsqueeze(0), M, c, G)
            
            # Extract values
            g_rr_val = g_rr.item()
            
            # Weak field approximation: g_rr ≈ 1 + 2γΦ where Φ = GM/(rc²)
            Phi = G * M_sun / (r_test * c**2)
            
            # Solve for γ
            gamma_est = (g_rr_val - 1) / (2 * Phi)
            gamma_values.append(gamma_est)
            
            if verbose:
                print(f"  At r = {r_test:.1e} m: g_rr = {g_rr_val:.10f}, γ ≈ {gamma_est:.6f}")
        
        # Take average of estimates (should be consistent in weak field)
        predicted_gamma = np.mean(gamma_values)
        gamma_std = np.std(gamma_values)
        
        if verbose:
            print(f"  Average γ = {predicted_gamma:.6f} ± {gamma_std:.2e}")
        
        # Store results
        result.observed_value = obs_data['value']
        result.predicted_value = float(predicted_gamma)
        result.units = obs_data['units']
        
        # Compute error
        result.error, result.error_percent = self.compute_error(
            result.observed_value, result.predicted_value
        )
        
        # Check if within observational uncertainty (3-sigma)
        tolerance = 3 * obs_data['uncertainty']  # 3-sigma = 6.9e-5
        
        result.passed = abs(result.predicted_value - result.observed_value) < tolerance
        
        if verbose:
            print(f"\nResults:")
            print(f"  Observed: {result.observed_value:.6f} ± {obs_data['uncertainty']:.2e}")
            print(f"  Predicted: {result.predicted_value:.6f}")
            print(f"  Error: {result.error:.2e} ({result.error_percent:.4f}%)")
            print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
            
        result.notes = f"Tolerance: ±{tolerance:.2e} (3-sigma)"
        
        return result 