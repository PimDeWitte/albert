#!/usr/bin/env python3
"""
Quantum Lagrangian Grounding Validator
<reason>chain: Implements pulsar-like empirical grounding for quantum field Lagrangians</reason>
<reason>chain: Prevents overfitting by tying predictions to real quantum experiments</reason>
<reason>chain: Mirrors structure of PSR J0740+6620 validator but for quantum regime</reason>
"""

import torch
import numpy as np
from typing import Dict, Any, TYPE_CHECKING

from .base_validation import ObservationalValidator, ValidationResult, safe_float_conversion
from physics_agent.utils import get_metric_wrapper

# Import all necessary constants from centralized module
from physics_agent.constants import (
    # Physical constants
    HBAR as hbar, BOLTZMANN_CONSTANT as k, GRAVITATIONAL_CONSTANT as G, 
    SPEED_OF_LIGHT as c, NEUTRON_MASS as m_n, ELECTRON_MASS as m_e, 
    ELEMENTARY_CHARGE as e_charge,
    # Earth parameters
    EARTH_MASS, EARTH_RADIUS,
    # Quantum experiment data
    COW_INTERFEROMETRY, ATOM_INTERFEROMETRY, QUANTUM_CLOCK
)

if TYPE_CHECKING:
    from physics_agent.base_theory import GravitationalTheory


class QuantumLagrangianGroundingValidator(ObservationalValidator):
    """
    <reason>chain: Grounds quantum Lagrangians in empirical data to prevent overfitting</reason>
    <reason>chain: Uses COW, atom interferometry, and quantum clock data as anchors</reason>
    <reason>chain: Ensures theories reduce to GR at zero quantum parameters</reason>
    
    Data sources:
    - COW: Colella, Overhauser, Werner (1975) - neutron interferometry
    - Atom: Müller et al. (2010) - atom interferometry  
    - Clock: Chou et al. (2010) - optical clock measurements
    """
    category = "observational"  # This is an observational validator for quantum theories
    
    def __init__(self, engine=None, tolerance: float = 1e-6):
        """Initialize the validator with a theory engine."""
        super().__init__(engine)
        self.name = "Quantum Lagrangian Grounding Validator"
        self.tolerance = tolerance
        
    def get_observational_data(self) -> Dict[str, Any]:
        """<reason>chain: Consolidated quantum experimental data for grounding</reason>"""
        return {
            'cow': {
                'phase_shift': COW_INTERFEROMETRY['phase_shift'],
                'uncertainty': COW_INTERFEROMETRY['uncertainty'],
                'reference': COW_INTERFEROMETRY['reference']
            },
            'atom': {
                'redshift': ATOM_INTERFEROMETRY['frequency_shift'],
                'uncertainty': ATOM_INTERFEROMETRY['uncertainty'],
                'reference': ATOM_INTERFEROMETRY['reference']
            },
            'clock': {
                'redshift': QUANTUM_CLOCK['frequency_shift'],
                'uncertainty': QUANTUM_CLOCK['uncertainty'],
                'reference': QUANTUM_CLOCK['reference']
            }
        }
    
    def compute_quantum_lagrangian_prediction(self, theory: "GravitationalTheory", 
                                            experiment: str) -> float:
        """
        <reason>chain: Compute prediction from quantum Lagrangian for specific experiment</reason>
        <reason>chain: Integrates 6D geodesic with quantum corrections from field terms</reason>
        """
        # Extract quantum parameters from Lagrangian components
        quantum_params = self.extract_quantum_parameters(theory)
        
        if experiment == 'cow':
            return self.compute_cow_phase_shift(theory, quantum_params)
        elif experiment == 'atom':
            return self.compute_atom_redshift(theory, quantum_params)
        elif experiment == 'clock':
            return self.compute_clock_redshift(theory, quantum_params)
        else:
            raise ValueError(f"Unknown experiment: {experiment}")
    
    def extract_quantum_parameters(self, theory: "GravitationalTheory") -> Dict[str, Any]:
        """<reason>chain: Extract quantum coupling constants from Lagrangian terms</reason>"""
        params = {}
        
        # Check for matter Lagrangian coupling
        if hasattr(theory, 'matter_lagrangian') and theory.matter_lagrangian:
            # Look for fermion mass term coefficient
            if hasattr(theory, 'm_f'):
                params['fermion_mass'] = safe_float_conversion(theory.m_f, m_e)
            else:
                params['fermion_mass'] = m_e  # Default to electron mass
        
        # Check for gauge coupling
        if hasattr(theory, 'gauge_lagrangian') and theory.gauge_lagrangian:
            if hasattr(theory, 'q'):
                params['charge'] = safe_float_conversion(theory.q, e_charge)
            elif hasattr(theory, 'e'):
                params['charge'] = safe_float_conversion(theory.e, e_charge)
            else:
                params['charge'] = e_charge  # Elementary charge
        
        # Check for quantum corrections
        if hasattr(theory, 'alpha'):
            params['alpha'] = safe_float_conversion(theory.alpha, 0.0)
        if hasattr(theory, 'kappa'):
            params['kappa'] = safe_float_conversion(theory.kappa, 0.0)
        if hasattr(theory, 'epsilon'):
            params['epsilon'] = safe_float_conversion(theory.epsilon, 0.0)
            
        return params
    
    def compute_cow_phase_shift(self, theory: "GravitationalTheory", 
                               quantum_params: Dict[str, Any]) -> float:
        """
        <reason>chain: Compute COW neutron interferometry phase shift</reason>
        <reason>chain: Δφ = (2π m g A sin α) / (h v) with quantum corrections</reason>
        """
        # Experimental parameters from COW constants
        A = COW_INTERFEROMETRY['area']
        v = COW_INTERFEROMETRY['neutron_velocity']
        sin_alpha = COW_INTERFEROMETRY['sin_alpha']
        
        # Get metric at Earth's surface
        r = torch.tensor(EARTH_RADIUS, device=self.engine.device, dtype=self.engine.dtype)
        M = torch.tensor(EARTH_MASS, device=self.engine.device, dtype=self.engine.dtype)
        
        metric_func = get_metric_wrapper(theory.get_metric)
        g_tt, g_rr, g_pp, g_tp = metric_func(r=r.unsqueeze(0), M=M, c=c, G=G)
        
        # Classical gravitational acceleration
        g_classical = G * EARTH_MASS / EARTH_RADIUS**2
        
        # Quantum corrections from metric
        rs = 2 * G * EARTH_MASS / c**2
        # For weak field at Earth's surface, g_tt ≈ -(1 - rs/r)
        # The correction factor should be small
        metric_correction = abs(g_tt.item() + 1.0) / (rs/EARTH_RADIUS)
        
        # Apply quantum parameter corrections
        quantum_correction = 1.0
        if 'alpha' in quantum_params and quantum_params['alpha'] != 0:
            alpha_val = safe_float_conversion(quantum_params['alpha'], 0.0)
            quantum_correction *= (1 + alpha_val)
        if 'kappa' in quantum_params and quantum_params['kappa'] != 0:
            kappa_val = safe_float_conversion(quantum_params['kappa'], 0.0)
            quantum_correction *= (1 + kappa_val * hbar / (m_n * c * EARTH_RADIUS))
        
        # Total phase shift using correct COW formula
        # Δφ = (2π m g A sin α) / (h v) where h = 2π ħ
        g_eff = g_classical * metric_correction * quantum_correction
        phase_shift = (m_n * g_eff * A * sin_alpha) / (hbar * v)
        
        return float(phase_shift)
    
    def compute_atom_redshift(self, theory: "GravitationalTheory",
                             quantum_params: Dict[str, Any]) -> float:
        """<reason>chain: Compute atom interferometry gravitational redshift</reason>"""
        # Height difference
        h = 1.0  # Per meter
        
        # Get metric and compute redshift
        r = torch.tensor(EARTH_RADIUS, device=self.engine.device, dtype=self.engine.dtype, requires_grad=True)
        M = torch.tensor(EARTH_MASS, device=self.engine.device, dtype=self.engine.dtype)
        
        metric_func = get_metric_wrapper(theory.get_metric)
        g_tt, _, _, _ = metric_func(r=r.unsqueeze(0), M=M, c=c, G=G)
        
        try:
            dg_tt_dr = torch.autograd.grad(g_tt.sum(), r, create_graph=True)[0]
        except RuntimeError:
            # Finite difference fallback
            dr = 1e-6 * r
            r_plus = r + dr
            g_tt_plus, _, _, _ = metric_func(r=r_plus.unsqueeze(0), M=M, c=c, G=G)
            dg_tt_dr = (g_tt_plus - g_tt) / dr
        
        freq_shift = - (dg_tt_dr * h) / (2 * (-g_tt)).item()
        
        # Apply quantum corrections
        if 'epsilon' in quantum_params:
            # Planck-scale corrections
            l_planck = np.sqrt(hbar * G / c**3)
            epsilon_val = safe_float_conversion(quantum_params['epsilon'], 0.0)
            freq_shift *= (1 + epsilon_val * l_planck / h)
        
        return float(freq_shift)
    
    def compute_clock_redshift(self, theory: "GravitationalTheory",
                              quantum_params: Dict[str, Any]) -> float:
        """<reason>chain: Compute quantum clock redshift for 33cm height</reason>"""
        # Use atom redshift calculation with specific height
        freq_shift_per_meter = self.compute_atom_redshift(theory, quantum_params)
        return freq_shift_per_meter * QUANTUM_CLOCK['height']  # Use height from constants
    
    def validate(self, theory: "GravitationalTheory", verbose: bool = False) -> ValidationResult:
        """
        <reason>chain: Main validation - ensures theory matches all quantum experiments</reason>
        <reason>chain: At zero quantum parameters, must reduce to GR exactly</reason>
        """
        result = ValidationResult(self.name, theory.name)
        obs_data = self.get_observational_data()
        
        if verbose:
            print(f"\n{self.name} for {theory.name}")
            print("=" * 60)
        
        # Test each experiment
        total_chi2 = 0.0
        experiment_results = {}
        
        for exp_name, exp_data in obs_data.items():
            try:
                predicted = self.compute_quantum_lagrangian_prediction(theory, exp_name)
                
                # Get the observed value with correct key
                if exp_name == 'cow':
                    observed = exp_data['phase_shift']
                elif exp_name == 'atom' or exp_name == 'clock':
                    observed = exp_data['redshift']
                else:
                    raise ValueError(f"Unknown experiment: {exp_name}")
                    
                uncertainty = exp_data['uncertainty']
                
                # Compute chi-squared contribution
                chi2 = ((predicted - observed) / uncertainty) ** 2
                total_chi2 += chi2
                
                experiment_results[exp_name] = {
                    'predicted': predicted,
                    'observed': observed,
                    'chi2': chi2,
                    'sigma': abs(predicted - observed) / uncertainty
                }
                
                if verbose:
                    print(f"\n{exp_name.upper()} Experiment:")
                    print(f"  Observed: {observed:.3e} ± {uncertainty:.3e}")
                    print(f"  Predicted: {predicted:.3e}")
                    print(f"  Deviation: {experiment_results[exp_name]['sigma']:.1f}σ")
                    
            except Exception as e:
                if verbose:
                    print(f"\n{exp_name.upper()} Experiment: ERROR - {str(e)}")
                experiment_results[exp_name] = {'error': str(e)}
                # <reason>chain: Use smaller penalty and treat as warning rather than failure</reason>
                total_chi2 += 10.0  # Smaller penalty for calculation errors
        
        # Check GR limit (set all quantum params to zero)
        if hasattr(theory, 'category') and theory.category in ['quantum', 'quantum']:
            gr_limit_chi2 = self.check_gr_limit(theory, verbose)
            total_chi2 += gr_limit_chi2
        
        # Overall assessment
        n_experiments = len(obs_data)
        reduced_chi2 = total_chi2 / n_experiments
        
        # <reason>chain: Properly handle validation based on chi-squared value and errors</reason>
        # Pass if reduced chi-squared < 10 (reasonable for quantum theories)
        # But FAIL if there were errors in any experiment
        has_errors = any('error' in exp_result for exp_result in experiment_results.values())
        
        if has_errors:
            result.passed = False  # Fail if any experiment had errors
        else:
            result.passed = reduced_chi2 < 10.0  # Pass if chi-squared is reasonable
        
        result.observed_value = 1.0  # Target reduced chi-squared
        result.predicted_value = min(reduced_chi2, 100.0)  # Cap the displayed value
        result.error = abs(min(reduced_chi2, 100.0) - 1.0)
        result.error_percent = 100.0 * result.error
        result.units = "reduced χ²"
        
        if has_errors:
            result.notes = f"[ERROR] Some experiments failed to compute. Total χ² = {min(total_chi2, 1000.0):.2f} over {n_experiments} experiments"
        else:
            result.notes = f"Total χ² = {min(total_chi2, 1000.0):.2f} over {n_experiments} experiments"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Total χ² = {total_chi2:.2f}")
            print(f"Reduced χ² = {reduced_chi2:.2f}")
            print(f"Result: {'PASS' if result.passed else 'FAIL'}")
        
        # Store detailed results
        result.details = experiment_results
        
        return result
    
    def check_gr_limit(self, theory: "GravitationalTheory", verbose: bool = False) -> float:
        """
        <reason>chain: Verify theory reduces to GR when quantum parameters → 0</reason>
        <reason>chain: Essential check to prevent overfitting to quantum regime</reason>
        """
        # Create a GR-equivalent version by zeroing quantum parameters
        original_params = {}
        quantum_param_names = ['alpha', 'kappa', 'epsilon', 'sigma', 'eta', 'gamma', 
                              'omega', 'lambda_c', 'beta', 'tau', 'delta']
        
        # Store and zero quantum parameters
        for param in quantum_param_names:
            if hasattr(theory, param):
                original_params[param] = getattr(theory, param)
                setattr(theory, param, 0.0)
        
        # Test COW experiment (most sensitive)
        try:
            quantum_params = self.extract_quantum_parameters(theory)
            gr_prediction = self.compute_cow_phase_shift(theory, quantum_params)
            
            # Expected GR value
            gr_expected = 2.70  # From COW data (corrected)
            
            # <reason>chain: Use appropriate error tolerance for GR limit</reason>
            # Allow 1% deviation from GR when quantum parameters are zero
            tolerance = 0.027  # 1% of 2.70
            
            chi2_gr = ((gr_prediction - gr_expected) / tolerance) ** 2
            
            if verbose:
                print(f"\nGR Limit Check:")
                print(f"  Expected (GR): {gr_expected:.3f}")
                print(f"  Theory at κ→0: {gr_prediction:.3f}")
                print(f"  χ²: {chi2_gr:.2f}")
                
        except Exception as e:
            if verbose:
                print(f"\nGR Limit Check: ERROR - {str(e)}")
            # <reason>chain: Return large penalty for GR limit check errors</reason>
            chi2_gr = 100.0  # Large penalty for errors
        
        # Restore original parameters
        for param, value in original_params.items():
            setattr(theory, param, value)
        
        return chi2_gr 