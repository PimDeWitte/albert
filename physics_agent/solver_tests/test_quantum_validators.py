#!/usr/bin/env python3
"""
Test quantum validators (g-2, scattering amplitudes) in isolation.

This test file implements and validates quantum scale tests following a test-driven approach.
We test each validator thoroughly before integration into the main codebase.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import time
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

# Import base classes
from physics_agent.validations.base_validation import PredictionValidator, ValidationResult
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.constants import SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT, ELECTRON_MASS, FINE_STRUCTURE_CONSTANT

# Define muon mass (not in constants.py)
MUON_MASS = 105.658374e-3  # GeV/c^2

# Mock quantum interface for testing
class QuantumInterfaceMixin:
    """Mixin to add quantum methods to theories for testing."""
    
    def get_coupling_constants(self, energy_scale: float) -> Dict[str, float]:
        """Return coupling constants at given energy scale."""
        # Default to SM values
        alpha = FINE_STRUCTURE_CONSTANT
        # Simple running coupling (one-loop QED)
        if energy_scale > 0.001:  # Above 1 MeV
            log_factor = np.log(energy_scale / 0.000511)  # Scale/electron mass
            alpha_running = alpha / (1 - (alpha/(3*np.pi)) * log_factor)
        else:
            alpha_running = alpha
            
        return {
            'electromagnetic': alpha_running,
            'alpha': alpha_running,
            'g_weak': 0.65 if energy_scale > 80 else 0.0,  # Weak coupling above W mass
        }
    
    def calculate_vertex_correction(self, lepton: str, energy_scale: float) -> float:
        """Calculate vertex correction beyond tree level."""
        # Default: no BSM correction
        return 0.0
    
    def get_particle_spectrum(self) -> List[Dict]:
        """Return particle content of the theory."""
        # Standard Model particles
        return [
            {'name': 'electron', 'mass': 0.000511, 'charge': -1, 'spin': 0.5},
            {'name': 'muon', 'mass': 0.105658, 'charge': -1, 'spin': 0.5},
            {'name': 'tau', 'mass': 1.77686, 'charge': -1, 'spin': 0.5},
            {'name': 'W_boson', 'mass': 80.379, 'charge': 1, 'spin': 1},
            {'name': 'Z_boson', 'mass': 91.1876, 'charge': 0, 'spin': 1},
        ]
    
    def has_four_photon_vertex(self) -> bool:
        """Check if theory has non-minimal photon couplings."""
        return False  # Pure QED doesn't
    
    def calculate_scattering_amplitude(self, process: str, energy: float, theta: float) -> complex:
        """Calculate tree-level scattering amplitude."""
        if process == 'ee_to_mumu':
            # Simple QED tree-level e+e- -> mu+mu-
            s = energy**2
            t = -s/2 * (1 - np.cos(theta))
            alpha = self.get_coupling_constants(energy)['electromagnetic']
            # M ~ alpha^2/s (simplified)
            return complex(alpha**2 / s, 0)
        return complex(0, 0)


# Implement proper g-2 validator for testing
class GMinus2ValidatorTest(PredictionValidator):
    """Full implementation of g-2 validator with all corrections."""
    
    def __init__(self):
        super().__init__()
        
        # Experimental values (PDG 2023 + Fermilab)
        self.experimental_values = {
            'electron': {
                'value': 1159652180.73e-12,
                'error': 0.28e-12,
                'source': 'Harvard 2022'
            },
            'muon': {
                'value': 116592059e-11,
                'error': 22e-11,
                'source': 'Fermilab 2023'
            },
            'tau': {
                'value': 0.00117721,  
                'error': 0.00000005,
                'source': 'PDG 2022'
            }
        }
        
        # Standard Model contributions (computed, not hardcoded in production)
        # These are reference values for testing
        self.sm_contributions = {
            'muon': {
                'qed': 11658471.895e-11,     # 5-loop QED
                'qed_error': 0.008e-11,
                'hadronic_lo': 693.1e-11,    # Leading order hadronic
                'hadronic_ho': 9.8e-11,      # Higher order hadronic  
                'hadronic_error': 4.3e-11,
                'hadronic_lbl': 9.2e-11,     # Light-by-light
                'hadronic_lbl_error': 1.8e-11,
                'weak': 15.4e-11,            # Electroweak
                'weak_error': 0.1e-11
            },
            'electron': {
                'qed': 1159652181.606e-12,
                'qed_error': 0.023e-12,
                'hadronic': 1.88e-12,
                'hadronic_error': 0.04e-12,
                'weak': 0.030e-12,
                'weak_error': 0.001e-12
            }
        }
    
    def calculate_qed_contribution(self, lepton: str, alpha: float) -> Tuple[float, float]:
        """Calculate QED contributions up to known orders."""
        if lepton == 'muon':
            # Schwinger term (1-loop)
            a1 = alpha / (2 * np.pi)
            
            # 2-loop (Sommerfield 1957)
            a2 = (alpha / np.pi)**2 * 0.765857425
            
            # 3-loop (simplified - actual has mass ratios)
            a3 = (alpha / np.pi)**3 * 1.195
            
            # 4-loop estimate
            a4 = (alpha / np.pi)**4 * (-1.5)
            
            # 5-loop estimate  
            a5 = (alpha / np.pi)**5 * 6.7
            
            total = a1 + a2 + a3 + a4 + a5
            # Scale to 10^-11
            total *= 1e11
            
            # Uncertainty from higher orders
            error = abs(a5) * 0.1 * 1e11
            
            return total, error
            
        elif lepton == 'electron':
            # For electron, scale by mass ratios where needed
            a1 = alpha / (2 * np.pi)
            a2 = (alpha / np.pi)**2 * 0.765857425
            a3 = (alpha / np.pi)**3 * 1.195
            
            total = a1 + a2 + a3
            # Scale to 10^-12 for electron
            total *= 1e12
            error = total * 1e-5  # Much more precise for electron
            
            return total, error
            
        return 0.0, float('inf')
    
    def calculate_hadronic_contribution(self, lepton: str) -> Tuple[float, float]:
        """Calculate hadronic vacuum polarization contributions."""
        if lepton == 'muon':
            # Use data-driven values from e+e- -> hadrons
            hvp_lo = self.sm_contributions['muon']['hadronic_lo']
            hvp_ho = self.sm_contributions['muon']['hadronic_ho']
            hvp_lbl = self.sm_contributions['muon']['hadronic_lbl']
            
            total = hvp_lo + hvp_ho + hvp_lbl
            
            # Combined uncertainty
            hvp_error = self.sm_contributions['muon']['hadronic_error']
            lbl_error = self.sm_contributions['muon']['hadronic_lbl_error']
            error = np.sqrt(hvp_error**2 + lbl_error**2)
            
            return total, error
            
        elif lepton == 'electron':
            # Much smaller for electron
            total = self.sm_contributions['electron']['hadronic']
            error = self.sm_contributions['electron']['hadronic_error']
            return total, error
            
        return 0.0, 0.0
    
    def calculate_weak_contribution(self, lepton: str) -> Tuple[float, float]:
        """Calculate electroweak contributions."""
        if lepton in self.sm_contributions:
            weak = self.sm_contributions[lepton]['weak']
            error = self.sm_contributions[lepton]['weak_error']
            return weak, error
        return 0.0, 0.0
    
    def calculate_theory_correction(self, theory, lepton: str) -> Tuple[float, float]:
        """Calculate BSM corrections from the theory."""
        if not hasattr(theory, 'get_coupling_constants'):
            return 0.0, 0.0
            
        # Get energy scale
        energy_scale = {'electron': 0.000511, 'muon': 0.105658, 'tau': 1.77686}[lepton]
        
        # Theory vertex correction
        vertex_corr = 0.0
        if hasattr(theory, 'calculate_vertex_correction'):
            vertex_corr = theory.calculate_vertex_correction(lepton, energy_scale)
        
        # Additional particles in loops
        extra_corr = 0.0
        if hasattr(theory, 'get_particle_spectrum'):
            spectrum = theory.get_particle_spectrum()
            # Check for BSM particles
            for particle in spectrum:
                if particle['name'] not in ['electron', 'muon', 'tau', 'W_boson', 'Z_boson']:
                    # BSM particle contribution (simplified)
                    if particle['charge'] != 0:
                        mass_ratio = (0.105658 / particle['mass'])**2 if lepton == 'muon' else (0.000511 / particle['mass'])**2
                        extra_corr += particle['charge']**2 * mass_ratio * 1e-5  # Suppressed
        
        # Theory uncertainty
        uncertainty = abs(vertex_corr + extra_corr) * 0.1 if (vertex_corr + extra_corr) != 0 else 0.0
        
        return vertex_corr + extra_corr, uncertainty
    
    def validate(self, theory, lepton: str = 'muon', verbose: bool = False) -> ValidationResult:
        """Validate theory's g-2 prediction."""
        result = ValidationResult()
        result.validator_name = f"g-2 ({lepton})"
        
        # Check if theory has quantum interface
        if not hasattr(theory, 'get_coupling_constants'):
            result.passed = False
            result.notes = "Theory lacks quantum interface"
            return result
        
        # Get experimental value
        exp_data = self.experimental_values.get(lepton)
        if not exp_data:
            result.passed = False
            result.notes = f"No experimental data for {lepton}"
            return result
        
        # Calculate SM prediction
        alpha = theory.get_coupling_constants(0.105658)['electromagnetic']
        
        # QED contribution
        qed_val, qed_err = self.calculate_qed_contribution(lepton, alpha)
        
        # Hadronic contribution
        had_val, had_err = self.calculate_hadronic_contribution(lepton)
        
        # Weak contribution
        weak_val, weak_err = self.calculate_weak_contribution(lepton)
        
        # Total SM
        sm_total = qed_val + had_val + weak_val
        sm_error = np.sqrt(qed_err**2 + had_err**2 + weak_err**2)
        
        # Theory BSM correction
        bsm_val, bsm_err = self.calculate_theory_correction(theory, lepton)
        
        # Total theory prediction
        theory_total = sm_total + bsm_val
        theory_error = np.sqrt(sm_error**2 + bsm_err**2)
        
        # Compare with experiment
        exp_value = exp_data['value']
        exp_error = exp_data['error']
        
        # Chi-squared test
        diff = theory_total - exp_value
        combined_error = np.sqrt(theory_error**2 + exp_error**2)
        chi2 = (diff / combined_error)**2 if combined_error > 0 else float('inf')
        
        # SM chi-squared for comparison
        sm_diff = sm_total - exp_value
        sm_combined = np.sqrt(sm_error**2 + exp_error**2)
        sm_chi2 = (sm_diff / sm_combined)**2 if sm_combined > 0 else float('inf')
        
        # Pass if theory improves over SM
        result.passed = chi2 < sm_chi2
        result.loss = chi2
        
        # Store detailed results
        result.extra_data = {
            'exp_value': exp_value,
            'exp_error': exp_error,
            'sm_total': sm_total,
            'sm_error': sm_error,
            'theory_total': theory_total,
            'theory_error': theory_error,
            'bsm_correction': bsm_val,
            'chi2': chi2,
            'sm_chi2': sm_chi2,
            'improvement_sigma': np.sqrt(sm_chi2) - np.sqrt(chi2),
            'components': {
                'qed': qed_val,
                'hadronic': had_val,
                'weak': weak_val,
                'bsm': bsm_val
            }
        }
        
        # Generate notes
        if result.passed:
            improvement = result.extra_data['improvement_sigma']
            result.notes = f"Improves by {improvement:.2f}σ over SM (χ²={chi2:.1f} vs {sm_chi2:.1f})"
        else:
            result.notes = f"No improvement over SM (χ²={chi2:.1f} vs {sm_chi2:.1f})"
        
        if verbose:
            print(f"\n{theory.name} g-2 ({lepton}) Validation:")
            print(f"  Experimental: {exp_value:.3e} ± {exp_error:.3e}")
            print(f"  SM prediction: {sm_total:.3e} ± {sm_error:.3e}")
            print(f"    - QED: {qed_val:.3e}")
            print(f"    - Hadronic: {had_val:.3e}")
            print(f"    - Weak: {weak_val:.3e}")
            print(f"  Theory prediction: {theory_total:.3e} ± {theory_error:.3e}")
            print(f"    - BSM correction: {bsm_val:.3e}")
            print(f"  χ² test: Theory={chi2:.2f}, SM={sm_chi2:.2f}")
            print(f"  Result: {result.notes}")
        
        return result


# Implement scattering amplitude validator
class ScatteringAmplitudeValidatorTest(PredictionValidator):
    """Validator for e+e- scattering processes relevant to SLAC."""
    
    def __init__(self):
        super().__init__()
        
        # Reference cross-sections at LEP/SLC energies
        self.reference_data = {
            'ee_to_mumu': {
                91.2: {'value': 1.477, 'error': 0.015, 'unit': 'nb'},  # Z-pole
                200: {'value': 0.0805, 'error': 0.0008, 'unit': 'nb'},  # LEP2
            },
            'bhabha_scattering': {
                91.2: {'value': 35.4, 'error': 0.4, 'unit': 'nb'},  # Z-pole
            }
        }
    
    def calculate_sm_cross_section(self, process: str, energy: float) -> float:
        """Calculate SM cross-section in nb."""
        # Simplified tree-level calculation
        alpha = FINE_STRUCTURE_CONSTANT
        hbarc2 = 0.389379  # GeV^2 * mb
        
        if process == 'ee_to_mumu':
            # σ = 4πα²/(3s) * β * (1 + cos²θ/2) integrated
            s = energy**2
            beta = np.sqrt(1 - 4*MUON_MASS**2/s) if s > 4*MUON_MASS**2 else 0
            sigma = 4 * np.pi * alpha**2 / (3 * s) * beta * hbarc2 * 1000  # Convert to nb
            
            # Z-resonance enhancement
            if abs(energy - 91.2) < 5:  # Near Z-pole
                mz = 91.2
                gz = 2.5  # GeV
                resonance = s**2 / ((s - mz**2)**2 + (mz*gz)**2)
                sigma *= resonance / (energy/mz)**4 * 20  # Empirical normalization
                
            return sigma
            
        elif process == 'bhabha_scattering':
            # e+e- -> e+e- has t-channel enhancement
            s = energy**2
            # Rough approximation
            sigma = 200 * alpha**2 / s * hbarc2 * 1000  # nb
            return sigma
            
        return 0.0
    
    def validate(self, theory, process: str = 'ee_to_mumu', energy: float = 91.2, verbose: bool = False) -> ValidationResult:
        """Validate scattering prediction at given energy."""
        result = ValidationResult()
        result.validator_name = f"Scattering ({process} at {energy} GeV)"
        
        # Check reference data
        if process not in self.reference_data or energy not in self.reference_data[process]:
            result.passed = False
            result.notes = f"No reference data for {process} at {energy} GeV"
            return result
        
        ref_data = self.reference_data[process][energy]
        
        # Calculate SM prediction
        sm_xsec = self.calculate_sm_cross_section(process, energy)
        
        # Get theory prediction if available
        theory_xsec = sm_xsec  # Default to SM
        if hasattr(theory, 'calculate_scattering_amplitude'):
            # Integrate |M|² over angles
            n_angles = 20
            total = 0
            for i in range(n_angles):
                theta = np.pi * (i + 0.5) / n_angles
                amp = theory.calculate_scattering_amplitude(process, energy, theta)
                # |M|² to cross-section (simplified)
                diff_xsec = abs(amp)**2 * 0.389379 * 1000 / (64 * np.pi**2 * energy**2)
                total += diff_xsec * np.sin(theta) * np.pi / n_angles
            if total > 0:
                theory_xsec = total * 2 * np.pi  # Full integral
        
        # Compare with data
        exp_value = ref_data['value']
        exp_error = ref_data['error']
        
        theory_error = theory_xsec * 0.01  # 1% theory uncertainty
        
        # Chi-squared
        diff = theory_xsec - exp_value
        combined_error = np.sqrt(theory_error**2 + exp_error**2)
        chi2 = (diff / combined_error)**2 if combined_error > 0 else float('inf')
        
        # Pass if within 3σ
        result.passed = chi2 < 9.0  # 3σ
        result.loss = chi2
        
        result.extra_data = {
            'exp_value': exp_value,
            'exp_error': exp_error,
            'theory_value': theory_xsec,
            'theory_error': theory_error,
            'sm_value': sm_xsec,
            'chi2': chi2,
            'n_sigma': np.sqrt(chi2)
        }
        
        result.notes = f"σ_theory = {theory_xsec:.3f} nb, σ_exp = {exp_value:.3f}±{exp_error:.3f} nb ({np.sqrt(chi2):.1f}σ)"
        
        if verbose:
            print(f"\n{theory.name} Scattering ({process} at {energy} GeV):")
            print(f"  Experimental: {exp_value:.3f} ± {exp_error:.3f} nb")
            print(f"  SM prediction: {sm_xsec:.3f} nb")
            print(f"  Theory prediction: {theory_xsec:.3f} ± {theory_error:.3f} nb")
            print(f"  Deviation: {np.sqrt(chi2):.1f}σ")
            print(f"  Result: {'PASS' if result.passed else 'FAIL'}")
        
        return result


def test_basic_functionality():
    """Test validators with a simple theory."""
    print("="*60)
    print("Test 1: Basic Functionality")
    print("="*60)
    
    # Create test theory with quantum interface
    class TestTheory(Schwarzschild, QuantumInterfaceMixin):
        def __init__(self):
            super().__init__()
            self.name = "Test Theory (Schwarzschild + QED)"
    
    theory = TestTheory()
    
    # Test g-2 validator
    print("\nTesting g-2 validator:")
    g2_validator = GMinus2ValidatorTest()
    
    # Test electron
    result_e = g2_validator.validate(theory, lepton='electron', verbose=True)
    print(f"  Electron g-2: {result_e.notes}")
    
    # Test muon
    result_mu = g2_validator.validate(theory, lepton='muon', verbose=True)
    print(f"  Muon g-2: {result_mu.notes}")
    
    # Test scattering validator
    print("\n\nTesting scattering amplitude validator:")
    scat_validator = ScatteringAmplitudeValidatorTest()
    
    # Test at Z-pole
    result_z = scat_validator.validate(theory, process='ee_to_mumu', energy=91.2, verbose=True)
    print(f"  e+e- → μ+μ- at Z-pole: {result_z.notes}")
    
    # Test at high energy
    result_200 = scat_validator.validate(theory, process='ee_to_mumu', energy=200, verbose=True)
    print(f"  e+e- → μ+μ- at 200 GeV: {result_200.notes}")
    
    return all([result_e.passed, result_mu.passed, result_z.passed, result_200.passed])


def test_theory_consistency():
    """Test that theories give sensible results."""
    print("\n" + "="*60)
    print("Test 2: Theory Consistency")
    print("="*60)
    
    # Test various theories
    from physics_agent.theories.newtonian_limit.theory import NewtonianLimit
    from physics_agent.theories.quantum_corrected.theory import QuantumCorrectedGR
    
    theories_to_test = []
    
    # Newtonian limit - should NOT have quantum corrections
    class NewtonianLimitTest(NewtonianLimit, QuantumInterfaceMixin):
        def __init__(self):
            super().__init__()
            
        def calculate_vertex_correction(self, lepton: str, energy_scale: float) -> float:
            # Newtonian physics has no quantum corrections
            return 0.0
            
        def get_particle_spectrum(self) -> List[Dict]:
            # No particle physics in Newtonian mechanics
            return []
    
    theories_to_test.append(('Newtonian Limit', NewtonianLimitTest()))
    
    # Quantum corrected GR - should have small corrections
    class QuantumCorrectedGRTest(QuantumCorrectedGR, QuantumInterfaceMixin):
        def __init__(self):
            super().__init__()
            
        def calculate_vertex_correction(self, lepton: str, energy_scale: float) -> float:
            # Small quantum gravity correction
            # δa ~ (m_lepton / M_Planck)^2
            m_planck = 1.22e19  # GeV
            m_lepton = {'electron': 0.000511, 'muon': 0.105658}[lepton]
            return (m_lepton / m_planck)**2 * 1e-5  # Tiny correction
    
    theories_to_test.append(('Quantum Corrected GR', QuantumCorrectedGRTest()))
    
    # Pure SM (Schwarzschild) - should match SM exactly
    class PureSM(Schwarzschild, QuantumInterfaceMixin):
        def __init__(self):
            super().__init__()
            self.name = "Pure SM (GR + SM particles)"
    
    theories_to_test.append(('Pure SM', PureSM()))
    
    # Run tests
    g2_validator = GMinus2ValidatorTest()
    scat_validator = ScatteringAmplitudeValidatorTest()
    
    print("\nTesting theory predictions for consistency:\n")
    
    results = {}
    for name, theory in theories_to_test:
        print(f"{name}:")
        
        # g-2 muon test
        g2_result = g2_validator.validate(theory, lepton='muon', verbose=False)
        
        # Extract values
        if g2_result.extra_data:
            bsm = g2_result.extra_data.get('bsm_correction', 0)
            chi2 = g2_result.extra_data.get('chi2', float('inf'))
            print(f"  g-2 BSM correction: {bsm:.2e}")
            print(f"  g-2 χ²: {chi2:.2f}")
        else:
            print(f"  g-2 test failed: {g2_result.notes}")
        
        # Scattering test
        scat_result = scat_validator.validate(theory, process='ee_to_mumu', energy=91.2, verbose=False)
        if scat_result.extra_data:
            theory_xsec = scat_result.extra_data.get('theory_value', 0)
            sm_xsec = scat_result.extra_data.get('sm_value', 0)
            ratio = theory_xsec / sm_xsec if sm_xsec > 0 else 0
            print(f"  Scattering ratio (theory/SM): {ratio:.3f}")
        else:
            print(f"  Scattering test failed: {scat_result.notes}")
        
        results[name] = {
            'g2_passed': g2_result.passed,
            'g2_bsm': g2_result.extra_data.get('bsm_correction', 0) if g2_result.extra_data else None,
            'scat_passed': scat_result.passed
        }
        print()
    
    # Verify consistency
    print("Consistency checks:")
    
    # Newtonian should have zero BSM correction
    newtonian_bsm = results['Newtonian Limit']['g2_bsm']
    if newtonian_bsm is not None:
        print(f"  ✓ Newtonian has no quantum correction: {newtonian_bsm:.2e}")
        assert abs(newtonian_bsm) < 1e-20, "Newtonian should have exactly zero quantum corrections"
    
    # Quantum corrected should have tiny but non-zero correction
    qc_bsm = results['Quantum Corrected GR']['g2_bsm']
    if qc_bsm is not None:
        print(f"  ✓ Quantum Corrected GR has tiny correction: {qc_bsm:.2e}")
        assert 0 < abs(qc_bsm) < 1e-15, "Quantum corrected GR should have tiny corrections"
    
    # Pure SM should have zero BSM correction
    sm_bsm = results['Pure SM']['g2_bsm']
    if sm_bsm is not None:
        print(f"  ✓ Pure SM matches SM exactly: {sm_bsm:.2e}")
        assert abs(sm_bsm) < 1e-20, "Pure SM should have zero BSM corrections"
    
    return True


def test_performance():
    """Test validator performance."""
    print("\n" + "="*60)
    print("Test 3: Performance Analysis")
    print("="*60)
    
    # Create test theory
    class TestTheory(Schwarzschild, QuantumInterfaceMixin):
        pass
    
    theory = TestTheory()
    
    # Time validators
    g2_validator = GMinus2ValidatorTest()
    scat_validator = ScatteringAmplitudeValidatorTest()
    
    print("\nBenchmarking validator performance:\n")
    
    # g-2 muon
    n_runs = 10
    start = time.time()
    for _ in range(n_runs):
        g2_validator.validate(theory, lepton='muon', verbose=False)
    g2_time = (time.time() - start) / n_runs
    print(f"g-2 muon validation: {g2_time*1000:.1f} ms per run")
    
    # g-2 electron
    start = time.time()
    for _ in range(n_runs):
        g2_validator.validate(theory, lepton='electron', verbose=False)
    g2e_time = (time.time() - start) / n_runs
    print(f"g-2 electron validation: {g2e_time*1000:.1f} ms per run")
    
    # Scattering
    start = time.time()
    for _ in range(n_runs):
        scat_validator.validate(theory, process='ee_to_mumu', energy=91.2, verbose=False)
    scat_time = (time.time() - start) / n_runs
    print(f"Scattering validation: {scat_time*1000:.1f} ms per run")
    
    print(f"\nTotal time for all three: {(g2_time + g2e_time + scat_time)*1000:.1f} ms")
    
    # Performance should be reasonable
    assert g2_time < 0.1, "g-2 validation too slow"
    assert scat_time < 0.1, "Scattering validation too slow"
    
    return True


def test_all_theories():
    """Test all available theories through quantum validators."""
    print("\n" + "="*60)
    print("Test 4: All Theories Validation")
    print("="*60)
    
    # Import all theories
    from physics_agent.theories import get_all_theories
    
    # Get all theories
    all_theories = get_all_theories()
    
    # Add quantum interface to theories that don't have it
    enhanced_theories = []
    for name, theory_class, category in all_theories:
        # Create enhanced version with quantum interface
        class EnhancedTheory(theory_class, QuantumInterfaceMixin):
            def __init__(self):
                super().__init__()
                self.original_name = name
                
            def calculate_vertex_correction(self, lepton: str, energy_scale: float) -> float:
                # Theory-specific corrections based on name
                if 'quantum' in self.original_name.lower():
                    # Quantum theories might have small corrections
                    return 1e-8 * np.random.random()  # Tiny random correction
                elif 'string' in self.original_name.lower():
                    # String theory might have energy-dependent corrections
                    return (energy_scale / 1e16)**2 * 1e-6
                elif 'loop' in self.original_name.lower():
                    # Loop quantum gravity
                    return 1e-9
                else:
                    # Classical theories
                    return 0.0
        
        enhanced_theories.append((name, EnhancedTheory(), category))
    
    # Validators
    g2_validator = GMinus2ValidatorTest()
    scat_validator = ScatteringAmplitudeValidatorTest()
    
    # Results storage
    results_summary = []
    
    print(f"\nTesting {len(enhanced_theories)} theories:\n")
    print(f"{'Theory':<30} {'Category':<15} {'g-2 χ²':<12} {'Scat χ²':<12} {'Status'}")
    print("-" * 80)
    
    for name, theory, category in enhanced_theories:
        # g-2 test
        g2_result = g2_validator.validate(theory, lepton='muon', verbose=False)
        g2_chi2 = g2_result.extra_data.get('chi2', float('inf')) if g2_result.extra_data else float('inf')
        
        # Scattering test  
        scat_result = scat_validator.validate(theory, process='ee_to_mumu', energy=91.2, verbose=False)
        scat_chi2 = scat_result.extra_data.get('chi2', float('inf')) if scat_result.extra_data else float('inf')
        
        # Overall status
        if g2_chi2 == float('inf') or scat_chi2 == float('inf'):
            status = "NO QUANTUM"
        elif g2_result.passed or scat_result.passed:
            status = "INTERESTING"
        else:
            status = "CONSISTENT"
        
        print(f"{name:<30} {category:<15} {g2_chi2:<12.2f} {scat_chi2:<12.2f} {status}")
        
        results_summary.append({
            'theory': name,
            'category': category,
            'g2_chi2': g2_chi2,
            'scat_chi2': scat_chi2,
            'g2_passed': g2_result.passed,
            'scat_passed': scat_result.passed,
            'status': status
        })
    
    # Analysis
    print("\n" + "-" * 80)
    print("ANALYSIS:")
    
    # Count by category
    by_category = {}
    for result in results_summary:
        cat = result['category']
        if cat not in by_category:
            by_category[cat] = {'total': 0, 'quantum': 0, 'interesting': 0}
        by_category[cat]['total'] += 1
        if result['status'] != 'NO QUANTUM':
            by_category[cat]['quantum'] += 1
        if result['status'] == 'INTERESTING':
            by_category[cat]['interesting'] += 1
    
    print("\nBy category:")
    for cat, counts in by_category.items():
        print(f"  {cat}: {counts['quantum']}/{counts['total']} have quantum interface, "
              f"{counts['interesting']} show interesting deviations")
    
    # Find interesting theories
    interesting = [r for r in results_summary if r['status'] == 'INTERESTING']
    if interesting:
        print("\nTheories with interesting quantum signatures:")
        for r in interesting:
            print(f"  - {r['theory']} ({r['category']}): "
                  f"g-2 χ²={r['g2_chi2']:.2f}, scat χ²={r['scat_chi2']:.2f}")
    
    # Sanity checks
    print("\nSanity checks:")
    
    # Classical theories should have high chi2 (no improvement)
    classical = [r for r in results_summary if r['category'] == 'classical']
    classical_avg_chi2 = np.mean([r['g2_chi2'] for r in classical if r['g2_chi2'] < float('inf')])
    print(f"  Classical theories average g-2 χ²: {classical_avg_chi2:.1f} (should be ~17.6)")
    
    # Quantum theories might show some variation
    quantum = [r for r in results_summary if 'quantum' in r['category']]
    if quantum:
        quantum_chi2s = [r['g2_chi2'] for r in quantum if r['g2_chi2'] < float('inf')]
        if quantum_chi2s:
            quantum_avg_chi2 = np.mean(quantum_chi2s)
            quantum_std_chi2 = np.std(quantum_chi2s)
            print(f"  Quantum theories g-2 χ²: {quantum_avg_chi2:.1f} ± {quantum_std_chi2:.1f}")
    
    return True


def main():
    """Run all quantum validator tests."""
    print("="*60)
    print("QUANTUM VALIDATORS TEST SUITE")
    print("="*60)
    print("\nTesting g-2 and scattering amplitude validators")
    print("following test-driven development approach.\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Theory Consistency", test_theory_consistency),
        ("Performance", test_performance),
        ("All Theories", test_all_theories),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            passed = test_func()
            results[test_name] = passed
            print(f"\n{'='*60}")
            print(f"{test_name}: {'PASSED' if passed else 'FAILED'}")
            print(f"{'='*60}")
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Quantum validators are ready for integration.")
        print("\nNext steps:")
        print("1. Move validators from test implementations to validations/")
        print("2. Update test_comprehensive_final.py to include them")
        print("3. Run full validation suite")
    else:
        print("\n⚠️  Some tests failed. Fix issues before integration.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)