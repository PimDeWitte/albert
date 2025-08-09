"""
Electromagnetic Field Validator

Tests theories that include electromagnetic fields in curved spacetime:
- Maxwell equations in curved spacetime
- Electromagnetic energy-momentum tensor
- Charged black hole solutions (Reissner-Nordström, Kerr-Newman)
- Electromagnetic self-force effects
- Plasma physics in strong gravity

<reason>chain: Electromagnetism in curved spacetime is crucial for astrophysical plasmas and charged objects</reason>
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from physics_agent.validations.base_validation import ObservationalValidator, ValidationResult
from physics_agent.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT, VACUUM_PERMITTIVITY

class ElectromagneticFieldValidator(ObservationalValidator):
    """
    Validates electromagnetic field behavior in curved spacetime.
    
    <reason>chain: Tests consistency of Maxwell equations and electromagnetic stress-energy in gravity</reason>
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Electromagnetic Fields"
        self.description = "Tests electromagnetic field behavior in curved spacetime"
        self.elementary_charge = 1.602176634e-19  # Coulombs
        
    def validate(self, theory, **kwargs) -> ValidationResult:
        """
        Validate theory's electromagnetic predictions.
        
        <reason>chain: Tests Maxwell equations, charged black holes, and plasma physics</reason>
        """
        results = {
            'maxwell_equations': self._test_maxwell_in_curved_spacetime(theory),
            'charged_black_hole': self._test_charged_black_hole_solution(theory),
            'electromagnetic_stress_energy': self._test_em_stress_energy(theory),
            'plasma_frequency': self._test_plasma_oscillations(theory),
            'magnetic_field_limits': self._test_magnetic_field_constraints(theory)
        }
        
        # Calculate overall pass/fail
        passed = all(r.get('passed', False) for r in results.values())
        
        # Calculate aggregate score
        scores = [r.get('score', 0.0) for r in results.values()]
        overall_score = np.mean(scores) if scores else 0.0
        
        return ValidationResult(
            passed=passed,
            score=overall_score,
            details=results,
            notes=self._generate_notes(results, theory)
        )
    
    def _test_maxwell_in_curved_spacetime(self, theory) -> Dict[str, Any]:
        """Test Maxwell equations in curved spacetime: ∇_μ F^μν = 4π J^ν."""
        try:
            if not hasattr(theory, 'maxwell_tensor'):
                return {
                    'passed': True,  # Not required
                    'score': 0.5,
                    'notes': 'Theory does not implement Maxwell tensor'
                }
            
            # Test point: near a black hole
            M = 10 * SOLAR_MASS
            r = 10 * (2 * GRAVITATIONAL_CONSTANT * M / SPEED_OF_LIGHT**2)  # 10 Rs
            
            # Simple radial electric field E_r = Q/r²
            Q = 1e10 * self.elementary_charge  # Large charge
            
            # Get electromagnetic field tensor
            F_tensor = theory.maxwell_tensor(r, Q, M)
            
            # Check antisymmetry: F_μν = -F_νμ
            antisymmetry_error = torch.max(torch.abs(F_tensor + F_tensor.transpose(-2, -1)))
            antisymmetric = antisymmetry_error < 1e-10
            
            # Check covariant derivative (simplified test)
            if hasattr(theory, 'covariant_divergence_em'):
                div_F = theory.covariant_divergence_em(F_tensor, r, M)
                
                # Should equal 4π times current (zero for vacuum)
                vacuum_test = torch.max(torch.abs(div_F)) < 1e-8
            else:
                vacuum_test = True  # Pass if not implemented
            
            passed = antisymmetric and vacuum_test
            
            score = 0.0
            if antisymmetric:
                score += 0.5
            if vacuum_test:
                score += 0.5
            
            return {
                'passed': passed,
                'score': score,
                'antisymmetry_error': float(antisymmetry_error),
                'vacuum_divergence': vacuum_test,
                'notes': 'Maxwell equations satisfied' if passed else 'Maxwell equations violated'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_charged_black_hole_solution(self, theory) -> Dict[str, Any]:
        """Test Reissner-Nordström or Kerr-Newman solutions."""
        try:
            # Test if theory correctly handles charged black holes
            M = 1.0 * SOLAR_MASS
            
            # Maximum charge for black hole: Q_max = M√(G/k_e)
            # where k_e = 1/(4πε_0)
            k_e = 1 / (4 * np.pi * VACUUM_PERMITTIVITY)
            Q_max = M * np.sqrt(GRAVITATIONAL_CONSTANT / k_e)
            
            # Test with Q = 0.5 Q_max (sub-extremal)
            Q = 0.5 * Q_max
            
            if hasattr(theory, 'charged_black_hole_metric'):
                # Get metric at various radii
                r_values = np.array([3, 5, 10, 20]) * (2 * GRAVITATIONAL_CONSTANT * M / SPEED_OF_LIGHT**2)
                
                metrics = []
                for r in r_values:
                    g = theory.charged_black_hole_metric(r, M, Q)
                    metrics.append(g)
                
                # Check horizon structure
                # For RN: r_± = GM/c² ± √((GM/c²)² - GQ²/(4πε_0c⁴))
                r_plus = (GRAVITATIONAL_CONSTANT * M / SPEED_OF_LIGHT**2) * (1 + np.sqrt(1 - (Q/Q_max)**2))
                
                # Check if metric has correct horizon
                if hasattr(theory, 'event_horizon_radius'):
                    r_horizon = theory.event_horizon_radius(M, Q)
                    horizon_error = abs(r_horizon - r_plus) / r_plus
                    horizon_ok = horizon_error < 0.01  # 1% accuracy
                else:
                    horizon_ok = True  # Pass if not implemented
                    horizon_error = 0.0
                
                # Check metric signature remains Lorentzian
                signature_ok = all(self._check_metric_signature(g) for g in metrics)
                
                passed = horizon_ok and signature_ok
                
                score = 0.0
                if horizon_ok:
                    score += 0.6
                if signature_ok:
                    score += 0.4
                
                return {
                    'passed': passed,
                    'score': score,
                    'charge_fraction': Q / Q_max,
                    'horizon_error_percent': horizon_error * 100 if horizon_error else None,
                    'signature_ok': signature_ok,
                    'notes': f'Q/Q_max = {Q/Q_max:.2f}'
                }
            else:
                return {
                    'passed': True,
                    'score': 0.5,
                    'notes': 'Theory does not model charged black holes'
                }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_em_stress_energy(self, theory) -> Dict[str, Any]:
        """Test electromagnetic stress-energy tensor: T_μν = F_μα F_ν^α - (1/4)g_μν F_αβ F^αβ."""
        try:
            if not hasattr(theory, 'electromagnetic_stress_energy'):
                return {
                    'passed': True,
                    'score': 0.5,
                    'notes': 'Theory does not compute EM stress-energy'
                }
            
            # Test with simple magnetic field
            B = 1e12  # Tesla (magnetar-strength field)
            
            # Get stress-energy tensor
            T_em = theory.electromagnetic_stress_energy(B, E=0)  # Pure magnetic field
            
            # Check trace: T = T_μ^μ = 0 for electromagnetic field
            if hasattr(theory, 'metric_tensor'):
                g = theory.metric_tensor(r=1e4)  # Far from source
                g_inv = torch.inverse(g)
                trace = torch.einsum('ij,ij->', T_em, g_inv)
                traceless = abs(trace) < 1e-10
            else:
                traceless = True  # Pass if metric not available
                trace = 0.0
            
            # Check energy density: T_00 = (E² + B²)/(8π) in Gaussian units
            # Convert to SI: T_00 = (ε_0 E² + B²/μ_0)/2
            mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability
            expected_T00 = B**2 / (2 * mu_0)
            
            if T_em.shape[0] >= 4:  # Has time component
                T00_error = abs(T_em[0, 0] - expected_T00) / expected_T00
                energy_ok = T00_error < 0.1  # 10% accuracy
            else:
                energy_ok = True
                T00_error = 0.0
            
            passed = traceless and energy_ok
            
            score = 0.0
            if traceless:
                score += 0.5
            if energy_ok:
                score += 0.5
            
            return {
                'passed': passed,
                'score': score,
                'trace': float(trace) if isinstance(trace, torch.Tensor) else trace,
                'energy_density_error_percent': T00_error * 100 if T00_error else None,
                'magnetic_field_tesla': B,
                'notes': 'EM stress-energy correct' if passed else 'EM stress-energy incorrect'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_plasma_oscillations(self, theory) -> Dict[str, Any]:
        """Test plasma frequency and oscillations in curved spacetime."""
        try:
            if not hasattr(theory, 'plasma_frequency'):
                return {
                    'passed': True,
                    'score': 0.5,
                    'notes': 'Theory does not model plasma physics'
                }
            
            # Typical pulsar magnetosphere parameters
            n_e = 1e17  # electrons/m³
            B = 1e8  # Tesla (typical pulsar)
            M = 1.4 * SOLAR_MASS  # Neutron star
            r = 20e3  # 20 km from center
            
            # Get plasma frequency in curved spacetime
            omega_p = theory.plasma_frequency(n_e, r, M)
            
            # Flat space plasma frequency: ω_p = √(n_e e²/(ε_0 m_e))
            m_e = 9.10938356e-31  # kg
            omega_p_flat = np.sqrt(n_e * self.elementary_charge**2 / (VACUUM_PERMITTIVITY * m_e))
            
            # Should be modified by gravitational redshift
            z = np.sqrt(1 - 2 * GRAVITATIONAL_CONSTANT * M / (r * SPEED_OF_LIGHT**2))
            omega_p_expected = omega_p_flat * z  # Redshifted frequency
            
            error = abs(omega_p - omega_p_expected) / omega_p_expected
            
            # Also test cyclotron frequency if available
            if hasattr(theory, 'cyclotron_frequency'):
                omega_c = theory.cyclotron_frequency(B, r, M)
                omega_c_flat = self.elementary_charge * B / m_e
                omega_c_expected = omega_c_flat * z
                
                cyclotron_error = abs(omega_c - omega_c_expected) / omega_c_expected
                cyclotron_ok = cyclotron_error < 0.1
            else:
                cyclotron_ok = True
                cyclotron_error = None
            
            passed = error < 0.1 and cyclotron_ok
            
            score = 0.0
            if error < 0.1:
                score += 0.6
            if cyclotron_ok:
                score += 0.4
            
            return {
                'passed': passed,
                'score': score,
                'plasma_frequency_hz': omega_p / (2 * np.pi),
                'expected_frequency_hz': omega_p_expected / (2 * np.pi),
                'error_percent': error * 100,
                'cyclotron_error_percent': cyclotron_error * 100 if cyclotron_error else None,
                'gravitational_redshift': z,
                'notes': f'Plasma freq: {omega_p/(2*np.pi):.2e} Hz'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_magnetic_field_constraints(self, theory) -> Dict[str, Any]:
        """Test physical constraints on magnetic fields in strong gravity."""
        try:
            if not hasattr(theory, 'maximum_magnetic_field'):
                return {
                    'passed': True,
                    'score': 0.5,
                    'notes': 'Theory does not constrain magnetic fields'
                }
            
            # Test near neutron star surface
            M = 1.4 * SOLAR_MASS
            R = 12e3  # 12 km radius
            
            # Get maximum allowed field
            B_max = theory.maximum_magnetic_field(R, M)
            
            # Virial limit: magnetic energy < gravitational binding energy
            # B_max ~ √(ρ c²) ~ 10^18 G for nuclear density
            B_virial = 1e14  # Tesla (10^18 Gauss)
            
            # Quantum limit: eB < m_e² c³/ℏ
            # B_QED = m_e² c³/(eℏ) ≈ 4.4 × 10^9 T
            B_QED = 4.4e9  # Tesla
            
            # Check if constraints are reasonable
            above_observed = B_max > 1e11  # Above observed magnetar fields
            below_virial = B_max < 10 * B_virial  # Within order of magnitude
            
            # For extreme fields, check QED corrections
            if B_max > B_QED and hasattr(theory, 'qed_corrections'):
                qed_factor = theory.qed_corrections(B_max)
                qed_ok = 0.5 < qed_factor < 2.0  # Reasonable correction
            else:
                qed_ok = True
            
            passed = above_observed and below_virial and qed_ok
            
            score = 0.0
            if above_observed:
                score += 0.4
            if below_virial:
                score += 0.4
            if qed_ok:
                score += 0.2
            
            return {
                'passed': passed,
                'score': score,
                'maximum_field_tesla': B_max,
                'virial_limit_tesla': B_virial,
                'qed_limit_tesla': B_QED,
                'above_observed': above_observed,
                'below_virial': below_virial,
                'notes': f'B_max = {B_max:.2e} T'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _check_metric_signature(self, metric: torch.Tensor) -> bool:
        """Check if metric has Lorentzian signature (-,+,+,+)."""
        try:
            eigenvalues = torch.linalg.eigvals(metric).real
            negative_count = torch.sum(eigenvalues < 0)
            positive_count = torch.sum(eigenvalues > 0)
            
            # Should have exactly 1 negative and 3 positive eigenvalues
            return negative_count == 1 and positive_count == 3
        except:
            return False
    
    def _generate_notes(self, results: Dict[str, Any], theory) -> str:
        """Generate summary notes."""
        notes = []
        
        if results['maxwell_equations']['passed']:
            notes.append("✓ Maxwell equations")
        else:
            notes.append("✗ Maxwell equations violated")
        
        if 'charge_fraction' in results['charged_black_hole']:
            Q_frac = results['charged_black_hole']['charge_fraction']
            notes.append(f"✓ Charged BH: Q/Q_max={Q_frac:.2f}")
        
        if 'plasma_frequency_hz' in results['plasma_frequency']:
            freq = results['plasma_frequency']['plasma_frequency_hz']
            notes.append(f"✓ Plasma: {freq:.2e} Hz")
        
        if 'maximum_field_tesla' in results['magnetic_field_limits']:
            B_max = results['magnetic_field_limits']['maximum_field_tesla']
            notes.append(f"B_max: {B_max:.2e} T")
        
        return "; ".join(notes)


# Import required constants
try:
    from physics_agent.constants import SOLAR_MASS
except ImportError:
    SOLAR_MASS = 1.989e30  # kg
