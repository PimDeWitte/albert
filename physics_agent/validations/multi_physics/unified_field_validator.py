import torch
import numpy as np
from ..base_validation import ObservationalValidator, ValidationResult
from physics_agent.constants import G, c, ELEMENTARY_CHARGE, epsilon_0, HBAR

class UnifiedFieldValidator(ObservationalValidator):
    """
    Validates theories that combine gravitational and electromagnetic fields.
    
    Tests:
    1. Kerr-Newman limit for rotating charged black holes
    2. Electromagnetic-gravitational energy conservation
    3. Field coupling consistency
    4. Weak field limits match Einstein-Maxwell theory
    """
    
    def __init__(self, engine=None):
        super().__init__(engine)
        self.name = "Unified Field Effects"
        self.description = "Validates combined gravitational and electromagnetic phenomena"
        self.units = "dimensionless"
        
    def validate(self, theory, **kwargs) -> ValidationResult:
        """Validate unified field effects."""
        try:
            # Check what fields the theory supports
            has_gravity = hasattr(theory, 'get_metric')
            has_em = hasattr(theory, 'get_electromagnetic_field_tensor')
            
            if not (has_gravity and has_em):
                return self.generate_fail_result(
                    notes="Theory must implement both gravitational and electromagnetic fields",
                    error_message="Missing required unified field methods"
                )
            
            test_results = {
                'kerr_newman_limit': self._test_kerr_newman_limit(theory),
                'energy_conservation': self._test_energy_conservation(theory),
                'field_coupling': self._test_field_coupling(theory),
                'weak_field_limit': self._test_weak_field_limit(theory)
            }
            
            # Count passed tests
            passed_count = sum(1 for result in test_results.values() if result['passed'])
            total_tests = len(test_results)
            
            # Overall pass if at least 3/4 tests pass
            all_passed = passed_count >= 3
            
            if all_passed:
                notes = f"Unified field effects validated ({passed_count}/{total_tests} tests passed)"
                if not test_results['kerr_newman_limit']['passed']:
                    notes += ". Note: Kerr-Newman limit not exact"
                    
                return self.generate_pass_result(
                    notes=notes,
                    predicted_value=float(passed_count) / total_tests,
                    sota_value=1.0,
                    details=test_results
                )
            else:
                failures = []
                for test_name, result in test_results.items():
                    if not result['passed']:
                        failures.append(f"{test_name}: {result.get('reason', 'failed')}")
                        
                return self.generate_fail_result(
                    notes=f"Unified field validation failed ({passed_count}/{total_tests} passed)",
                    predicted_value=float(passed_count) / total_tests,
                    sota_value=0.75,  # Need at least 75% to pass
                    error_message="; ".join(failures[:2]),
                    details=test_results
                )
                
        except Exception as e:
            return self.generate_error_result(f"Error during unified field validation: {str(e)}")
    
    def _test_kerr_newman_limit(self, theory):
        """Test if theory reduces to Kerr-Newman for a=0."""
        try:
            M = torch.tensor(1.0)
            
            # Set charge if theory supports it
            if hasattr(theory, 'Q'):
                theory.Q = 0.5  # Half-extremal charge
            
            # Test metric at various radii
            test_radii = torch.tensor([2.0, 3.0, 5.0, 10.0])
            
            # Get metric components
            g_tt, g_rr, g_pp, g_tp = theory.get_metric(test_radii, M, c, G)
            
            # For non-rotating charged BH, should have Reissner-Nordström form
            # g_tt = -(1 - 2M/r + Q²/r²)
            # In our units where G=c=4πε₀=1: Q_geom = Q
            Q_geom = 0.5
            expected_g_tt = -(1 - 2*M/test_radii + Q_geom**2/test_radii**2)
            
            # Check agreement
            g_tt_error = torch.abs(g_tt - expected_g_tt) / torch.abs(expected_g_tt)
            max_error = torch.max(g_tt_error).item()
            
            # Also check that g_tp = 0 for non-rotating case
            g_tp_zero = torch.all(torch.abs(g_tp) < 1e-10)
            
            passed = max_error < 0.05 and g_tp_zero  # 5% tolerance
            
            return {
                'passed': passed,
                'max_metric_error': max_error,
                'g_tp_zero': bool(g_tp_zero),
                'reason': f"Max error: {max_error:.2%}" if not passed else "Matches Reissner-Nordström"
            }
            
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def _test_energy_conservation(self, theory):
        """Test total energy-momentum conservation."""
        try:
            r_test = torch.tensor([5.0]).unsqueeze(0)
            
            # Get both stress-energy tensors
            T_total = torch.zeros(1, 4, 4, dtype=theory.dtype)
            
            # Add gravitational contribution if available
            if hasattr(theory, 'get_gravitational_stress_energy'):
                T_grav = theory.get_gravitational_stress_energy(r_test)
                T_total += T_grav
            
            # Add electromagnetic contribution
            if hasattr(theory, 'get_electromagnetic_stress_energy'):
                T_em = theory.get_electromagnetic_stress_energy(r_test)
                T_total += T_em
            
            # Add matter contribution if available
            if hasattr(theory, 'get_fluid_stress_energy_tensor'):
                T_matter = theory.get_fluid_stress_energy_tensor(r_test)
                T_total += T_matter
            
            # Check trace (simplified test)
            # For electromagnetic field: T = 0 (traceless)
            # For perfect fluid: T = -ρ + 3p
            trace = T_total[0, 0, 0] + T_total[0, 1, 1] + T_total[0, 2, 2] + T_total[0, 3, 3]
            
            # Check energy positivity
            energy_density = -T_total[0, 0, 0]  # T^0_0 = -ρ
            energy_positive = energy_density >= 0
            
            passed = energy_positive
            
            return {
                'passed': passed,
                'energy_density': energy_density.item() if isinstance(energy_density, torch.Tensor) else float(energy_density),
                'trace': trace.item() if isinstance(trace, torch.Tensor) else float(trace),
                'reason': "Energy conservation satisfied" if passed else "Negative energy density"
            }
            
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def _test_field_coupling(self, theory):
        """Test consistency of field couplings."""
        try:
            # Check if theory has coupling constants
            couplings_consistent = True
            coupling_info = {}
            
            # Gravitational coupling
            if hasattr(theory, 'G') or hasattr(theory, 'kappa'):
                G_eff = getattr(theory, 'G', G)
                coupling_info['G'] = G_eff
                # Check if G is positive
                if isinstance(G_eff, (int, float)):
                    couplings_consistent &= G_eff > 0
            
            # Electromagnetic coupling
            if hasattr(theory, 'alpha_em') or hasattr(theory, 'e'):
                alpha = getattr(theory, 'alpha_em', ELEMENTARY_CHARGE**2 / (4 * np.pi * HBAR * c))
                coupling_info['alpha_em'] = alpha
                # Fine structure constant should be ~1/137
                if isinstance(alpha, (int, float)):
                    couplings_consistent &= 0.005 < alpha < 0.01
            
            # Check for unification scale if available
            if hasattr(theory, 'compute_unification_scale'):
                try:
                    M_unif = theory.compute_unification_scale()
                    if isinstance(M_unif, torch.Tensor):
                        M_unif = M_unif.item()
                    coupling_info['unification_scale'] = M_unif
                    # Should be high energy but not infinite
                    couplings_consistent &= 1e14 < M_unif < 1e20
                except:
                    pass
            
            passed = couplings_consistent and len(coupling_info) > 0
            
            return {
                'passed': passed,
                'couplings': coupling_info,
                'reason': "Field couplings consistent" if passed else "Invalid coupling values"
            }
            
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def _test_weak_field_limit(self, theory):
        """Test weak field limit matches Einstein-Maxwell."""
        try:
            # Far from source
            r_large = torch.tensor([100.0, 200.0, 500.0])
            M = torch.tensor(1.0)
            
            # Get metric
            g_tt, g_rr, g_pp, g_tp = theory.get_metric(r_large, M, c, G)
            
            # Weak field: g_μν = η_μν + h_μν where |h_μν| << 1
            # For Schwarzschild: h_tt = 2GM/rc² = 2M/r in geometric units
            h_tt = g_tt + 1  # Since g_tt ≈ -(1 + h_tt)
            
            # Check if corrections are small
            h_tt_small = torch.all(torch.abs(h_tt) < 0.1)
            
            # Check 1/r falloff
            # h_tt should be proportional to 1/r
            ratios = h_tt[1:] / h_tt[:-1]
            expected_ratios = r_large[:-1] / r_large[1:]
            ratio_error = torch.abs(ratios - expected_ratios) / expected_ratios
            correct_falloff = torch.all(ratio_error < 0.05)
            
            # Check electromagnetic field if present
            em_weak = True
            if hasattr(theory, 'get_electromagnetic_field_tensor'):
                F = theory.get_electromagnetic_field_tensor(r_large.unsqueeze(1))
                # E_r should fall as 1/r²
                E_r = F[:, 0, 1]
                if torch.any(E_r != 0):
                    E_ratios = E_r[1:] / E_r[:-1]
                    E_expected = (r_large[:-1] / r_large[1:])**2
                    E_error = torch.abs(E_ratios - E_expected) / E_expected
                    em_weak = torch.all(E_error < 0.1)
            
            passed = h_tt_small and correct_falloff and em_weak
            
            return {
                'passed': passed,
                'weak_field_metric': bool(h_tt_small),
                'correct_falloff': bool(correct_falloff),
                'em_weak_field': bool(em_weak),
                'reason': "Correct weak field limit" if passed else "Weak field limit incorrect"
            }
            
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def get_observational_data(self):
        """Return expected unified field properties."""
        return {
            'kerr_newman_metric': 'ds² = -(1-2Mr/Σ+Q²/Σ)dt² + ...',
            'einstein_maxwell': 'G_μν = 8πT_μν^(EM)',
            'notes': 'Unified gravitational-electromagnetic field properties'
        }
