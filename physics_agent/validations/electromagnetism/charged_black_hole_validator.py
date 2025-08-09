import torch
import numpy as np
from ..base_validation import ObservationalValidator, ValidationResult
from physics_agent.constants import G, c, epsilon_0

class ChargedBlackHoleValidator(ObservationalValidator):
    """
    Validates properties of charged black holes (Reissner-Nordström).
    
    Tests:
    1. Correct horizon structure (r± = M ± √(M² - Q²))
    2. Extremal limit (Q = M)
    3. Electromagnetic field strength
    4. No naked singularities for physical charges
    """
    
    def __init__(self, engine=None):
        super().__init__(engine)
        self.name = "Charged Black Hole Properties"
        self.description = "Validates Reissner-Nordström and charged black hole solutions"
        self.units = "geometric"
        
    def validate(self, theory, **kwargs) -> ValidationResult:
        """Validate charged black hole properties."""
        try:
            # Check if theory has electromagnetic field tensor
            if not hasattr(theory, 'get_electromagnetic_field_tensor'):
                return self.generate_fail_result(
                    notes="Theory does not implement get_electromagnetic_field_tensor method",
                    error_message="Missing required electromagnetic methods"
                )
            
            # Test parameters
            M = torch.tensor(1.0)  # Unit mass
            test_charges = torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0])  # Various Q/M ratios
            
            test_results = []
            all_tests_passed = True
            
            for Q_ratio in test_charges:
                Q = Q_ratio * M
                
                # Set charge if theory supports it
                if hasattr(theory, 'Q'):
                    theory.Q = Q.item()
                
                # Test 1: Check horizons
                horizon_test_passed = True
                if hasattr(theory, 'compute_horizons'):
                    r_plus, r_minus = theory.compute_horizons(M)
                    
                    if Q_ratio < 1.0:  # Sub-extremal
                        # Expected horizons
                        discriminant = torch.sqrt(M**2 - Q**2)
                        r_plus_expected = M + discriminant
                        r_minus_expected = M - discriminant
                        
                        if r_plus is not None and r_minus is not None:
                            r_plus_error = abs(r_plus - r_plus_expected) / r_plus_expected
                            r_minus_error = abs(r_minus - r_minus_expected) / r_minus_expected if r_minus_expected > 0 else 0
                            
                            horizon_test_passed = r_plus_error < 0.01 and r_minus_error < 0.01
                        else:
                            horizon_test_passed = False
                    elif Q_ratio == 1.0:  # Extremal
                        # Should have degenerate horizons at r = M
                        if r_plus is not None and r_minus is not None:
                            horizon_test_passed = abs(r_plus - M) < 0.01 and abs(r_minus - M) < 0.01
                        else:
                            horizon_test_passed = False
                
                # Test 2: Check electromagnetic field
                r_test = torch.tensor([2.0, 5.0, 10.0]).unsqueeze(1)  # Test at various radii
                F_tensor = theory.get_electromagnetic_field_tensor(r_test)
                
                # Electric field should be F_tr = Q/r²
                E_r_predicted = F_tensor[:, 0, 1]  # F_tr component
                E_r_expected = Q / r_test.squeeze()**2
                
                field_errors = torch.abs(E_r_predicted - E_r_expected) / (E_r_expected + 1e-10)
                field_test_passed = torch.all(field_errors < 0.01)
                
                # Test 3: Check metric at horizon for extremal case
                metric_test_passed = True
                if Q_ratio >= 0.99:  # Near-extremal
                    r_horizon = M.unsqueeze(0)
                    g_tt, g_rr, _, _ = theory.get_metric(r_horizon, M, c, G)
                    
                    # Near horizon, metric should be regular
                    if torch.any(torch.isnan(g_tt)) or torch.any(torch.isnan(g_rr)):
                        metric_test_passed = False
                    # g_tt should vanish at horizon
                    if abs(g_tt.item()) > 0.1:
                        metric_test_passed = False
                
                # Store results
                test_passed = horizon_test_passed and field_test_passed and metric_test_passed
                all_tests_passed &= test_passed
                
                test_results.append({
                    'Q/M': Q_ratio.item(),
                    'horizon_test': horizon_test_passed,
                    'field_test': field_test_passed,
                    'metric_test': metric_test_passed,
                    'overall': test_passed
                })
            
            # Test 4: Energy conditions for electromagnetic field
            em_energy_test_passed = True
            if hasattr(theory, 'get_electromagnetic_stress_energy'):
                r_test = torch.tensor([5.0]).unsqueeze(0)
                T_EM = theory.get_electromagnetic_stress_energy(r_test)
                
                # Electromagnetic stress-energy should satisfy:
                # T_00 ≥ 0 (positive energy density)
                # T_ii have specific structure
                rho_EM = T_EM[0, 0, 0]
                em_energy_test_passed = rho_EM >= 0
            
            # Overall results
            results = {
                'test_results': test_results,
                'em_energy_condition': em_energy_test_passed,
                'all_tests': all_tests_passed and em_energy_test_passed
            }
            
            if all_tests_passed and em_energy_test_passed:
                return self.generate_pass_result(
                    notes="Charged black hole properties correctly implemented",
                    predicted_value=1.0,  # All tests passed
                    sota_value=1.0,
                    details=results
                )
            else:
                failures = []
                for result in test_results:
                    if not result['overall']:
                        Q_M = result['Q/M']
                        if not result['horizon_test']:
                            failures.append(f"Horizon structure wrong at Q/M={Q_M}")
                        if not result['field_test']:
                            failures.append(f"E-field wrong at Q/M={Q_M}")
                        if not result['metric_test']:
                            failures.append(f"Metric singular at Q/M={Q_M}")
                
                if not em_energy_test_passed:
                    failures.append("EM energy condition violated")
                    
                return self.generate_fail_result(
                    notes=f"Charged black hole validation failed",
                    predicted_value=sum(r['overall'] for r in test_results) / len(test_results),
                    sota_value=1.0,
                    error_message="; ".join(failures[:3]),  # Limit to first 3 failures
                    details=results
                )
                
        except Exception as e:
            return self.generate_error_result(f"Error during charged black hole validation: {str(e)}")
    
    def get_observational_data(self):
        """Return theoretical expectations for charged black holes."""
        return {
            'horizon_formula': 'r± = M ± √(M² - Q²)',
            'extremal_condition': 'Q = M',
            'electric_field': 'E_r = Q/r²',
            'notes': 'Reissner-Nordström solution properties'
        }
