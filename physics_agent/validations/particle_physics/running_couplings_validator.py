import torch
import numpy as np
from ..base_validation import ObservationalValidator, ValidationResult

class RunningCouplingsValidator(ObservationalValidator):
    """
    Validates running of coupling constants with energy scale.
    
    Tests:
    1. Strong coupling decreases with energy (asymptotic freedom)
    2. Electromagnetic coupling increases with energy
    3. Correct values at Z boson mass scale
    4. Unification scale predictions
    """
    
    def __init__(self, engine=None):
        super().__init__(engine)
        self.name = "Running Couplings"
        self.description = "Validates renormalization group running of Standard Model couplings"
        self.units = "dimensionless"
        
    def validate(self, theory, **kwargs) -> ValidationResult:
        """Validate running coupling behavior."""
        try:
            # Check if theory provides running couplings
            if not hasattr(theory, 'compute_running_couplings'):
                return self.generate_fail_result(
                    notes="Theory does not implement compute_running_couplings method",
                    error_message="Missing required particle physics methods"
                )
            
            # Test at various energy scales
            # Energy scales in GeV
            M_Z = 91.2  # Z boson mass
            test_scales = torch.tensor([
                10.0,      # Low energy
                M_Z,       # Electroweak scale
                1000.0,    # TeV scale
                1e4,       # 10 TeV
                1e16       # Near GUT scale
            ])
            
            results_at_scales = []
            tests_passed = {
                'alpha_s_decreasing': True,
                'alpha_em_increasing': True,
                'z_mass_values': True,
                'physical_ranges': True
            }
            
            # Get couplings at each scale
            for E in test_scales:
                couplings = theory.compute_running_couplings(E)
                
                alpha_s = couplings.get('alpha_s', 0)
                alpha_em = couplings.get('alpha_em', 0)
                alpha_w = couplings.get('alpha_w', 0)
                
                # Convert to values if tensors
                if isinstance(alpha_s, torch.Tensor):
                    alpha_s = alpha_s.item()
                if isinstance(alpha_em, torch.Tensor):
                    alpha_em = alpha_em.item()
                if isinstance(alpha_w, torch.Tensor):
                    alpha_w = alpha_w.item()
                
                results_at_scales.append({
                    'energy': E.item(),
                    'alpha_s': alpha_s,
                    'alpha_em': alpha_em,
                    'alpha_w': alpha_w
                })
            
            # Test 1: Check alpha_s decreases with energy (asymptotic freedom)
            alpha_s_values = [r['alpha_s'] for r in results_at_scales]
            for i in range(1, len(alpha_s_values)):
                if alpha_s_values[i] >= alpha_s_values[i-1]:
                    tests_passed['alpha_s_decreasing'] = False
                    break
            
            # Test 2: Check alpha_em increases with energy
            alpha_em_values = [r['alpha_em'] for r in results_at_scales]
            for i in range(1, len(alpha_em_values)):
                if alpha_em_values[i] <= alpha_em_values[i-1]:
                    tests_passed['alpha_em_increasing'] = False
                    break
            
            # Test 3: Check values at Z mass
            # Find result closest to M_Z
            z_result = None
            for r in results_at_scales:
                if abs(r['energy'] - M_Z) < 1.0:
                    z_result = r
                    break
            
            if z_result:
                # Expected values at M_Z
                alpha_s_Z_expected = 0.118  # PDG value
                alpha_em_Z_expected = 1/128.9  # PDG value
                
                alpha_s_error = abs(z_result['alpha_s'] - alpha_s_Z_expected) / alpha_s_Z_expected
                alpha_em_error = abs(z_result['alpha_em'] - alpha_em_Z_expected) / alpha_em_Z_expected
                
                # Allow 10% deviation
                tests_passed['z_mass_values'] = alpha_s_error < 0.1 and alpha_em_error < 0.1
            
            # Test 4: Check all values are in physical ranges
            for r in results_at_scales:
                if not (0 < r['alpha_s'] < 10):  # alpha_s should be positive and not too large
                    tests_passed['physical_ranges'] = False
                if not (0 < r['alpha_em'] < 1):  # alpha_em should be small
                    tests_passed['physical_ranges'] = False
                if not (0 < r['alpha_w'] < 1):  # alpha_w should be small
                    tests_passed['physical_ranges'] = False
            
            # Test 5: Check unification scale if available
            unification_test = {'available': False, 'passed': False}
            if hasattr(theory, 'compute_unification_scale'):
                M_GUT = theory.compute_unification_scale()
                if isinstance(M_GUT, torch.Tensor):
                    M_GUT_value = M_GUT.item()
                else:
                    M_GUT_value = float(M_GUT)
                
                # GUT scale should be around 10^16 GeV
                unification_test['available'] = True
                unification_test['scale'] = M_GUT_value
                unification_test['passed'] = 1e15 < M_GUT_value < 1e17
                
                # Get couplings at GUT scale
                if unification_test['passed']:
                    gut_couplings = theory.compute_running_couplings(torch.tensor(M_GUT_value))
                    # Check if couplings are close at GUT scale
                    alpha_gut = [
                        gut_couplings.get('alpha_s', 0),
                        gut_couplings.get('alpha_em', 0) * 5/3,  # Include normalization
                        gut_couplings.get('alpha_w', 0)
                    ]
                    alpha_gut = [a.item() if isinstance(a, torch.Tensor) else float(a) for a in alpha_gut]
                    
                    # Check if they're within 20% of each other
                    alpha_mean = np.mean(alpha_gut)
                    deviations = [abs(a - alpha_mean) / alpha_mean for a in alpha_gut]
                    unification_test['unified'] = all(d < 0.2 for d in deviations)
            
            # Overall pass/fail
            all_basic_tests = all(tests_passed.values())
            
            results = {
                'scales_tested': results_at_scales,
                'tests': tests_passed,
                'unification': unification_test,
                'z_mass_comparison': {
                    'alpha_s': z_result['alpha_s'] if z_result else None,
                    'alpha_s_expected': 0.118,
                    'alpha_em': z_result['alpha_em'] if z_result else None,
                    'alpha_em_expected': 1/128.9
                }
            }
            
            if all_basic_tests:
                notes = "Running couplings correctly implemented"
                if unification_test['available'] and unification_test['passed']:
                    notes += f". GUT scale: {unification_test['scale']:.2e} GeV"
                    if unification_test.get('unified', False):
                        notes += " with coupling unification"
                        
                return self.generate_pass_result(
                    notes=notes,
                    predicted_value=z_result['alpha_s'] if z_result else 0.118,
                    sota_value=0.118,  # PDG value for alpha_s(M_Z)
                    details=results
                )
            else:
                failures = []
                if not tests_passed['alpha_s_decreasing']:
                    failures.append("α_s not decreasing (no asymptotic freedom)")
                if not tests_passed['alpha_em_increasing']:
                    failures.append("α_em not increasing with energy")
                if not tests_passed['z_mass_values']:
                    failures.append("Incorrect values at Z mass")
                if not tests_passed['physical_ranges']:
                    failures.append("Couplings outside physical ranges")
                    
                return self.generate_fail_result(
                    notes=f"Running coupling validation failed",
                    predicted_value=z_result['alpha_s'] if z_result else 0.0,
                    sota_value=0.118,
                    error_message="; ".join(failures),
                    details=results
                )
                
        except Exception as e:
            return self.generate_error_result(f"Error during running couplings validation: {str(e)}")
    
    def get_observational_data(self):
        """Return PDG values for coupling constants."""
        return {
            'alpha_s_MZ': 0.118,
            'alpha_em_MZ': 1/128.9,
            'sin2_theta_W': 0.2229,
            'notes': 'Particle Data Group 2023 values at M_Z = 91.2 GeV'
        }
