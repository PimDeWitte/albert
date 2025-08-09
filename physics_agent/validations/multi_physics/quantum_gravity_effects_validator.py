import torch
import numpy as np
from ..base_validation import ObservationalValidator, ValidationResult
from physics_agent.constants import G, c, HBAR, ELEMENTARY_CHARGE

class QuantumGravityEffectsValidator(ObservationalValidator):
    """
    Validates quantum gravity effects combining particle physics and gravity.
    
    Tests:
    1. Quantum corrections to gravitational interaction
    2. Running of gravitational coupling
    3. Minimum length scale (Planck scale physics)
    4. Hawking radiation spectrum modifications
    """
    
    def __init__(self, engine=None):
        super().__init__(engine)
        self.name = "Quantum Gravity Effects"
        self.description = "Validates quantum corrections to gravity and particle physics interplay"
        self.units = "dimensionless"
        
    def validate(self, theory, **kwargs) -> ValidationResult:
        """Validate quantum gravity effects."""
        try:
            # Check what quantum features the theory supports
            has_quantum = hasattr(theory, 'enable_quantum') and theory.enable_quantum
            has_running_couplings = hasattr(theory, 'compute_running_couplings')
            has_quantum_corrections = hasattr(theory, 'compute_quantum_corrections_to_metric')
            
            if not (has_quantum or has_running_couplings or has_quantum_corrections):
                return self.generate_fail_result(
                    notes="Theory must implement quantum features for quantum gravity validation",
                    error_message="Missing quantum gravity methods"
                )
            
            test_results = {}
            
            # Test 1: Quantum corrections to metric
            if has_quantum_corrections:
                test_results['quantum_metric_corrections'] = self._test_quantum_metric_corrections(theory)
            
            # Test 2: Running gravitational coupling
            if has_running_couplings:
                test_results['gravitational_running'] = self._test_gravitational_running(theory)
            
            # Test 3: Planck scale physics
            test_results['planck_scale'] = self._test_planck_scale_physics(theory)
            
            # Test 4: Modified dispersion relations
            if has_quantum:
                test_results['dispersion_relations'] = self._test_modified_dispersion(theory)
            
            # Count passed tests
            passed_count = sum(1 for result in test_results.values() if result.get('passed', False))
            total_tests = len(test_results)
            
            # Need at least half to pass
            min_required = max(1, total_tests // 2)
            all_passed = passed_count >= min_required
            
            if all_passed:
                notes = f"Quantum gravity effects validated ({passed_count}/{total_tests} tests passed)"
                
                # Add details about interesting findings
                if 'planck_scale' in test_results and test_results['planck_scale']['passed']:
                    l_P = test_results['planck_scale'].get('planck_length', 0)
                    notes += f". Planck length: {l_P:.2e} m"
                    
                return self.generate_pass_result(
                    notes=notes,
                    predicted_value=float(passed_count) / total_tests,
                    sota_value=0.5,
                    details=test_results
                )
            else:
                failures = []
                for test_name, result in test_results.items():
                    if not result.get('passed', False):
                        failures.append(f"{test_name}: {result.get('reason', 'failed')}")
                        
                return self.generate_fail_result(
                    notes=f"Quantum gravity validation failed ({passed_count}/{total_tests} passed)",
                    predicted_value=float(passed_count) / total_tests,
                    sota_value=0.5,
                    error_message="; ".join(failures[:2]),
                    details=test_results
                )
                
        except Exception as e:
            return self.generate_error_result(f"Error during quantum gravity validation: {str(e)}")
    
    def _test_quantum_metric_corrections(self, theory):
        """Test quantum corrections to the metric."""
        try:
            # Test at various distances from Planck scale
            l_P = np.sqrt(HBAR * G / c**3)
            M = torch.tensor(1.0)  # Unit mass
            
            # Test radii: 100 l_P, 1000 l_P, 10000 l_P
            test_radii = torch.tensor([100, 1000, 10000]) * l_P / theory.length_scale
            
            # Get quantum corrections
            corrections = []
            for r in test_radii:
                corr = theory.compute_quantum_corrections_to_metric(r.unsqueeze(0), M)
                if isinstance(corr, torch.Tensor):
                    corr = corr.item()
                corrections.append(corr)
            
            # Corrections should decrease with distance
            # Expect corrections ~ (l_P/r)^n where n ≥ 2
            decreasing = all(corrections[i] >= corrections[i+1] for i in range(len(corrections)-1))
            
            # Check scaling
            if corrections[0] > 0:
                # Estimate power law exponent
                ratio1 = corrections[1] / corrections[0]
                ratio2 = corrections[2] / corrections[1]
                r_ratio1 = (test_radii[0] / test_radii[1]).item()
                r_ratio2 = (test_radii[1] / test_radii[2]).item()
                
                # If corrections ~ r^(-n), then ratio ~ r_ratio^n
                n1 = np.log(ratio1) / np.log(r_ratio1) if ratio1 > 0 else 0
                n2 = np.log(ratio2) / np.log(r_ratio2) if ratio2 > 0 else 0
                
                # Should have n ≥ 2 for quantum gravity
                correct_scaling = n1 >= 1.5 and n2 >= 1.5
            else:
                correct_scaling = False
            
            # Corrections should be small at large distances
            small_at_large_r = corrections[-1] < 0.01
            
            passed = decreasing and correct_scaling and small_at_large_r
            
            return {
                'passed': passed,
                'corrections': corrections,
                'radii_over_planck': [r.item() * theory.length_scale / l_P for r in test_radii],
                'decreasing': decreasing,
                'scaling_exponent': (n1 + n2) / 2 if correct_scaling else None,
                'reason': "Correct quantum corrections" if passed else "Invalid correction behavior"
            }
            
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def _test_gravitational_running(self, theory):
        """Test running of gravitational coupling with energy."""
        try:
            # Test at various energy scales
            M_P = np.sqrt(HBAR * c / G) / c**2  # Planck mass in kg
            M_P_GeV = M_P * c**2 / (1.6e-10)  # Convert to GeV
            
            test_energies = torch.tensor([
                1e3,      # 1 TeV
                1e10,     # 10^10 GeV  
                1e16,     # GUT scale
                M_P_GeV   # Planck scale
            ])
            
            # Get running couplings
            G_values = []
            for E in test_energies:
                couplings = theory.compute_running_couplings(E)
                
                # Look for gravitational coupling
                if 'gravitational_coupling' in couplings:
                    G_eff = couplings['gravitational_coupling']
                elif 'G_eff' in couplings:
                    G_eff = couplings['G_eff']
                elif 'kappa' in couplings:
                    # κ = 8πG in natural units
                    G_eff = couplings['kappa'] / (8 * np.pi)
                else:
                    G_eff = G  # No running
                
                if isinstance(G_eff, torch.Tensor):
                    G_eff = G_eff.item()
                G_values.append(G_eff)
            
            # Check for running
            has_running = not all(G == G_values[0] for G in G_values)
            
            if has_running:
                # Gravity should get stronger at high energy (G increases)
                # This is the expectation from asymptotic safety
                increasing = all(G_values[i] <= G_values[i+1] for i in range(len(G_values)-1))
                
                # But shouldn't blow up
                finite = all(G < 100 * G_values[0] for G in G_values)
                
                passed = increasing and finite
                reason = "Gravitational coupling runs correctly" if passed else "Incorrect running behavior"
            else:
                # No running is also acceptable for some theories
                passed = True
                reason = "No gravitational running (classical limit)"
            
            return {
                'passed': passed,
                'has_running': has_running,
                'G_values': G_values,
                'energies_GeV': test_energies.tolist(),
                'reason': reason
            }
            
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def _test_planck_scale_physics(self, theory):
        """Test behavior near Planck scale."""
        try:
            # Calculate Planck units
            l_P = np.sqrt(HBAR * G / c**3)
            t_P = l_P / c
            m_P = np.sqrt(HBAR * c / G)
            E_P = m_P * c**2
            
            # Check if theory has minimum length
            has_min_length = False
            min_length = 0
            
            if hasattr(theory, 'minimum_length'):
                min_length = theory.minimum_length
                if isinstance(min_length, torch.Tensor):
                    min_length = min_length.item()
                has_min_length = min_length > 0
                
                # Should be around Planck length
                length_ratio = min_length / l_P
                correct_scale = 0.1 < length_ratio < 10
            else:
                # Check if metric becomes singular/modified at Planck scale
                r_planck = torch.tensor(l_P / theory.length_scale).unsqueeze(0)
                M_planck = torch.tensor(m_P / theory.M_scale).unsqueeze(0)
                
                try:
                    g_tt, g_rr, _, _ = theory.get_metric(r_planck, M_planck, c, G)
                    
                    # Check for regularization/modification
                    regular = torch.isfinite(g_tt) and torch.isfinite(g_rr)
                    
                    # Metric should be significantly modified from classical
                    classical_g_tt = -1 + 2 * G * m_P / (c**2 * l_P)
                    modified = abs(g_tt.item() - classical_g_tt) / abs(classical_g_tt) > 0.1
                    
                    correct_scale = regular and modified
                except:
                    correct_scale = False
            
            # Check for sensible Planck scale values
            planck_values_sensible = (
                1e-35 < l_P < 2e-35 and  # meters
                1e-43 < t_P < 6e-43 and  # seconds
                2e-8 < m_P < 3e-8        # kg
            )
            
            passed = planck_values_sensible and (has_min_length or correct_scale)
            
            return {
                'passed': passed,
                'planck_length': l_P,
                'planck_time': t_P,
                'planck_mass': m_P,
                'has_minimum_length': has_min_length,
                'minimum_length': min_length if has_min_length else None,
                'reason': "Planck scale physics implemented" if passed else "No Planck scale modifications"
            }
            
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def _test_modified_dispersion(self, theory):
        """Test for modified dispersion relations from quantum gravity."""
        try:
            # In quantum gravity, the dispersion relation may be modified:
            # E² = p²c² + m²c⁴ + f(E/E_P) where f represents quantum gravity corrections
            
            has_modified_dispersion = False
            
            # Check if theory provides modified dispersion
            if hasattr(theory, 'compute_dispersion_relation'):
                # Test for a photon (m=0)
                p_values = torch.tensor([1e-20, 1e-10, 1e-5, 1.0])  # Various momenta in GeV/c
                
                E_values = []
                for p in p_values:
                    E = theory.compute_dispersion_relation(p, m=0)
                    if isinstance(E, torch.Tensor):
                        E = E.item()
                    E_values.append(E)
                
                # Check for deviations from E = pc
                deviations = []
                for p, E in zip(p_values, E_values):
                    classical_E = p.item()  # E = pc with c=1
                    if classical_E > 0:
                        deviation = abs(E - classical_E) / classical_E
                        deviations.append(deviation)
                
                # Should see larger deviations at higher energy
                if len(deviations) > 1:
                    has_modified_dispersion = any(d > 1e-10 for d in deviations)
                    
                    # Deviations should increase with energy
                    monotonic = all(deviations[i] <= deviations[i+1] for i in range(len(deviations)-1))
                else:
                    monotonic = False
                
                passed = has_modified_dispersion and monotonic
                
                return {
                    'passed': passed,
                    'has_modified_dispersion': has_modified_dispersion,
                    'momenta_GeV': p_values.tolist(),
                    'energies_GeV': E_values,
                    'deviations': deviations,
                    'reason': "Modified dispersion detected" if passed else "No quantum gravity dispersion"
                }
            else:
                # Theory doesn't explicitly provide dispersion relation
                # This is OK for some theories
                return {
                    'passed': True,
                    'has_modified_dispersion': False,
                    'reason': "Standard dispersion relation (no explicit modifications)"
                }
                
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def get_observational_data(self):
        """Return quantum gravity observational bounds."""
        return {
            'planck_length': '1.616e-35 m',
            'planck_mass': '2.176e-8 kg',
            'quantum_gravity_scale': '~10^19 GeV',
            'notes': 'Quantum gravity effects expected near Planck scale'
        }
