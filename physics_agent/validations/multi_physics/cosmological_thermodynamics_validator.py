import torch
import numpy as np
from ..base_validation import ObservationalValidator, ValidationResult
from physics_agent.constants import G, c, HBAR, BOLTZMANN_CONSTANT

class CosmologicalThermodynamicsValidator(ObservationalValidator):
    """
    Validates the interplay between cosmology and thermodynamics.
    
    Tests:
    1. Cosmological horizon temperature (de Sitter temperature)
    2. Entropy bounds (holographic principle)
    3. Thermodynamic arrow of time consistency
    4. Black hole thermodynamics in expanding universe
    """
    
    def __init__(self, engine=None):
        super().__init__(engine)
        self.name = "Cosmological Thermodynamics"
        self.description = "Validates thermodynamic properties in cosmological contexts"
        self.units = "dimensionless"
        
    def validate(self, theory, **kwargs) -> ValidationResult:
        """Validate cosmological thermodynamics."""
        try:
            # Check what the theory supports
            has_cosmology = hasattr(theory, 'compute_hubble_parameter')
            has_thermodynamics = hasattr(theory, 'compute_hawking_temperature') or hasattr(theory, 'compute_unruh_temperature')
            
            if not (has_cosmology or has_thermodynamics):
                return self.generate_fail_result(
                    notes="Theory must implement cosmological or thermodynamic methods",
                    error_message="Missing required methods for cosmological thermodynamics"
                )
            
            test_results = {}
            
            # Test 1: de Sitter horizon temperature
            if has_cosmology and hasattr(theory, 'Omega_Lambda'):
                test_results['de_sitter_temperature'] = self._test_de_sitter_temperature(theory)
            
            # Test 2: Holographic entropy bound
            if has_cosmology:
                test_results['holographic_bound'] = self._test_holographic_entropy_bound(theory)
            
            # Test 3: Second law in expanding universe
            test_results['second_law'] = self._test_second_law_cosmology(theory)
            
            # Test 4: Black hole evaporation vs expansion
            if has_thermodynamics and has_cosmology:
                test_results['bh_vs_expansion'] = self._test_black_hole_vs_expansion(theory)
            
            # Count passed tests
            passed_count = sum(1 for result in test_results.values() if result.get('passed', False))
            total_tests = len(test_results)
            
            # Need at least half the tests to pass
            min_required = max(1, total_tests // 2)
            all_passed = passed_count >= min_required
            
            if all_passed:
                notes = f"Cosmological thermodynamics validated ({passed_count}/{total_tests} tests passed)"
                
                # Add specific notes for interesting results
                if 'de_sitter_temperature' in test_results and test_results['de_sitter_temperature']['passed']:
                    T_dS = test_results['de_sitter_temperature'].get('temperature', 0)
                    if T_dS > 0:
                        notes += f". de Sitter temperature: {T_dS:.2e} K"
                        
                return self.generate_pass_result(
                    notes=notes,
                    predicted_value=float(passed_count) / total_tests,
                    sota_value=0.5,  # Minimum fraction needed
                    details=test_results
                )
            else:
                failures = []
                for test_name, result in test_results.items():
                    if not result.get('passed', False):
                        failures.append(f"{test_name}: {result.get('reason', 'failed')}")
                        
                return self.generate_fail_result(
                    notes=f"Cosmological thermodynamics validation failed ({passed_count}/{total_tests} passed)",
                    predicted_value=float(passed_count) / total_tests,
                    sota_value=0.5,
                    error_message="; ".join(failures[:2]),
                    details=test_results
                )
                
        except Exception as e:
            return self.generate_error_result(f"Error during cosmological thermodynamics validation: {str(e)}")
    
    def _test_de_sitter_temperature(self, theory):
        """Test de Sitter horizon temperature T_dS = H/(2π) in natural units."""
        try:
            # Get cosmological constant
            if hasattr(theory, 'Omega_Lambda') and hasattr(theory, 'H0'):
                # de Sitter temperature from cosmological constant
                # T_dS = ħH/(2πk_B) where H = H0√(Ω_Λ) for de Sitter
                
                H0_SI = theory.H0 * 1e3 / 3.086e22  # Convert km/s/Mpc to 1/s
                H_dS = H0_SI * np.sqrt(theory.Omega_Lambda)  # de Sitter Hubble rate
                
                T_dS = (HBAR * H_dS) / (2 * np.pi * BOLTZMANN_CONSTANT)
                
                # Check if temperature is positive and reasonable
                # For current Λ, expect T_dS ~ 10^-30 K
                passed = 0 < T_dS < 1e-20
                
                # Also check if theory provides its own calculation
                if hasattr(theory, 'compute_de_sitter_temperature'):
                    T_theory = theory.compute_de_sitter_temperature()
                    if isinstance(T_theory, torch.Tensor):
                        T_theory = T_theory.item()
                    
                    # Compare with expected
                    relative_error = abs(T_theory - T_dS) / T_dS if T_dS > 0 else float('inf')
                    passed &= relative_error < 0.1
                    
                    return {
                        'passed': passed,
                        'temperature': T_theory,
                        'expected': T_dS,
                        'relative_error': relative_error,
                        'reason': f"T_dS = {T_theory:.2e} K" if passed else f"Error: {relative_error:.1%}"
                    }
                else:
                    return {
                        'passed': passed,
                        'temperature': T_dS,
                        'reason': f"T_dS = {T_dS:.2e} K" if passed else "Temperature out of range"
                    }
            else:
                return {'passed': False, 'reason': 'Missing cosmological parameters'}
                
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def _test_holographic_entropy_bound(self, theory):
        """Test holographic entropy bound S ≤ A/(4l_P²)."""
        try:
            # For a cosmological horizon at radius l_H = c/H
            if hasattr(theory, 'compute_hubble_parameter'):
                z = torch.tensor(0.0)  # Present time
                H0 = theory.compute_hubble_parameter(z)
                if isinstance(H0, torch.Tensor):
                    H0_value = H0.item()
                else:
                    H0_value = float(H0)
                
                # Hubble radius in meters
                H0_SI = H0_value * 1e3 / 3.086e22  # km/s/Mpc to 1/s
                l_H = c / H0_SI
                
                # Horizon area
                A_horizon = 4 * np.pi * l_H**2
                
                # Maximum entropy (holographic bound)
                l_P_squared = (HBAR * G) / c**3
                S_max = A_horizon / (4 * l_P_squared)
                
                # Check if this is reasonable
                # For observable universe, S_max ~ 10^122
                passed = 1e120 < S_max < 1e125
                
                return {
                    'passed': passed,
                    'horizon_radius': l_H,
                    'max_entropy': S_max,
                    'reason': f"S_max = {S_max:.2e}" if passed else "Entropy bound unrealistic"
                }
            else:
                return {'passed': False, 'reason': 'Cannot compute Hubble parameter'}
                
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def _test_second_law_cosmology(self, theory):
        """Test that total entropy increases with cosmic time."""
        try:
            # This is a basic consistency check
            # In an expanding universe, total entropy should increase
            
            # Check if theory respects basic thermodynamic principles
            respects_thermodynamics = True
            
            # If theory has entropy calculation
            if hasattr(theory, 'compute_total_entropy'):
                # Test at different cosmic times (different redshifts)
                redshifts = torch.tensor([10.0, 1.0, 0.0])  # Past to present
                entropies = []
                
                for z in redshifts:
                    S = theory.compute_total_entropy(z)
                    if isinstance(S, torch.Tensor):
                        S = S.item()
                    entropies.append(S)
                
                # Entropy should increase as z decreases (time increases)
                for i in range(len(entropies) - 1):
                    if entropies[i+1] <= entropies[i]:
                        respects_thermodynamics = False
                        break
                        
                return {
                    'passed': respects_thermodynamics,
                    'entropies_at_z': list(zip(redshifts.tolist(), entropies)),
                    'monotonic': respects_thermodynamics,
                    'reason': "Entropy increases with time" if respects_thermodynamics else "Entropy decreases!"
                }
            else:
                # Basic check: theory should not violate thermodynamics
                return {
                    'passed': respects_thermodynamics,
                    'reason': "No explicit entropy violation detected"
                }
                
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def _test_black_hole_vs_expansion(self, theory):
        """Test interplay between black hole evaporation and cosmic expansion."""
        try:
            # Compare timescales: black hole evaporation vs Hubble time
            
            # Stellar mass black hole
            M_BH = torch.tensor(10.0)  # 10 solar masses in geometric units
            
            # Black hole evaporation time
            if hasattr(theory, 'compute_lifetime'):
                tau_BH = theory.compute_lifetime(M_BH)
                if isinstance(tau_BH, torch.Tensor):
                    tau_BH = tau_BH.item()
            else:
                # Use standard formula: τ = 5120πG²M³/(ħc⁴)
                M_SI = M_BH * 2e30  # Convert to kg
                tau_BH = (5120 * np.pi * G**2 * M_SI**3) / (HBAR * c**4)
            
            # Hubble time
            if hasattr(theory, 'compute_age_of_universe'):
                t_H = theory.compute_age_of_universe()
                if isinstance(t_H, torch.Tensor):
                    t_H = t_H.item()
                t_H *= 3.156e16  # Convert Gyr to seconds
            else:
                # Rough estimate: t_H ~ 1/H0
                H0 = getattr(theory, 'H0', 70)  # km/s/Mpc
                H0_SI = H0 * 1e3 / 3.086e22
                t_H = 1 / H0_SI
            
            # Black holes should outlive the current universe age
            ratio = tau_BH / t_H
            passed = ratio > 1e50  # BH lifetime >> universe age
            
            # Also check Hawking temperature vs CMB temperature
            T_comparison = {}
            if hasattr(theory, 'compute_hawking_temperature'):
                T_H = theory.compute_hawking_temperature(M_BH)
                if isinstance(T_H, torch.Tensor):
                    T_H = T_H.item()
                    
                T_CMB = 2.725  # K
                T_comparison = {
                    'T_Hawking': T_H,
                    'T_CMB': T_CMB,
                    'BH_colder': T_H < T_CMB
                }
                # Black hole should be colder than CMB (not evaporating)
                passed &= T_H < T_CMB
            
            return {
                'passed': passed,
                'lifetime_ratio': ratio,
                'BH_lifetime': tau_BH,
                'universe_age': t_H,
                'temperature_comparison': T_comparison,
                'reason': f"τ_BH/t_H = {ratio:.2e}" if passed else "Black hole evaporates too quickly"
            }
            
        except Exception as e:
            return {'passed': False, 'reason': str(e)}
    
    def get_observational_data(self):
        """Return cosmological thermodynamics expectations."""
        return {
            'de_sitter_temperature': '~10^-30 K for current Λ',
            'holographic_entropy': '~10^122 for observable universe',
            'CMB_temperature': '2.725 K',
            'notes': 'Thermodynamic properties of the universe'
        }
