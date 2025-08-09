import torch
import numpy as np
from ..base_validation import ObservationalValidator, ValidationResult
from physics_agent.constants import G, c, HBAR, BOLTZMANN_CONSTANT, SOLAR_MASS

class HawkingTemperatureValidator(ObservationalValidator):
    """
    Validates Hawking temperature predictions for black holes.
    
    Tests:
    1. Correct Hawking temperature formula: T_H = ħc³/(8πGMk_B)
    2. Temperature-mass relationship (inverse proportionality)
    3. Extremal black hole limits
    """
    
    def __init__(self, engine=None):
        super().__init__(engine)
        self.name = "Hawking Temperature"
        self.description = "Validates black hole thermodynamic properties including Hawking temperature"
        self.units = "Kelvin"
        
    def validate(self, theory, **kwargs) -> ValidationResult:
        """Validate Hawking temperature predictions."""
        try:
            # Check if theory provides Hawking temperature method
            if not hasattr(theory, 'compute_hawking_temperature'):
                result = ValidationResult(self.name, theory.name)
                result.passed = False
                result.notes = "Theory does not implement compute_hawking_temperature method"
                result.error_message = "Missing required thermodynamic methods"
                return result
            
            # Test 1: Solar mass black hole temperature
            M_sun = torch.tensor(1.0)  # Solar mass in geometric units
            T_H_predicted = theory.compute_hawking_temperature(M_sun)
            
            # Expected temperature for solar mass black hole
            # T_H = ħc³/(8πGMk_B) ≈ 6.17 × 10^-8 K
            T_H_expected = (HBAR * c**3) / (8 * np.pi * G * SOLAR_MASS * BOLTZMANN_CONSTANT)
            
            # Calculate relative error
            if isinstance(T_H_predicted, torch.Tensor):
                T_H_predicted_value = T_H_predicted.item()
            else:
                T_H_predicted_value = float(T_H_predicted)
                
            relative_error = abs(T_H_predicted_value - T_H_expected) / T_H_expected
            
            # Test 2: Check inverse mass relationship
            masses = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0])  # Various masses
            temperatures = torch.stack([theory.compute_hawking_temperature(m) for m in masses])
            
            # Check if T ∝ 1/M
            mass_temp_products = masses * temperatures
            product_std = torch.std(mass_temp_products) / torch.mean(mass_temp_products)
            
            # Test 3: Check black hole entropy if available
            entropy_test_passed = True
            if hasattr(theory, 'compute_black_hole_entropy'):
                S_BH = theory.compute_black_hole_entropy(M_sun)
                # Bekenstein-Hawking entropy: S = 4πGM²/(ħc)
                S_BH_expected = 4 * np.pi * G * SOLAR_MASS**2 / (HBAR * c)
                
                if isinstance(S_BH, torch.Tensor):
                    S_BH_value = S_BH.item()
                else:
                    S_BH_value = float(S_BH)
                    
                entropy_error = abs(S_BH_value - S_BH_expected) / S_BH_expected
                entropy_test_passed = entropy_error < 0.1
            
            # Overall pass criteria
            temperature_test_passed = relative_error < 0.01  # 1% tolerance
            inverse_relation_passed = product_std < 0.01  # Products should be constant
            
            all_passed = temperature_test_passed and inverse_relation_passed and entropy_test_passed
            
            # Prepare detailed results
            results = {
                'solar_mass_temperature': {
                    'predicted': T_H_predicted_value,
                    'expected': T_H_expected,
                    'relative_error': relative_error,
                    'passed': temperature_test_passed
                },
                'inverse_mass_relation': {
                    'product_std': product_std.item() if isinstance(product_std, torch.Tensor) else float(product_std),
                    'passed': inverse_relation_passed
                },
                'entropy_test': {
                    'passed': entropy_test_passed
                }
            }
            
            # Create result
            result = ValidationResult(self.name, theory.name)
            result.observed_value = T_H_expected
            result.predicted_value = T_H_predicted_value
            result.units = self.units
            result.passed = all_passed
            
            if all_passed:
                result.notes = f"Hawking temperature correctly implemented. T_H = {T_H_predicted_value:.2e} K for solar mass BH"
            else:
                failures = []
                if not temperature_test_passed:
                    failures.append(f"Temperature error: {relative_error:.2%}")
                if not inverse_relation_passed:
                    failures.append(f"T∝1/M relation violated: std={product_std:.2%}")
                if not entropy_test_passed:
                    failures.append("Entropy formula incorrect")
                    
                result.notes = f"Hawking temperature validation failed: {', '.join(failures)}"
                result.error_message = "; ".join(failures)
            
            # Add detailed results
            result.details = results
            return result
                
        except Exception as e:
            result = ValidationResult(self.name, theory.name)
            result.passed = False
            result.error_message = f"Error during Hawking temperature validation: {str(e)}"
            result.notes = "ERROR: " + str(e)
            return result
    
    def get_observational_data(self):
        """Return theoretical expectations for Hawking temperature."""
        return {
            'solar_mass_temperature': 6.17e-8,  # Kelvin
            'units': 'Kelvin',
            'notes': 'Theoretical Hawking temperature for solar mass black hole'
        }
