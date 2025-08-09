import torch
import numpy as np
from ..base_validation import ObservationalValidator, ValidationResult

class HubbleParameterValidator(ObservationalValidator):
    """
    Validates Hubble parameter and cosmological distance measures.
    
    Tests:
    1. Hubble constant H0 value
    2. Deceleration parameter q0
    3. Age of universe
    4. Distance-redshift relations
    """
    
    def __init__(self, engine=None):
        super().__init__(engine)
        self.name = "Hubble Parameter"
        self.description = "Validates cosmological expansion history and distance measures"
        self.units = "km/s/Mpc"
        
    def validate(self, theory, **kwargs) -> ValidationResult:
        """Validate cosmological expansion properties."""
        try:
            # Check if theory provides Hubble parameter
            if not hasattr(theory, 'compute_hubble_parameter'):
                return self.generate_fail_result(
                    notes="Theory does not implement compute_hubble_parameter method",
                    error_message="Missing required cosmology methods"
                )
            
            # Test 1: Hubble constant at z=0
            z_zero = torch.tensor(0.0)
            H0_predicted = theory.compute_hubble_parameter(z_zero)
            
            if isinstance(H0_predicted, torch.Tensor):
                H0_value = H0_predicted.item()
            else:
                H0_value = float(H0_predicted)
            
            # Observational constraints
            # Planck 2018: H0 = 67.4 ± 0.5 km/s/Mpc
            # SH0ES 2022: H0 = 73.0 ± 1.0 km/s/Mpc
            # We'll use a range that encompasses both
            H0_min = 65.0
            H0_max = 75.0
            H0_test_passed = H0_min <= H0_value <= H0_max
            
            # Test 2: Expansion history at various redshifts
            test_redshifts = torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0])
            H_values = []
            
            for z in test_redshifts:
                H_z = theory.compute_hubble_parameter(z)
                if isinstance(H_z, torch.Tensor):
                    H_values.append(H_z.item())
                else:
                    H_values.append(float(H_z))
            
            # Check that H(z) increases with redshift (for standard cosmology)
            expansion_test_passed = all(H_values[i] <= H_values[i+1] for i in range(len(H_values)-1))
            
            # Test 3: Age of universe
            age_test_passed = True
            age_value = None
            if hasattr(theory, 'compute_age_of_universe'):
                t_0 = theory.compute_age_of_universe()
                if isinstance(t_0, torch.Tensor):
                    age_value = t_0.item()
                else:
                    age_value = float(t_0)
                
                # Age should be 13-14 Gyr
                age_test_passed = 13.0 <= age_value <= 14.5
            
            # Test 4: Check if universe is accelerating today
            accel_test_passed = True
            is_accel = None
            if hasattr(theory, 'is_accelerating'):
                is_accel = theory.is_accelerating(z_zero)
                if isinstance(is_accel, torch.Tensor):
                    is_accel = bool(is_accel.item())
                else:
                    is_accel = bool(is_accel)
                
                # Modern universe should be accelerating (dark energy dominated)
                accel_test_passed = is_accel
            
            # Test 5: Luminosity distance (simplified test)
            distance_test_passed = True
            if hasattr(theory, 'compute_luminosity_distance'):
                # Test at z=1
                z_test = torch.tensor(1.0)
                d_L = theory.compute_luminosity_distance(z_test)
                if isinstance(d_L, torch.Tensor):
                    d_L_value = d_L.item()
                else:
                    d_L_value = float(d_L)
                
                # For ΛCDM with H0~70, Ωm~0.3, d_L(z=1) ~ 6500 Mpc
                # Allow broad range since we don't know exact parameters
                distance_test_passed = 4000 < d_L_value < 8000
            
            # Check cosmological parameters if available
            param_info = {}
            if hasattr(theory, 'Omega_m'):
                param_info['Omega_m'] = theory.Omega_m
            if hasattr(theory, 'Omega_Lambda'):
                param_info['Omega_Lambda'] = theory.Omega_Lambda
            if hasattr(theory, 'Omega_k'):
                param_info['Omega_k'] = theory.Omega_k
            
            # Overall pass/fail
            all_tests_passed = (H0_test_passed and expansion_test_passed and 
                               age_test_passed and accel_test_passed and distance_test_passed)
            
            results = {
                'H0': {
                    'value': H0_value,
                    'range': [H0_min, H0_max],
                    'passed': H0_test_passed
                },
                'expansion_history': {
                    'redshifts': test_redshifts.tolist(),
                    'H_values': H_values,
                    'monotonic': expansion_test_passed
                },
                'age_of_universe': {
                    'value': age_value,
                    'passed': age_test_passed
                },
                'acceleration': {
                    'is_accelerating': is_accel,
                    'passed': accel_test_passed
                },
                'parameters': param_info
            }
            
            if all_tests_passed:
                notes = f"Cosmological expansion correctly implemented. H0 = {H0_value:.1f} km/s/Mpc"
                if age_value:
                    notes += f", Age = {age_value:.1f} Gyr"
                    
                return self.generate_pass_result(
                    notes=notes,
                    predicted_value=H0_value,
                    sota_value=70.0,  # Approximate consensus value
                    details=results
                )
            else:
                failures = []
                if not H0_test_passed:
                    failures.append(f"H0 = {H0_value:.1f} outside range [{H0_min}, {H0_max}]")
                if not expansion_test_passed:
                    failures.append("Non-monotonic expansion history")
                if not age_test_passed:
                    failures.append(f"Age = {age_value:.1f} Gyr outside [13, 14.5]")
                if not accel_test_passed:
                    failures.append("Universe not accelerating today")
                if not distance_test_passed:
                    failures.append("Incorrect luminosity distance")
                    
                return self.generate_fail_result(
                    notes=f"Cosmological validation failed",
                    predicted_value=H0_value,
                    sota_value=70.0,
                    error_message="; ".join(failures),
                    details=results
                )
                
        except Exception as e:
            return self.generate_error_result(f"Error during Hubble parameter validation: {str(e)}")
    
    def get_observational_data(self):
        """Return observational constraints on cosmological parameters."""
        return {
            'H0_planck': 67.4,  # km/s/Mpc
            'H0_shoes': 73.0,   # km/s/Mpc
            'age_universe': 13.8,  # Gyr
            'Omega_m': 0.315,
            'Omega_Lambda': 0.685,
            'notes': 'Planck 2018 and SH0ES 2022 results'
        }
