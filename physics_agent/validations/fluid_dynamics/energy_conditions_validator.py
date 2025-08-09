import torch
import numpy as np
from ..base_validation import ObservationalValidator, ValidationResult
from physics_agent.constants import c

class EnergyConditionsValidator(ObservationalValidator):
    """
    Validates that fluid theories satisfy physical energy conditions.
    
    Tests:
    1. Null Energy Condition (NEC): ρ + p ≥ 0
    2. Weak Energy Condition (WEC): ρ ≥ 0 and ρ + p ≥ 0  
    3. Strong Energy Condition (SEC): ρ + 3p ≥ 0 and ρ + p ≥ 0
    4. Dominant Energy Condition (DEC): ρ ≥ |p|
    """
    
    def __init__(self, engine=None):
        super().__init__(engine)
        self.name = "Energy Conditions"
        self.description = "Validates that fluid stress-energy tensor satisfies energy conditions"
        self.units = "dimensionless"
        
    def validate(self, theory, **kwargs) -> ValidationResult:
        """Validate energy conditions for fluid theories."""
        try:
            # Check if theory provides stress-energy tensor
            if not hasattr(theory, 'get_fluid_stress_energy_tensor'):
                return self.generate_fail_result(
                    notes="Theory does not implement get_fluid_stress_energy_tensor method",
                    error_message="Missing required fluid dynamics methods"
                )
            
            # Test at various radii
            test_radii = torch.linspace(2.0, 20.0, 10)
            
            violations = {
                'null': 0,
                'weak': 0,
                'strong': 0,
                'dominant': 0
            }
            
            results_details = []
            
            for r in test_radii:
                # Get stress-energy tensor
                r_tensor = r.unsqueeze(0)  # Add batch dimension
                T = theory.get_fluid_stress_energy_tensor(r_tensor)
                
                # Extract energy density and pressure
                # T^0_0 = -ρ (energy density)
                # T^i_i = p (pressure, assuming isotropy)
                rho = -T[0, 0, 0]  # Energy density
                p = T[0, 1, 1]     # Radial pressure
                
                # Check if we have valid values
                if torch.isnan(rho) or torch.isnan(p):
                    continue
                    
                # Check energy conditions
                nec = (rho + p >= 0)
                wec = (rho >= 0) and (rho + p >= 0)
                sec = (rho + 3*p >= 0) and (rho + p >= 0)
                dec = (rho >= torch.abs(p))
                
                # Count violations
                if not nec:
                    violations['null'] += 1
                if not wec:
                    violations['weak'] += 1
                if not sec:
                    violations['strong'] += 1
                if not dec:
                    violations['dominant'] += 1
                    
                results_details.append({
                    'r': r.item(),
                    'rho': rho.item() if isinstance(rho, torch.Tensor) else float(rho),
                    'p': p.item() if isinstance(p, torch.Tensor) else float(p),
                    'nec': bool(nec),
                    'wec': bool(wec),
                    'sec': bool(sec),
                    'dec': bool(dec)
                })
            
            # Check sound speed if available
            sound_speed_valid = True
            if hasattr(theory, 'compute_sound_speed'):
                c_s = theory.compute_sound_speed()
                if isinstance(c_s, torch.Tensor):
                    c_s_value = c_s.item()
                else:
                    c_s_value = float(c_s)
                    
                # Sound speed should be subluminal: c_s ≤ c (in units where c=1)
                sound_speed_valid = 0 <= c_s_value <= 1
            
            # Calculate violation rates
            total_tests = len(test_radii)
            violation_rates = {
                cond: count / total_tests for cond, count in violations.items()
            }
            
            # Determine pass/fail
            # We allow some violations of strong/dominant conditions
            # but null and weak should always be satisfied
            critical_violations = violations['null'] > 0 or violations['weak'] > 0
            
            results = {
                'violation_rates': violation_rates,
                'violations': violations,
                'total_tests': total_tests,
                'sound_speed_valid': sound_speed_valid,
                'details': results_details
            }
            
            if not critical_violations and sound_speed_valid:
                notes = "All critical energy conditions satisfied"
                if violations['strong'] > 0:
                    notes += f" (SEC violated at {violation_rates['strong']:.0%} of points)"
                if violations['dominant'] > 0:
                    notes += f" (DEC violated at {violation_rates['dominant']:.0%} of points)"
                    
                return self.generate_pass_result(
                    notes=notes,
                    predicted_value=0.0,  # No critical violations
                    sota_value=0.0,
                    details=results
                )
            else:
                failure_reasons = []
                if violations['null'] > 0:
                    failure_reasons.append(f"NEC violated at {violation_rates['null']:.0%} of points")
                if violations['weak'] > 0:
                    failure_reasons.append(f"WEC violated at {violation_rates['weak']:.0%} of points")
                if not sound_speed_valid:
                    failure_reasons.append(f"Superluminal sound speed: c_s = {c_s_value:.2f}")
                    
                return self.generate_fail_result(
                    notes=f"Energy condition violations: {'; '.join(failure_reasons)}",
                    predicted_value=violations['null'] + violations['weak'],  # Critical violations
                    sota_value=0.0,
                    error_message="; ".join(failure_reasons),
                    details=results
                )
                
        except Exception as e:
            return self.generate_error_result(f"Error during energy conditions validation: {str(e)}")
    
    def get_observational_data(self):
        """Return expected values for energy conditions."""
        return {
            'null_energy_condition': 'ρ + p ≥ 0',
            'weak_energy_condition': 'ρ ≥ 0 and ρ + p ≥ 0',
            'strong_energy_condition': 'ρ + 3p ≥ 0 and ρ + p ≥ 0',
            'dominant_energy_condition': 'ρ ≥ |p|',
            'notes': 'Physical fluids should satisfy at least NEC and WEC'
        }
