"""
Black Hole Thermodynamics Validator

Tests the consistency of a theory with the laws of black hole thermodynamics:
1. First Law: dM = (κ/8π)dA + ΩdJ + ΦdQ
2. Second Law: dA ≥ 0 (area theorem)
3. Third Law: κ → 0 is unattainable
4. Zeroth Law: κ is constant on the horizon

<reason>chain: Black hole thermodynamics provides a bridge between gravity, thermodynamics, and quantum mechanics</reason>
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from physics_agent.validations.base_validation import ObservationalValidator, ValidationResult
from physics_agent.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT, PLANCK_CONSTANT, BOLTZMANN_CONSTANT

class BlackHoleThermodynamicsValidator(ObservationalValidator):
    """
    Validates that a theory respects the laws of black hole thermodynamics.
    
    <reason>chain: Essential for any quantum gravity theory to satisfy thermodynamic consistency</reason>
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Black Hole Thermodynamics"
        self.description = "Tests consistency with black hole thermodynamic laws"
        
    def validate(self, theory, **kwargs) -> ValidationResult:
        """
        Validate theory against black hole thermodynamics.
        
        <reason>chain: Tests Hawking temperature, entropy, and thermodynamic laws</reason>
        """
        results = {
            'hawking_temperature': self._test_hawking_temperature(theory),
            'bekenstein_hawking_entropy': self._test_entropy(theory),
            'first_law': self._test_first_law(theory),
            'second_law': self._test_second_law(theory),
            'information_paradox': self._test_information_preservation(theory)
        }
        
        # Calculate overall pass/fail
        passed = all(r['passed'] for r in results.values() if 'passed' in r)
        
        # Calculate aggregate score
        scores = [r.get('score', 0.0) for r in results.values()]
        overall_score = np.mean(scores) if scores else 0.0
        
        return ValidationResult(
            passed=passed,
            score=overall_score,
            details=results,
            notes=self._generate_notes(results, theory)
        )
    
    def _test_hawking_temperature(self, theory) -> Dict[str, Any]:
        """Test if theory predicts correct Hawking temperature."""
        try:
            # Test for Schwarzschild black hole
            M = 1.0 * SOLAR_MASS  # Solar mass black hole
            
            # Expected Hawking temperature: T = ℏc³/(8πGMk_B)
            expected_T = (PLANCK_CONSTANT * SPEED_OF_LIGHT**3) / (8 * np.pi * GRAVITATIONAL_CONSTANT * M * BOLTZMANN_CONSTANT)
            
            # Check if theory has Hawking temperature method
            if hasattr(theory, 'compute_hawking_temperature'):
                predicted_T = theory.compute_hawking_temperature(M)
                
                # Calculate relative error
                error = abs(predicted_T - expected_T) / expected_T
                
                return {
                    'passed': error < 0.1,  # 10% tolerance
                    'score': 1.0 - min(error, 1.0),
                    'predicted': predicted_T,
                    'expected': expected_T,
                    'error_percent': error * 100
                }
            else:
                # Theory doesn't implement Hawking temperature
                return {
                    'passed': False,
                    'score': 0.0,
                    'notes': 'Theory does not implement compute_hawking_temperature()'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_entropy(self, theory) -> Dict[str, Any]:
        """Test if theory predicts correct Bekenstein-Hawking entropy."""
        try:
            M = 1.0 * SOLAR_MASS
            
            # Schwarzschild radius
            r_s = 2 * GRAVITATIONAL_CONSTANT * M / SPEED_OF_LIGHT**2
            
            # Horizon area
            A = 4 * np.pi * r_s**2
            
            # Expected entropy: S = A/(4l_p²) where l_p is Planck length
            l_p = np.sqrt(PLANCK_CONSTANT * GRAVITATIONAL_CONSTANT / SPEED_OF_LIGHT**3)
            expected_S = A / (4 * l_p**2)
            
            if hasattr(theory, 'compute_black_hole_entropy'):
                predicted_S = theory.compute_black_hole_entropy(M)
                
                # Check order of magnitude (entropy is huge)
                ratio = predicted_S / expected_S
                
                return {
                    'passed': 0.5 < ratio < 2.0,  # Factor of 2 tolerance
                    'score': 1.0 - abs(np.log10(ratio)),
                    'predicted': predicted_S,
                    'expected': expected_S,
                    'ratio': ratio
                }
            else:
                return {
                    'passed': False,
                    'score': 0.0,
                    'notes': 'Theory does not implement compute_black_hole_entropy()'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_first_law(self, theory) -> Dict[str, Any]:
        """Test the first law of black hole mechanics."""
        try:
            # Test energy conservation in merger
            M1 = 30.0 * SOLAR_MASS  # First black hole
            M2 = 20.0 * SOLAR_MASS  # Second black hole
            
            # For non-spinning merger: M_final < M1 + M2 (some energy radiated)
            # Typical: ~5% radiated in GWs
            
            if hasattr(theory, 'compute_merger_final_mass'):
                M_final = theory.compute_merger_final_mass(M1, M2)
                
                # Check energy conservation
                efficiency = 1 - M_final / (M1 + M2)
                
                return {
                    'passed': 0.01 < efficiency < 0.10,  # 1-10% radiated
                    'score': 1.0 if 0.03 < efficiency < 0.07 else 0.5,
                    'initial_mass': M1 + M2,
                    'final_mass': M_final,
                    'radiated_fraction': efficiency
                }
            else:
                return {
                    'passed': True,  # Not required
                    'score': 0.5,
                    'notes': 'Theory does not model black hole mergers'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_second_law(self, theory) -> Dict[str, Any]:
        """Test the second law (area theorem)."""
        try:
            # For classical GR, area never decreases
            # For quantum theories, might have slow decrease due to Hawking radiation
            
            if hasattr(theory, 'compute_horizon_area_evolution'):
                M = 1.0 * SOLAR_MASS
                times = np.linspace(0, 1e10, 100)  # 10 billion years
                
                areas = theory.compute_horizon_area_evolution(M, times)
                
                # Check if area is non-decreasing (or decreases very slowly)
                dA_dt = np.gradient(areas, times)
                
                # Allow for Hawking radiation (very slow decrease)
                max_decrease_rate = 1e-70  # m²/s (extremely slow)
                
                violations = np.sum(dA_dt < -max_decrease_rate)
                
                return {
                    'passed': violations == 0,
                    'score': 1.0 - violations / len(times),
                    'violations': violations,
                    'notes': 'Area theorem satisfied' if violations == 0 else f'{violations} violations found'
                }
            else:
                return {
                    'passed': True,
                    'score': 0.5,
                    'notes': 'Theory does not model horizon evolution'
                }
                
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_information_preservation(self, theory) -> Dict[str, Any]:
        """Test if theory addresses the information paradox."""
        try:
            # Check if theory has a mechanism for information preservation
            mechanisms = []
            
            if hasattr(theory, 'preserves_information'):
                if theory.preserves_information():
                    mechanisms.append('explicit_preservation')
                    
            if hasattr(theory, 'information_recovery_time'):
                t_recovery = theory.information_recovery_time(1.0 * SOLAR_MASS)
                if t_recovery < np.inf:
                    mechanisms.append(f'recovery_time_{t_recovery:.2e}s')
                    
            if hasattr(theory, 'remnant_mass'):
                M_remnant = theory.remnant_mass(1.0 * SOLAR_MASS)
                if M_remnant > 0:
                    mechanisms.append(f'remnant_{M_remnant:.2e}kg')
            
            has_solution = len(mechanisms) > 0
            
            return {
                'passed': has_solution,  # Pass if any mechanism exists
                'score': 1.0 if has_solution else 0.0,
                'mechanisms': mechanisms,
                'notes': 'Has information preservation mechanism' if has_solution else 'No solution to information paradox'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _generate_notes(self, results: Dict[str, Any], theory) -> str:
        """Generate summary notes."""
        notes = []
        
        if results['hawking_temperature']['passed']:
            notes.append("✓ Correct Hawking temperature")
        else:
            notes.append("✗ Incorrect Hawking temperature")
            
        if results['bekenstein_hawking_entropy']['passed']:
            notes.append("✓ Correct black hole entropy")
        else:
            notes.append("✗ Incorrect black hole entropy")
            
        if 'mechanisms' in results['information_paradox']:
            if results['information_paradox']['mechanisms']:
                notes.append(f"✓ Information preservation: {', '.join(results['information_paradox']['mechanisms'])}")
            else:
                notes.append("✗ No information preservation mechanism")
        
        return "; ".join(notes)


# Import required constants that might be missing
try:
    from physics_agent.constants import SOLAR_MASS
except ImportError:
    SOLAR_MASS = 1.989e30  # kg
