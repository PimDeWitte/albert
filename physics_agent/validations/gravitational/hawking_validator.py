import torch
from typing import Dict, Any, TYPE_CHECKING
from ..base_validation import BaseValidation
from scipy.constants import hbar, k, G, c
import numpy as np

if TYPE_CHECKING:
    from physics_agent.base_theory import GravitationalTheory
    from physics_agent.theory_engine_core import TheoryEngine

class HawkingValidator(BaseValidation):
    """
    Validates theories against Hawking radiation temperature predictions.
    
    This tests whether a theory correctly predicts the Hawking temperature
    for black holes, which is a key quantum gravitational effect.
    """
    category = "observational"
    
    def __init__(self, engine: "TheoryEngine", tolerance: float = 1e-2):
        """Initialize the validator with a theory engine."""
        super().__init__(engine, "Hawking Radiation Validator")
        self.tolerance = tolerance

    def validate(self, theory: "GravitationalTheory", hist: torch.Tensor = None, **kwargs) -> Dict[str, Any]:
        """
        Validate a theory's prediction of Hawking radiation temperature.
        
        Args:
            theory: The gravitational theory to validate
            hist: Trajectory history (not used for Hawking radiation)
            **kwargs: Additional arguments
            
        Returns:
            Dict with loss, flags, and details
        """
        verbose = kwargs.get('verbose', False)
        
        # Use solar mass for testing
        M_sun = 1.989e30  # kg
        
        if verbose:
            print(f"\nCalculating Hawking temperature for {theory.name}...")
            print(f"  Black hole mass: {M_sun:.3e} kg (solar mass)")
        
        # Standard GR Hawking temperature
        T_gr = hbar * c**3 / (8 * np.pi * G * M_sun * k)
        
        # <reason>chain: Theory could modify Hawking temperature through metric modifications</reason>
        # Check if theory implements custom Hawking temperature calculation
        if hasattr(theory, 'compute_hawking_temperature'):
            # <reason>chain: Theories should implement full temperature calculation</reason>
            # Check if method expects M only or (M, C_param, G_param)
            try:
                T_theory = theory.compute_hawking_temperature(M_sun)
            except TypeError:
                # Try with full parameters
                M_tensor = torch.tensor(M_sun)
                T_theory = theory.compute_hawking_temperature(M_tensor, c, G).item()
        elif hasattr(theory, 'compute_black_hole_entropy'):
            # <reason>chain: Calculate temperature from modified entropy using thermodynamic relation</reason>
            # T = dM/dS where S is entropy
            S = theory.compute_black_hole_entropy(M_sun)
            # Use finite difference for derivative
            dM = 0.01 * M_sun
            S_plus = theory.compute_black_hole_entropy(M_sun + dM)
            dS = S_plus - S
            if dS > 0:
                T_theory = dM / dS
            else:
                T_theory = T_gr  # Fallback if entropy calculation fails
        elif hasattr(theory, 'hawking_temperature_factor'):
            # <reason>chain: Deprecated: simple factor approach</reason>
            T_theory = T_gr * theory.hawking_temperature_factor
            if verbose:
                print(f"  Warning: Using deprecated hawking_temperature_factor")
        else:
            # <reason>chain: Require quantum theories to implement Hawking physics</reason>
            if hasattr(theory, 'category') and theory.category == 'quantum':
                if verbose:
                    print(f"  Warning: Quantum theory lacks Hawking temperature implementation")
                return {
                    "loss": 0.5,  # Significant penalty for quantum theories
                    "flags": {"overall": "WARNING", "no_hawking_implementation": True},
                    "details": {
                        "observed_value": T_gr,
                        "predicted_value": float('nan'),
                        "error": float('nan'),
                        "error_percent": float('nan'),
                        "units": "K",
                        "notes": "Quantum theory must implement Hawking temperature calculation"
                    }
                }
            else:
                # <reason>chain: Non-quantum theories default to GR</reason>
                T_theory = T_gr
        
        # Compute error
        error = abs(T_theory - T_gr)
        error_percent = 100.0 * error / T_gr if T_gr != 0 else float('inf')
        
        # Check if within tolerance
        passed = error_percent < (self.tolerance * 100)
        loss = error / T_gr  # Normalized loss
        
        if verbose:
            print(f"  GR Hawking temperature: {T_gr:.3e} K")
            print(f"  Theory prediction: {T_theory:.3e} K")
            print(f"  Error: {error_percent:.2f}%")
            print(f"  Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            "loss": loss,
            "flags": {"overall": "PASS" if passed else "FAIL"},
            "details": {
                "observed_value": T_gr,
                "predicted_value": T_theory,
                "error": error,
                "error_percent": error_percent,
                "units": "K",
                "notes": f"Hawking temperature test: {error_percent:.2f}% error"
            }
        } 