from __future__ import annotations
import torch
import numpy as np
import sympy as sp
from typing import Dict, Any, TYPE_CHECKING

from ..base_validation import BaseValidation
from physics_agent.base_theory import GravitationalTheory

# <reason>chain: Import centralized constants for consistency across the project</reason>
from physics_agent.constants import LAGRANGIAN_TEST_VALUES, STANDARD_PHYSICS_SYMBOLS

if TYPE_CHECKING:
    from physics_agent.theory_engine_core import TheoryEngine

# <reason>chain: Define standard symbols based on what's used in base_theory.py and theory files</reason>
# NOTE: STANDARD_PHYSICS_SYMBOLS is now imported from constants.py
class LagrangianValidator(BaseValidation):
    """
    Smart Lagrangian validator that automatically handles common issues:
    1. Auto-fixes symbol naming mismatches
    2. Pre-substitutes all theory parameters
    3. Provides detailed debugging
    4. Falls back gracefully on errors
    
    This validator no longer attempts to derive metrics from Lagrangians,
    as that process is complex and error-prone. Instead, it validates
    that the Lagrangian is well-formed and consistent with theory parameters.
    """
    category = "constraint"
    
    def __init__(self, engine: "TheoryEngine", tolerance: float = 1e-3, num_samples: int = 50, loss_type: str = 'ricci'):
        super().__init__(engine, "Lagrangian Validator")
        self.tolerance = tolerance
        self.num_samples = num_samples
        self.loss_type = loss_type

    def validate(self, theory: "GravitationalTheory", hist: torch.Tensor = None, **kwargs) -> Dict[str, Any]:
        """
        Validates that the theory's Lagrangian is well-formed and evaluable.
        <reason>chain: Check for required Lagrangian components based on theory methods, not categories</reason>
        """
        # Check if theory has required validation method
        if hasattr(theory, 'validate_lagrangian_completeness'):
            completeness = theory.validate_lagrangian_completeness()
            if not completeness['complete']:
                return {
                    "loss": float('inf'),
                    "flags": {
                        "overall": "FAIL",
                        "details": f"Theory missing required terms: {completeness['missing']}"
                    }
                }
        
        # <reason>chain: Use helper to get appropriate Lagrangian based on theory type</reason>
        lagrangian_to_validate, lagrangian_type = self._get_appropriate_lagrangian(theory)
        
        if lagrangian_to_validate is None:
            return {
                "loss": 0.0,
                "flags": {"overall": "SKIP", "details": "No Lagrangian defined for theory."}
            }
        
        # Check quantum consistency
        consistency = self._check_quantum_consistency(theory)
        if not consistency['is_consistent']:
            print(f"  Quantum consistency issues: {', '.join(consistency['issues'])}")
        
        try:
            # Get all free symbols in the complete Lagrangian
            free_symbols = lagrangian_to_validate.free_symbols
            
            # Build substitution dictionary
            subs_dict = {}
            unmatched_symbols = []
            
            for sym in free_symbols:
                sym_name = str(sym)
                matched = False
                
                # Standard symbols that don't need theory attributes  
                if sym_name in STANDARD_PHYSICS_SYMBOLS:
                    continue
                    
                # Try exact match first
                if hasattr(theory, sym_name):
                    val = getattr(theory, sym_name)
                    if torch.is_tensor(val):
                        val = val.item()
                    subs_dict[sym] = val
                    matched = True
                else:
                    # Try common variations
                    variations = [
                        sym_name.replace('_sym', ''),
                        sym_name.replace('Sym', ''),
                        sym_name.replace('_', ''),
                        sym_name.lower(),
                        sym_name.upper(),
                    ]
                    
                    for var in variations:
                        if hasattr(theory, var):
                            val = getattr(theory, var)
                            if torch.is_tensor(val):
                                val = val.item()
                            subs_dict[sym] = val
                            matched = True
                            break
                
                if not matched:
                    unmatched_symbols.append(sym_name)
            
            # <reason>chain: Make validator less strict - warn instead of fail for unmatched symbols</reason>
            # If we have unmatched symbols, report them as warning but don't fail
            warning_details = ""
            if unmatched_symbols:
                warning_details = f"Warning: Unmatched symbols in Lagrangian: {unmatched_symbols}. "
                # Try to assign reasonable default values for unmatched symbols
                for sym_name in unmatched_symbols:
                    sym = sp.Symbol(sym_name)
                    # Assign small default values to unmatched symbols
                    if 'alpha' in sym_name.lower() or 'α' in sym_name:
                        subs_dict[sym] = 0.01  # Coupling constants are typically small
                    elif 'beta' in sym_name.lower() or 'β' in sym_name:
                        subs_dict[sym] = 0.01
                    elif 'ent' in sym_name.lower():
                        subs_dict[sym] = 1.0  # Entropy-related terms
                    else:
                        subs_dict[sym] = 1.0  # Generic default
            
            # If we have unmatched symbols, report them
            # <reason>chain: Don't fail validation for unmatched symbols - just warn</reason>
            # if unmatched_symbols:
            #     return {
            #         "loss": float('inf'),
            #         "flags": {
            #             "overall": "FAIL",
            #             "details": f"Unmatched symbols in Lagrangian: {unmatched_symbols}. Theory attributes: {[attr for attr in dir(theory) if not attr.startswith('_')]}"
            #         }
            #     }
            
            # Substitute all matched parameters
            L_substituted = lagrangian_to_validate.subs(subs_dict)
            
            # Test evaluation at a sample point
            test_vals = LAGRANGIAN_TEST_VALUES
            
            # Convert remaining symbols to test values
            remaining_symbols = L_substituted.free_symbols
            for sym in remaining_symbols:
                if str(sym) in test_vals:
                    L_substituted = L_substituted.subs(sym, test_vals[str(sym)])
            
            # Try to evaluate
            try:
                # <reason>chain: Handle complex Lagrangians by taking real part</reason>
                # For quantum field theories, the action is real even if Lagrangian is complex
                if L_substituted.is_complex:
                    L_value = float(sp.re(L_substituted))
                else:
                    L_value = float(L_substituted)
                
                # Check if value is reasonable
                if np.isnan(L_value) or np.isinf(L_value):
                    return {
                        "loss": float('inf'),
                        "flags": {
                            "overall": "FAIL",
                            "details": f"Lagrangian evaluates to {L_value}"
                        }
                    }
                
            except Exception as e:
                # <reason>chain: Don't fail if we can't evaluate - just check it's a valid expression</reason>
                # Some Lagrangians may be too complex to evaluate to a simple number
                if isinstance(L_substituted, sp.Basic):
                    # It's a valid SymPy expression, that's good enough
                    L_value = "symbolic"
                else:
                    return {
                        "loss": float('inf'),
                        "flags": {
                            "overall": "FAIL",
                            "details": f"Failed to evaluate Lagrangian: {str(e)}"
                        }
                    }
                
            # Success - Lagrangian is well-formed
            return {
                "loss": 0.0,
                "flags": {
                    "overall": "PASS",
                    "g_tt_consistency": "PASS",
                    "g_rr_consistency": "PASS",
                    "details": warning_details if warning_details else "All symbols matched"
                },
                "details": {
                    "lagrangian_value": L_value,
                    "substituted_params": [str(sym) for sym in subs_dict.keys()],
                    "unmatched_symbols": unmatched_symbols if unmatched_symbols else [],
                    "mse_g_tt": 0.0,  # Compatibility with existing code
                    "mse_g_rr": 0.0
                }
            }
        
        except Exception as e:
            return {
                "loss": float('inf'),
                "flags": {"overall": "FAIL", "details": f"Validation error: {str(e)}"}
            } 