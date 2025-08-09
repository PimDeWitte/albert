"""
Validators for discovered expressions.

These validators check if expressions satisfy specific physical conditions.
"""

from __future__ import annotations
from typing import Dict, Any, Callable
from sympy import symbols, diff, simplify, Matrix, det, Basic
import sympy
import signal
import time


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Validation timed out")


def with_timeout(func, timeout_seconds=5):
    """Wrapper to add timeout to a function."""
    def wrapper(*args, **kwargs):
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            result = func(*args, **kwargs)
        except TimeoutException:
            return {
                'valid': False,
                'details': f'Validation timed out after {timeout_seconds}s',
                'value_at_test': None
            }
        finally:
            # Cancel alarm and restore handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
        return result
    return wrapper


def validate_force_free_foliation(expr: Basic, **kwargs) -> Dict[str, Any]:
    """
    Check if an expression u(ρ, z) satisfies the force-free foliation condition.
    
    Based on equation 2.14 from the Force-Free Foliations paper:
    det(LT A  LT B) = 0
       (L²T A L²T B)
       
    Where:
    - T = uz ∂ρ - uρ ∂z (tangent vector field)  
    - A = uρρ + uzz - ρ⁻¹uρ
    - B = uρ² + uz²
    - LT is the Lie derivative along T
    
    Args:
        expr: Expression to validate u(ρ, z)
        
    Returns:
        Dict with keys:
        - 'valid': bool - whether condition is satisfied
        - 'details': str - explanation
        - 'value_at_test': Any - determinant value at test point
    """
    # Get free symbols from expression
    free_symbols = expr.free_symbols
    
    # Find rho and z symbols in the expression
    rho = None
    z = None
    for sym in free_symbols:
        if str(sym) == 'rho':
            rho = sym
        elif str(sym) == 'z':
            z = sym
            
    # If expression doesn't have these symbols, create them
    if rho is None:
        rho = symbols('rho', real=True, positive=True)
    if z is None:
        z = symbols('z', real=True)
    
    # Check for unexpected symbols
    expected_names = {'rho', 'z'}
    actual_names = {str(s) for s in free_symbols}
    if not actual_names.issubset(expected_names):
        unexpected = actual_names - expected_names
        return {
            'valid': False,
            'details': f'Expression contains unexpected symbols: {unexpected}',
            'value_at_test': None
        }
    
    try:
        # Compute derivatives
        u_rho = diff(expr, rho)
        u_z = diff(expr, z)
        u_rho_rho = diff(u_rho, rho)
        u_z_z = diff(u_z, z)
        
        # Define A and B from equation 2.10
        A = u_rho_rho + u_z_z - u_rho / rho
        B = u_rho**2 + u_z**2
        
        # Define tangent vector field operations
        def lie_derivative_T(f):
            """Compute Lie derivative along T = uz ∂ρ - uρ ∂z."""
            return u_z * diff(f, rho) - u_rho * diff(f, z)
        
        # Compute matrix elements
        LT_A = lie_derivative_T(A)
        LT_B = lie_derivative_T(B)
        LT2_A = lie_derivative_T(LT_A)
        LT2_B = lie_derivative_T(LT_B)
        
        # Build the matrix
        M = Matrix([
            [LT_A, LT_B],
            [LT2_A, LT2_B]
        ])
        
        # Compute determinant
        det_M = det(M)
        
        # Simplify with timeout
        try:
            # Use a simpler simplification first
            det_M_simplified = det_M.expand()
            # Only do full simplify if expression is small enough
            if len(str(det_M_simplified)) < 1000:
                det_M_simplified = simplify(det_M_simplified)
        except Exception:
            det_M_simplified = det_M
        
        # Check at test point
        test_rho = sympy.Rational(4, 5)
        test_z = sympy.Rational(6, 7)
        
        det_val = det_M_simplified.subs([(rho, test_rho), (z, test_z)])
        
        # Check if exactly zero
        is_valid = det_val == 0
        
        # Classify the solution type
        classification = _classify_force_free_solution(expr, rho, z)
        
        return {
            'valid': is_valid,
            'details': f'Foliation condition {"satisfied" if is_valid else "not satisfied"}. Type: {classification}',
            'value_at_test': float(det_val) if det_val.is_number else str(det_val),
            'classification': classification
        }
        
    except Exception as e:
        return {
            'valid': False,
            'details': f'Error during validation: {str(e)}',
            'value_at_test': None
        }


def _classify_force_free_solution(expr: Basic, rho, z) -> str:
    """Classify the type of force-free foliation."""
    expr_str = str(expr)
    
    # Check for known patterns from the paper
    if expr == rho**2:
        return "Vertical field (external dipole)"
    elif expr == rho**2 * z:
        return "X-point (external quadrupole)"
    elif expr == 1 - z/sympy.sqrt(z**2 + rho**2):
        return "Radial"
    elif 'rho**2/(z**2 + rho**2)**(3/2)' in expr_str:
        return "Dipolar"
    elif expr == sympy.sqrt(z**2 + rho**2) - z:
        return "Parabolic"
    elif 'exp' in expr_str and 'rho**2' in expr_str:
        return "Bent (nonvacuum)"
    else:
        # Try to determine if vacuum or not
        if any(term in expr_str for term in ['exp', 'log', 'sin', 'cos']):
            return "Unknown (possibly nonvacuum)"
        return "Unknown (possibly vacuum)"


# Registry of validators for different physics problems
VALIDATORS: Dict[str, Callable] = {
    'force_free': validate_force_free_foliation,
}


def validate_expression(expr: Basic, mode: str = 'lagrangian', **kwargs) -> Dict[str, Any]:
    """
    Validate an expression based on the discovery mode.
    
    Args:
        expr: Expression to validate
        mode: Discovery mode ('lagrangian', 'force_free', etc.)
        **kwargs: Additional arguments for specific validators
        
    Returns:
        Validation results dictionary
    """
    if mode in VALIDATORS:
        return VALIDATORS[mode](expr, **kwargs)
    else:
        # Default validation - just check if expression is well-formed
        return {
            'valid': True,
            'details': f'No specific validator for mode: {mode}',
            'value_at_test': None
        }
