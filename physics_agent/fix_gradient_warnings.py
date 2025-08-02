#!/usr/bin/env python3
"""Fix the gradient calculation warnings in trajectory visualization."""

import numpy as np

def safe_gradient(y, x):
    """
    Compute gradient with proper handling of edge cases.
    
    This avoids divide-by-zero warnings when time steps are identical.
    """
    if len(y) < 2:
        return np.zeros_like(y)
    
    # Use numpy's gradient but handle edge cases
    dx = np.diff(x)
    
    # Check for zero or near-zero time differences
    min_dt = np.min(np.abs(dx[dx != 0])) if np.any(dx != 0) else 1e-20
    dx[dx == 0] = min_dt  # Replace zeros with small value
    
    # Manual gradient calculation to avoid warnings
    grad = np.zeros_like(y)
    
    # Forward difference for first point
    if len(y) > 1:
        grad[0] = (y[1] - y[0]) / dx[0]
    
    # Central differences for interior points
    for i in range(1, len(y) - 1):
        grad[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    
    # Backward difference for last point
    if len(y) > 1:
        grad[-1] = (y[-1] - y[-2]) / dx[-1]
    
    return grad

# Test the function
if __name__ == "__main__":
    # Test case that would cause warnings
    t = np.array([0, 1e-20, 1e-20, 2e-20, 3e-20])  # Some identical time steps
    r = np.array([1.0, 1.0, 1.0, 1.01, 1.02])
    
    print("Testing safe gradient calculation...")
    print(f"Time array: {t}")
    print(f"Position array: {r}")
    
    # This would cause warnings
    print("\nUsing numpy.gradient (may show warnings):")
    try:
        grad_numpy = np.gradient(r, t)
        print(f"Result: {grad_numpy}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Our safe version
    print("\nUsing safe_gradient (no warnings):")
    grad_safe = safe_gradient(r, t)
    print(f"Result: {grad_safe}")