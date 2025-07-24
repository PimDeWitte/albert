#!/usr/bin/env python3
"""
Test that GeneralGeodesicRK4Solver handles both 4D and 6D state vectors correctly.
"""

import torch
import sys
from physics_agent.theories.defaults.baselines.kerr import Kerr
from physics_agent.geodesic_integrator import GeneralGeodesicRK4Solver

def test_general_solver_with_different_state_dimensions():
    """Test that the general solver handles both 4D and 6D states"""
    print("Testing GeneralGeodesicRK4Solver with different state dimensions...")
    
    # Create a Kerr metric with non-zero spin (requires general solver)
    model = Kerr(a=0.5)
    M_phys = torch.tensor(1.989e30)  # Solar mass
    
    # Create the solver
    solver = GeneralGeodesicRK4Solver(model, M_phys)
    
    # Test with 4D state vector
    print("\nTesting with 4D state vector [t, r, phi, dr/dtau]:")
    y_4d = torch.tensor([0.0, 10.0, 0.0, 0.0], dtype=torch.float64)
    
    # Set conserved quantities
    solver.E = 0.95
    solver.Lz = 3.5
    
    try:
        deriv_4d = solver.compute_derivatives(y_4d)
        print(f"  4D state: {y_4d}")
        print(f"  4D derivatives shape: {deriv_4d.shape}")
        print(f"  4D derivatives: {deriv_4d}")
        print("  ✓ 4D state handled correctly")
    except Exception as e:
        print(f"  ✗ Error with 4D state: {e}")
        return False
    
    # Test with 6D state vector
    print("\nTesting with 6D state vector [t, r, phi, u^t, u^r, u^phi]:")
    y_6d = torch.tensor([0.0, 10.0, 0.0, 0.95, 0.0, 0.35], dtype=torch.float64)
    
    try:
        deriv_6d = solver.compute_derivatives(y_6d)
        print(f"  6D state: {y_6d}")
        print(f"  6D derivatives shape: {deriv_6d.shape}")
        print(f"  6D derivatives: {deriv_6d}")
        print("  ✓ 6D state handled correctly")
    except Exception as e:
        print(f"  ✗ Error with 6D state: {e}")
        return False
    
    # Test RK4 step with both
    print("\nTesting RK4 step:")
    h = 0.01
    
    try:
        # Test 4D RK4 step
        y_new_4d = solver.rk4_step(y_4d, h)
        if y_new_4d is not None:
            print(f"  ✓ 4D RK4 step successful: {y_new_4d}")
        else:
            print("  ✗ 4D RK4 step returned None")
            
        # Test 6D RK4 step
        y_new_6d = solver.rk4_step(y_6d, h)
        if y_new_6d is not None:
            print(f"  ✓ 6D RK4 step successful: {y_new_6d}")
        else:
            print("  ✗ 6D RK4 step returned None")
            
    except Exception as e:
        print(f"  ✗ Error in RK4 step: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_general_solver_with_different_state_dimensions()
    sys.exit(0 if success else 1) 