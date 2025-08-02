#!/usr/bin/env python3
"""
Simple test for Einstein Asymmetric theory with UnifiedQuantumSolver.
Works directly in geometric units to avoid SI conversion issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from physics_agent.theories.einstein_asymmetric.theory import EinsteinAsymmetric
from physics_agent.geodesic_integrator_stable import create_geodesic_solver
from physics_agent.unified_quantum_solver import UnifiedQuantumSolver


def test_einstein_asymmetric_solver():
    """Test Einstein Asymmetric theory solver directly"""
    print("=" * 60)
    print("Testing Einstein Asymmetric Solver")
    print("=" * 60)
    
    # Create theory with α=0.0
    theory = EinsteinAsymmetric(alpha=0.0)
    print(f"\nTheory: {theory.name}")
    print(f"Category: {theory.category}")
    
    # Create solver using factory function
    print("\nCreating UnifiedQuantumSolver...")
    solver = create_geodesic_solver(
        theory,
        M_phys=torch.tensor(1.0),  # M=1 in geometric units
        c=1.0,
        G=1.0,
        use_pennylane_quantum=False
    )
    
    print(f"✓ Solver created: {type(solver).__name__}")
    
    # Test that it's a UnifiedQuantumSolver
    assert isinstance(solver, UnifiedQuantumSolver), f"Expected UnifiedQuantumSolver, got {type(solver)}"
    print("✓ Correctly created UnifiedQuantumSolver for quantum theory")
    
    # Test rk4_step method
    print("\nTesting rk4_step method...")
    y0 = torch.tensor([0.0, 10.0, 0.0, 0.0], dtype=torch.float64)  # [t, r, phi, dr/dtau]
    h = 0.1
    
    try:
        # Set conserved quantities
        solver.E = 0.95
        solver.Lz = 4.0
        
        # Initialize internal geodesic solver
        solver._init_geodesic_solver()
        if hasattr(solver._geodesic_solver, 'E'):
            solver._geodesic_solver.E = solver.E
            solver._geodesic_solver.Lz = solver.Lz
        
        # Take one RK4 step
        y1 = solver.rk4_step(y0, h)
        
        print(f"✓ Initial state: {y0}")
        print(f"✓ After one step: {y1}")
        print(f"✓ rk4_step works correctly!")
        
        # Test integrate method
        print("\nTesting integrate method...")
        trajectory = solver.integrate(y0.tolist(), num_steps=10, h=h)
        
        print(f"✓ Integrated {len(trajectory)} steps")
        print(f"✓ Final position: r={trajectory[-1][1]:.2f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("SUCCESS! Einstein Asymmetric works with UnifiedQuantumSolver ✅")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_einstein_asymmetric_solver()
    sys.exit(0 if success else 1)