#!/usr/bin/env python3
"""
Test script specifically for Einstein Asymmetric theory with UnifiedQuantumSolver.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from physics_agent.theories.einstein_asymmetric.theory import EinsteinAsymmetric
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.constants import SOLAR_MASS


def test_einstein_asymmetric():
    """Test Einstein Asymmetric theory with the UnifiedQuantumSolver"""
    print("=" * 60)
    print("Testing Einstein Asymmetric Theory")
    print("=" * 60)
    
    # Create theory with α=0.0 (as in the error)
    theory = EinsteinAsymmetric(alpha=0.0)
    print(f"\nTheory: {theory.name}")
    print(f"Category: {theory.category}")
    print(f"Alpha parameter: {theory.alpha}")
    
    # Create engine
    engine = TheoryEngine(
        device='cpu',
        dtype=torch.float64,
        quantum_field_content='all'
    )
    
    # Test validation trajectory
    print("\nRunning validation trajectory...")
    try:
        # Run a short trajectory
        # Note: run_trajectory expects r0_si and DTau_si in SI units
        r_schwarzschild = 2 * 6.674e-11 * SOLAR_MASS / (3e8**2)  # 2GM/c^2
        r0_si = 12.0 * r_schwarzschild  # 12 Schwarzschild radii
        DTau_si = 10.0  # 10 seconds proper time
        
        validation_hist, _, _ = engine.run_trajectory(
            theory,
            r0_si=r0_si,
            DTau_si=DTau_si,
            N_STEPS=100,
            verbose=True
        )
        
        print(f"✓ Trajectory completed successfully")
        print(f"  - Number of steps: {len(validation_hist)}")
        print(f"  - Final position: r={validation_hist[-1][1]:.2f}")
        
    except Exception as e:
        print(f"✗ Trajectory failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # Test with quantum solver specifically
    print("\nTesting with UnifiedQuantumSolver...")
    try:
        from physics_agent.geodesic_integrator_stable import create_geodesic_solver
        
        # Create quantum solver
        solver = create_geodesic_solver(
            theory,
            torch.tensor(1.0),  # M=1 in geometric units
            1.0,  # c=1
            1.0,  # G=1
            use_pennylane_quantum=False  # Without PennyLane
        )
        
        print(f"  Solver type: {type(solver).__name__}")
        
        # Test rk4_step method
        y0 = torch.tensor([0.0, 12.0, 0.0, 0.0], dtype=torch.float64)
        y1 = solver.rk4_step(y0, 0.1)
        
        print(f"✓ rk4_step works: {y0} -> {y1}")
        
    except Exception as e:
        print(f"✗ Quantum solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n" + "=" * 60)
    print("Einstein Asymmetric Test Complete ✅")
    print("The UnifiedQuantumSolver integration is working correctly!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_einstein_asymmetric()
    sys.exit(0 if success else 1)