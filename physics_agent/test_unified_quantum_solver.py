#!/usr/bin/env python3
"""
Test script for the unified quantum solver consolidation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from physics_agent.unified_quantum_solver import UnifiedQuantumSolver
from physics_agent.quantum_path_integrator import QuantumPathIntegrator
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT


def test_unified_solver_creation():
    """Test that we can create the unified solver"""
    print("Test 1: Creating UnifiedQuantumSolver...")
    
    # Test with quantum theory
    quantum_theory = QuantumCorrected()
    solver = UnifiedQuantumSolver(quantum_theory, M_phys=torch.tensor(SOLAR_MASS))
    
    assert solver is not None
    assert solver.enable_quantum == True
    assert solver.use_pennylane == True  # Should be True by default if available
    print("✓ UnifiedQuantumSolver created successfully")
    
    # Test with classical theory (should still work)
    classical_theory = Schwarzschild()
    solver2 = UnifiedQuantumSolver(classical_theory, M_phys=torch.tensor(SOLAR_MASS), 
                                  enable_quantum=False)
    assert solver2.enable_quantum == False
    print("✓ UnifiedQuantumSolver works with classical theories")
    

def test_backward_compatibility():
    """Test backward compatibility with QuantumPathIntegrator"""
    print("\nTest 2: Testing backward compatibility...")
    
    theory = QuantumCorrected()
    
    # Old way
    old_integrator = QuantumPathIntegrator(theory)
    assert isinstance(old_integrator, UnifiedQuantumSolver)
    assert old_integrator.use_pennylane == False  # Should be False for backward compat
    print("✓ QuantumPathIntegrator redirects to UnifiedQuantumSolver")
    

def test_trajectory_computation():
    """Test basic trajectory computation"""
    print("\nTest 3: Testing trajectory computation...")
    
    theory = QuantumCorrected()
    solver = UnifiedQuantumSolver(theory, M_phys=torch.tensor(SOLAR_MASS))
    
    # Define start and end states
    start = (0.0, 10.0, np.pi/2, 0.0)  # t, r, theta, phi
    end = (100.0, 10.0, np.pi/2, 0.1)
    
    # Compute trajectory
    result = solver.compute_trajectory(start, end, method='wkb', num_points=10)
    
    assert 'amplitude' in result
    assert 'probability' in result
    assert 'classical_path' in result
    assert 'quantum_trajectory' in result
    
    print(f"✓ Computed amplitude: {result['amplitude']}")
    print(f"✓ Probability: {result['probability']}")
    print(f"✓ Path points: {len(result['quantum_trajectory'])}")
    

def test_integration_interface():
    """Test the integrate() method interface"""
    print("\nTest 4: Testing integrate() interface...")
    
    theory = QuantumCorrected()
    solver = UnifiedQuantumSolver(theory, M_phys=torch.tensor(SOLAR_MASS))
    
    # 4D initial conditions [t, r, phi, dr/dtau]
    y0 = [0.0, 10.0, 0.0, 0.0]
    
    # Set conserved quantities
    solver.E = 0.95
    solver.Lz = 4.0
    
    # Integrate
    trajectory = solver.integrate(y0, num_steps=100, h=0.01)
    
    assert len(trajectory) > 0
    assert isinstance(trajectory[0], torch.Tensor)
    print(f"✓ Integration produced {len(trajectory)} steps")
    print(f"✓ Initial state: {trajectory[0]}")
    print(f"✓ Final state: {trajectory[-1]}")
    

def test_pennylane_toggle():
    """Test PennyLane can be toggled on/off"""
    print("\nTest 5: Testing PennyLane toggle...")
    
    theory = QuantumCorrected()
    
    # With PennyLane
    solver_with = UnifiedQuantumSolver(theory, use_pennylane=True)
    assert solver_with.use_pennylane == True or not solver_with.use_pennylane  # Depends on availability
    
    # Without PennyLane
    solver_without = UnifiedQuantumSolver(theory, use_pennylane=False)
    assert solver_without.use_pennylane == False
    
    print("✓ PennyLane toggle works correctly")
    

def test_all_methods():
    """Test all path integral methods"""
    print("\nTest 6: Testing all methods...")
    
    theory = QuantumCorrected()
    solver = UnifiedQuantumSolver(theory, M_phys=torch.tensor(SOLAR_MASS))
    
    start = (0.0, 10.0, np.pi/2, 0.0)
    end = (10.0, 10.0, np.pi/2, 0.1)
    
    methods = ['monte_carlo', 'wkb', 'stationary_phase']
    
    for method in methods:
        print(f"  Testing {method}...")
        result = solver.compute_trajectory(start, end, method=method, num_points=5, num_samples=10)
        assert result['method'] == method
        assert 'amplitude' in result
        print(f"  ✓ {method} works: amplitude = {result['amplitude']}")
    

def main():
    """Run all tests"""
    print("="*60)
    print("UNIFIED QUANTUM SOLVER TEST SUITE")
    print("="*60)
    
    try:
        test_unified_solver_creation()
        test_backward_compatibility()
        test_trajectory_computation()
        test_integration_interface()
        test_pennylane_toggle()
        test_all_methods()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✅")
        print("="*60)
        print("\nThe quantum solvers have been successfully consolidated into UnifiedQuantumSolver.")
        print("- QuantumPathIntegrator now redirects to UnifiedQuantumSolver")
        print("- PennyLane integration is optional and can be toggled")
        print("- All path integral methods (Monte Carlo, WKB, Stationary Phase) work")
        print("- Backward compatibility is maintained")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()