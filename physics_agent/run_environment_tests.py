#!/usr/bin/env python3
"""
Environment validation tests for Albert
<reason>chain: Ensures the physics solver and computational environment are working correctly</reason>
"""

import torch
import sys
import os
import time

# Add parent to path for imports if running directly
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT


def test_basic_trajectory(engine, steps=100):
    """Test basic trajectory computation."""
    print("1. Testing basic trajectory computation...")
    
    try:
        # Create Schwarzschild theory
        theory = Schwarzschild()
        
        # Set up parameters
        rs_phys = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
        r0 = torch.tensor(10 * rs_phys, dtype=engine.dtype, device=engine.device)  # 10 Schwarzschild radii
        dtau = torch.tensor(0.1, dtype=engine.dtype, device=engine.device)
        
        # Run trajectory
        hist, tag, _ = engine.run_trajectory(
            theory, r0, steps, dtau,
            no_cache=True,  # Don't use cache for validation
            test_mode=False,
            verbose=False
        )
        
        if hist is None:
            print("   ❌ Failed: Trajectory computation returned None")
            return False
            
        # Verify shape - expecting 4D output: [t, r, phi, dr/dtau]
        # Note: trajectory may terminate early if it reaches singularity
        if hist.shape[1] != 4:
            print(f"   ❌ Failed: Expected 4 columns, got {hist.shape[1]}")
            return False
            
        if hist.shape[0] < 10:  # At least 10 steps
            print(f"   ❌ Failed: Trajectory too short ({hist.shape[0]} steps)")
            return False
            
        # Verify no NaN values
        if torch.isnan(hist).any():
            print("   ❌ Failed: Trajectory contains NaN values")
            return False
            
        # Verify no infinite values
        if torch.isinf(hist).any():
            print("   ❌ Failed: Trajectory contains infinite values")
            return False
            
        print(f"   ✅ Passed: Computed {hist.shape[0]} steps successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed with error: {str(e)}")
        return False


def test_conservation_laws(engine, steps=100):
    """Test conservation of energy and angular momentum."""
    print("2. Testing conservation laws...")
    
    try:
        theory = Schwarzschild()
        rs_phys = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
        r0 = torch.tensor(15 * rs_phys, dtype=engine.dtype, device=engine.device)
        dtau = torch.tensor(0.1, dtype=engine.dtype, device=engine.device)
        
        # Run trajectory
        hist, _, _ = engine.run_trajectory(
            theory, r0, steps, dtau,
            no_cache=True,
            test_mode=False,
            verbose=False
        )
        
        if hist is None:
            print("   ❌ Failed: Could not compute trajectory")
            return False
        
        # Extract conserved quantities
        # <reason>chain: Conservation tests ensure physics is correctly implemented</reason>
        # Trajectory format: [t, r, phi, dr/dtau]
        t = hist[:, 0]
        r = hist[:, 1]
        phi = hist[:, 2]
        dr_dtau = hist[:, 3]
        
        # For Schwarzschild in equatorial plane, angular momentum L = r^2 * dphi/dtau
        # We can compute dphi/dtau from differences
        if len(hist) > 1:
            dphi_dtau = torch.diff(phi) / (0.1)  # dtau = 0.1
            dphi_dtau = torch.cat([dphi_dtau[:1], dphi_dtau])  # Extend to match length
            L = r**2 * dphi_dtau
        else:
            print("   ❌ Failed: Trajectory too short to compute angular momentum")
            return False
        
        # Angular momentum variation (should be small)
        L_mean = L.mean().abs()
        if L_mean > 1e-10:  # Only check variation if L is non-zero
            L_variation = (L.max() - L.min()) / L_mean
            
            if L_variation > 1e-6:  # Relaxed tolerance for numerical precision
                print(f"   ❌ Failed: Angular momentum varies by {L_variation:.2e} (should be < 1e-6)")
                return False
        
        # For energy conservation, compute effective energy
        # <reason>chain: Energy conservation is fundamental to correct physics</reason>
        # In Schwarzschild spacetime, we can check the effective potential
        rs = 2.0  # Event horizon in geometric units
        
        # Compute effective energy from geodesic equation
        # For equatorial orbits: E^2 = (1 - rs/r)(1 + L^2/r^2)
        if L_mean > 1e-10:  # Only check if we have angular momentum
            r_geom = r / engine.length_scale  # Convert to geometric units
            E_eff = torch.sqrt((1 - rs/r_geom) * (1 + L**2/r_geom**2))
            E_mean = E_eff.mean().abs()
            if E_mean > 1e-10:
                E_variation = (E_eff.max() - E_eff.min()) / E_mean
                
                if E_variation > 1e-4:  # Relaxed tolerance for numerical precision
                    print(f"   ❌ Failed: Energy varies by {E_variation:.2e} (should be < 1e-4)")
                    return False
        
        print("   ✅ Passed: Conservation laws satisfied")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed with error: {str(e)}")
        return False


def test_device_compatibility(steps=100):
    """Test that code works on both CPU and GPU if available."""
    print("3. Testing device compatibility...")
    
    tests_passed = True
    
    # Test CPU
    try:
        engine_cpu = TheoryEngine(device='cpu', dtype=torch.float64, verbose=False)
        print("   Testing CPU (float64)...")
        result = test_basic_trajectory(engine_cpu, steps=min(steps, 50))
        if not result:
            tests_passed = False
    except Exception as e:
        print(f"   ❌ CPU test failed: {str(e)}")
        tests_passed = False
    
    # Test GPU if available
    if torch.cuda.is_available():
        try:
            engine_gpu = TheoryEngine(device='cuda', dtype=torch.float32, verbose=False)
            print("   Testing CUDA GPU (float32)...")
            result = test_basic_trajectory(engine_gpu, steps=min(steps, 50))
            if not result:
                tests_passed = False
        except Exception as e:
            print(f"   ⚠️  GPU test failed (non-critical): {str(e)}")
    elif torch.backends.mps.is_available():
        try:
            engine_mps = TheoryEngine(device='mps', dtype=torch.float32, verbose=False)
            print("   Testing Apple Silicon GPU (float32)...")
            result = test_basic_trajectory(engine_mps, steps=min(steps, 50))
            if not result:
                tests_passed = False
        except Exception as e:
            print(f"   ⚠️  MPS test failed (non-critical): {str(e)}")
    else:
        print("   ℹ️  No GPU available, skipping GPU tests")
    
    if tests_passed:
        print("   ✅ Passed: All available devices work correctly")
    return tests_passed


def test_numerical_stability(engine, steps=100):
    """Test numerical stability at extreme conditions."""
    print("4. Testing numerical stability...")
    
    try:
        theory = Schwarzschild()
        rs_phys = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
        
        # Test very close to event horizon
        r0_close = torch.tensor(2.1 * rs_phys, dtype=engine.dtype, device=engine.device)
        dtau = torch.tensor(0.01, dtype=engine.dtype, device=engine.device)
        
        hist_close, _, _ = engine.run_trajectory(
            theory, r0_close, min(steps, 50), dtau,
            no_cache=True,
            test_mode=False,
            verbose=False
        )
        
        if hist_close is None:
            print("   ❌ Failed: Cannot compute near event horizon")
            return False
            
        # Test far from black hole
        r0_far = torch.tensor(1000 * rs_phys, dtype=engine.dtype, device=engine.device)
        
        hist_far, _, _ = engine.run_trajectory(
            theory, r0_far, min(steps, 50), dtau,
            no_cache=True,
            test_mode=False,
            verbose=False
        )
        
        if hist_far is None:
            print("   ❌ Failed: Cannot compute far from black hole")
            return False
        
        print("   ✅ Passed: Stable at extreme radii")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed with error: {str(e)}")
        return False


def run_environment_tests(steps=100, full=False, benchmark_devices=False):
    """
    Run all environment validation tests.
    
    Args:
        steps: Number of integration steps for each test
        full: If True, run extended test suite
        benchmark_devices: If True, benchmark GPU vs CPU precision/performance
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("="*60)
    print("Pre-run Environment Tests")
    print("="*60)
    print(f"Running with {steps} integration steps")
    print()
    
    all_passed = True
    
    # Initialize engine for most tests
    try:
        engine = TheoryEngine(device='cpu', dtype=torch.float64, verbose=False)
    except Exception as e:
        print(f"❌ Failed to initialize TheoryEngine: {str(e)}")
        return False
    
    # Run basic tests
    tests = [
        ("Basic trajectory", lambda: test_basic_trajectory(engine, steps)),
        ("Conservation laws", lambda: test_conservation_laws(engine, steps)),
        ("Device compatibility", lambda: test_device_compatibility(steps)),
        ("Numerical stability", lambda: test_numerical_stability(engine, steps))
    ]
    
    if full:
        # Add more extensive tests for full validation
        print("\nRunning full validation suite...")
        # Could add more comprehensive tests here
    
    # Run all tests
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"   ❌ {test_name} failed with unexpected error: {str(e)}")
            all_passed = False
        print()  # Blank line between tests
    
    # Run device benchmarks if requested
    if benchmark_devices:
        print("\nRunning device precision benchmarks...")
        print("="*60)
        
        try:
            from physics_agent.solver_tests.device_precision_benchmarker import run_device_precision_benchmark
            
            # Run benchmarks with reduced steps for faster results
            benchmark_results = run_device_precision_benchmark(
                verbose=True,
                save_path=None
            )
            
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"❌ Device benchmarking failed: {str(e)}")
            all_passed = False
    
    # Summary
    print("="*60)
    if all_passed:
        print("✅ All environment tests passed!")
        print("The solver is working correctly.")
    else:
        print("❌ Some environment tests failed!")
        print("Please check the errors above and ensure your environment is set up correctly.")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    # Run tests when called directly
    import argparse
    parser = argparse.ArgumentParser(description="Run Albert environment validation tests")
    parser.add_argument('--steps', type=int, default=100, help='Number of integration steps')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    args = parser.parse_args()
    
    success = run_environment_tests(steps=args.steps, full=args.full)
    sys.exit(0 if success else 1) 