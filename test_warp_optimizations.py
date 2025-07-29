#!/usr/bin/env python3
"""
Test script for NVIDIA Warp GPU optimizations in Albert.

This script demonstrates:
1. How to check if Warp is available
2. How to enable Warp optimizations
3. Performance comparisons
4. Running physics validation tests
"""

import sys
import time
import torch

def check_warp_installation():
    """Check if Warp is installed and available"""
    print("=" * 60)
    print("Checking NVIDIA Warp Installation")
    print("=" * 60)
    
    try:
        import warp as wp
        print(f"✓ Warp {wp.__version__} is installed")
        
        # Initialize Warp
        wp.init()
        devices = wp.get_devices()
        print(f"✓ Available devices: {devices}")
        
        # Check for CUDA
        cuda_available = any('cuda' in str(d) for d in devices)
        if cuda_available:
            print("✓ CUDA device detected - GPU acceleration available!")
        else:
            print("⚠ No CUDA device - CPU-only mode")
            
        return True, cuda_available
        
    except ImportError:
        print("✗ Warp not installed")
        print("\nTo install:")
        print("  pip install warp-lang")
        return False, False


def test_warp_theory_engine():
    """Test Warp optimizations with TheoryEngine"""
    print("\n" + "=" * 60)
    print("Testing Warp Optimizations with TheoryEngine")
    print("=" * 60)
    
    # Import required modules
    from physics_agent.theory_engine_core import TheoryEngine
    from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
    from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT
    
    # Check if we should enable Warp
    try:
        from physics_agent.theory_engine_warp_integration import inject_warp_optimizations
        import physics_agent.theory_engine_core as tec
        
        print("\nEnabling Warp optimizations...")
        inject_warp_optimizations(tec)
        print("✓ Warp optimizations injected")
        warp_enabled = True
    except ImportError:
        print("⚠ Warp integration not available")
        warp_enabled = False
    
    # Create theory and engine
    theory = Schwarzschild()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    engine = TheoryEngine(device=device)
    
    print(f"\nTesting with:")
    print(f"  Theory: {theory.name}")
    print(f"  Device: {device}")
    print(f"  Warp: {'Enabled' if warp_enabled else 'Disabled'}")
    
    # Test parameters
    rs_phys = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
    r0 = 100 * rs_phys  # 100 Schwarzschild radii
    n_steps = 10000
    dtau = 1e-5
    
    # Run trajectory computation
    print(f"\nComputing trajectory ({n_steps} steps)...")
    start = time.time()
    hist, tag, _ = engine.run_trajectory(
        theory, r0, n_steps, dtau, 
        no_cache=True, 
        verbose=False,
        use_warp=warp_enabled
    )
    elapsed = time.time() - start
    
    print(f"✓ Computation completed in {elapsed:.3f}s")
    print(f"  Steps/second: {n_steps/elapsed:.0f}")
    print(f"  Tag: {tag}")
    
    return elapsed


def run_validation_test_11():
    """Run Test 11 from geodesic validator comparison"""
    print("\n" + "=" * 60)
    print("Running Validation Test 11: Warp GPU Optimization")
    print("=" * 60)
    
    try:
        # Import the test
        sys.path.append('physics_agent/solver_tests')
        from test_geodesic_validator_comparison import test_warp_gpu_optimization
        
        # Run the test
        print("\nExecuting test_warp_gpu_optimization()...")
        result = test_warp_gpu_optimization()
        
        if result:
            print("\n✅ Test 11 PASSED")
        else:
            print("\n❌ Test 11 FAILED")
            
        return result
        
    except Exception as e:
        print(f"\n❌ Error running Test 11: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_performance():
    """Compare performance with and without Warp"""
    print("\n" + "=" * 60)
    print("Performance Comparison: Standard vs Warp")
    print("=" * 60)
    
    # Only run if Warp is available
    warp_available, cuda_available = check_warp_installation()
    if not warp_available:
        print("\nSkipping performance comparison - Warp not available")
        return
    
    from physics_agent.warp_simple_demo import benchmark_simple_warp
    
    print("\nRunning performance benchmark...")
    speedup = benchmark_simple_warp()
    
    if speedup and speedup > 1.0:
        print(f"\n✅ Warp provides {speedup:.1f}x speedup!")
    else:
        print(f"\n⚠ Limited speedup ({speedup:.1f}x) - may need GPU")


def main():
    """Main test function"""
    print("NVIDIA Warp GPU Optimization Test Suite for Albert")
    print("=" * 60)
    
    # 1. Check installation
    warp_available, cuda_available = check_warp_installation()
    
    if not warp_available:
        print("\n⚠ Please install Warp to continue:")
        print("  pip install warp-lang")
        return 1
    
    # 2. Test with TheoryEngine
    test_warp_theory_engine()
    
    # 3. Run validation test
    run_validation_test_11()
    
    # 4. Performance comparison
    compare_performance()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("\nTo use Warp in your runs:")
    print("  albert run --experimental-warp --gpu-f32")
    print("  python -m physics_agent.theory_engine_core --experimental-warp --gpu-f32")
    print("\nBest practices:")
    print("- Use with symmetric spacetimes (Schwarzschild, Reissner-Nordström)")
    print("- Combine with --gpu-f32 for maximum performance")
    print("- Expect 10-100x speedup on NVIDIA GPUs")
    print("- CPU-only systems will see limited benefit")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 