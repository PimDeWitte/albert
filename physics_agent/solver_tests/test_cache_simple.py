#!/usr/bin/env python3
"""
Simple test to verify trajectory caching works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import time
import shutil

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT
from physics_agent.cache import TrajectoryCache


def test_simple_caching():
    """Test basic caching functionality."""
    print("Testing trajectory caching...")
    
    # Clear cache first
    cache = TrajectoryCache()
    if os.path.exists(cache.cache_base_dir):
        shutil.rmtree(cache.cache_base_dir)
    os.makedirs(cache.trajectories_dir, exist_ok=True)
    
    # Initialize
    theory = Schwarzschild()
    engine = TheoryEngine(device='cpu', dtype=torch.float64, verbose=True)
    
    # Set up parameters
    rs_phys = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
    r0 = torch.tensor(10 * rs_phys, dtype=torch.float64)  # 10 Schwarzschild radii
    n_steps = 1000
    dtau = torch.tensor(0.1, dtype=torch.float64)
    
    print(f"\nTest parameters:")
    print(f"  Initial radius: {r0.item()/1000:.1f} km")
    print(f"  Steps: {n_steps}")
    print(f"  Time step: {dtau.item()}")
    
    # First run - should compute
    print("\n1. First run (computing trajectory):")
    start1 = time.time()
    hist1, tag1, kicks1 = engine.run_trajectory(
        theory, r0, n_steps, dtau,
        no_cache=False,
        test_mode=False,
        verbose=True
    )
    time1 = time.time() - start1
    
    print(f"\n  Time: {time1:.3f}s")
    print(f"  Tag: {tag1}")
    if hist1 is not None:
        print(f"  Trajectory shape: {hist1.shape}")
    else:
        print(f"  Trajectory: None (failed)")
    
    # Second run - should use cache
    print("\n2. Second run (should use cache):")
    start2 = time.time()
    hist2, tag2, kicks2 = engine.run_trajectory(
        theory, r0, n_steps, dtau,
        no_cache=False,
        test_mode=False,
        verbose=True
    )
    time2 = time.time() - start2
    
    print(f"\n  Time: {time2:.3f}s")
    print(f"  Tag: {tag2}")
    print(f"  Speedup: {time1/time2:.1f}x")
    
    # Verify results are identical
    if hist1 is not None and hist2 is not None:
        if torch.allclose(hist1, hist2):
            print("\n✓ Trajectories match exactly")
        else:
            print("\n✗ ERROR: Trajectories don't match!")
    else:
        print("\n✗ ERROR: One or both trajectories are None")
    
    # Check cache files
    cache_files = []
    for root, dirs, files in os.walk(cache.trajectories_dir):
        for f in files:
            if f.endswith('.pt'):
                cache_files.append(f)
    
    print(f"\nCache files created: {len(cache_files)}")
    for f in cache_files:
        print(f"  - {f}")
    
    # Test with no_cache=True
    print("\n3. Third run with no_cache=True:")
    start3 = time.time()
    hist3, tag3, kicks3 = engine.run_trajectory(
        theory, r0, n_steps, dtau,
        no_cache=True,  # Disable cache
        test_mode=False,
        verbose=False
    )
    time3 = time.time() - start3
    
    print(f"  Time: {time3:.3f}s")
    print(f"  Tag: {tag3}")
    print(f"  Should be similar to first run time: {abs(time3 - time1) < 0.5}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"First run (compute): {time1:.3f}s")
    print(f"Second run (cache): {time2:.3f}s")
    print(f"Third run (no cache): {time3:.3f}s")
    print(f"Cache speedup: {time1/time2:.1f}x")
    
    success = tag2 == 'cached_trajectory' and time2 < time1 * 0.5
    print(f"\nCaching {'WORKS' if success else 'FAILED'}!")
    
    return success


if __name__ == "__main__":
    success = test_simple_caching()
    sys.exit(0 if success else 1) 