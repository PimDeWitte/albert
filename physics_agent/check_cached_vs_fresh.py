#!/usr/bin/env python3
"""Compare cached trajectory vs fresh calculation."""

import torch
import sys
sys.path.insert(0, '.')

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild

def main():
    print("=== Comparing Cached vs Fresh Trajectories ===\n")
    
    engine = TheoryEngine(black_hole_preset='primordial_mini', verbose=False)
    theory = Schwarzschild()
    
    # Parameters
    r0_si = 10 * engine.length_scale
    n_steps = 1000  # Same as comprehensive test
    dtau_geom = engine.bh_preset.integration_parameters['dtau_geometric']
    dtau_si = dtau_geom * engine.time_scale
    
    print(f"Parameters:")
    print(f"  Theory: {theory.name}")
    print(f"  Initial radius: {r0_si:.3e} m")
    print(f"  Steps: {n_steps}")
    print(f"  Timestep: {dtau_si:.3e} s\n")
    
    # First, try with cache (default)
    print("1. Running WITH cache...")
    hist_cached, solver_cached, _ = engine.run_trajectory(
        theory, r0_si, n_steps, dtau_si,
        verbose=False
    )
    
    if hist_cached is not None:
        r_cached = hist_cached[:, 1]
        print(f"  Solver: {solver_cached}")
        print(f"  r range: {r_cached.min():.3e} to {r_cached.max():.3e} m")
        print(f"  r[0] = {r_cached[0]:.3e} m")
        print(f"  Is at origin? {r_cached.max() < 1e-10}\n")
    
    # Now force fresh calculation
    print("2. Running WITHOUT cache (fresh calculation)...")
    hist_fresh, solver_fresh, _ = engine.run_trajectory(
        theory, r0_si, n_steps, dtau_si,
        no_cache=True,
        verbose=False
    )
    
    if hist_fresh is not None:
        r_fresh = hist_fresh[:, 1]
        print(f"  Solver: {solver_fresh}")
        print(f"  r range: {r_fresh.min():.3e} to {r_fresh.max():.3e} m")
        print(f"  r[0] = {r_fresh[0]:.3e} m")
        print(f"  Is at origin? {r_fresh.max() < 1e-10}\n")
    
    # Compare
    print("3. Comparison:")
    if hist_cached is not None and hist_fresh is not None:
        if 'cached' in solver_cached:
            print("  ✓ Cached trajectory was used")
        else:
            print("  ⚠️ Cache was not used despite being available")
            
        r_diff = (r_cached.max() - r_fresh.max()).item()
        print(f"  Difference in max r: {r_diff:.3e} m")
        
        if r_cached.max() < 1e-10 and r_fresh.max() > 1e-10:
            print("\n  ⚠️ PROBLEM FOUND: Cached trajectory is stuck at origin!")
            print("     Fresh calculation shows proper motion.")
            print("     → Need to clear bad cache files")
    
    # Check the cache file directly
    cache_path = engine.get_trajectory_cache_path(theory.name, r0_si, n_steps, dtau_si, 'float64')
    print(f"\n4. Cache file path:")
    print(f"  {cache_path}")
    
    import os
    if os.path.exists(cache_path):
        data = torch.load(cache_path)
        if isinstance(data, dict) and 'trajectory' in data:
            cached_traj = data['trajectory']
        else:
            cached_traj = data
        print(f"  File exists, shape: {cached_traj.shape}")
        print(f"  r values in file: min={cached_traj[:,1].min():.3e}, max={cached_traj[:,1].max():.3e}")

if __name__ == "__main__":
    main()