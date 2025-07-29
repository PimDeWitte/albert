#!/usr/bin/env python3
"""Test if proton is hitting event horizon"""

import sys
sys.path.append('.')
import torch
import importlib.util

from physics_agent.theory_engine_core import TheoryEngine

# Load theory
theory_path = "runs/run_20250728_225529_float64/Regularised_Core_QG_ε_1_0e-04/code/theory_source.py"
spec = importlib.util.spec_from_file_location("theory_module", theory_path)
theory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(theory_module)
theory = theory_module.EinsteinRegularisedCore()

print(f"Testing: {theory.name}")

# Create engine
engine = TheoryEngine(verbose=True)

# Test with just electron first (should work)
print("\n1. Testing electron only (should complete)...")
try:
    result = engine.run_multi_particle_trajectories(
        theory, 7e8, 10, 0.01,
        theory_category='quantum',
        particle_names=['electron'],
        progress_type='none',
        verbose=True,
        no_cache=True,
        allow_horizon_crossing=False  # Stop at event horizon
    )
    print(f"Electron result: {result.get('electron', {}).get('tag', 'No tag')}")
except Exception as e:
    print(f"Electron failed: {e}")

# Test with proton (might hit horizon)
print("\n2. Testing proton only...")
try:
    result = engine.run_multi_particle_trajectories(
        theory, 7e8, 10, 0.01,
        theory_category='quantum',
        particle_names=['proton'],
        progress_type='none',
        verbose=True,
        no_cache=True,
        allow_horizon_crossing=False,
        early_stopping=True  # Enable early stopping
    )
    
    if 'proton' in result:
        traj = result['proton'].get('trajectory')
        tag = result['proton'].get('tag', 'No tag')
        print(f"Proton result: {tag}")
        
        if traj is not None:
            # Check radial positions
            r_si = traj[:, 1]  # Radius in SI units
            r_geom = r_si / (6.674e-11 * 1.989e30 / 2.998e8**2)  # Convert to geometric units
            
            print(f"Proton trajectory: {len(traj)} points")
            print(f"Initial r: {r_geom[0]:.3f}M")
            print(f"Final r: {r_geom[-1]:.3f}M")
            print(f"Min r: {min(r_geom):.3f}M")
            
            # Check if it's near event horizon (r = 2M)
            if min(r_geom) < 2.1:
                print(f"CONFIRMED: Proton reached event horizon!")
    else:
        print("No proton trajectory generated")
        
except Exception as e:
    print(f"Proton test failed: {e}")
    import traceback
    traceback.print_exc()

# Test with different initial conditions for proton
print("\n3. Testing proton with larger initial radius...")
try:
    result = engine.run_multi_particle_trajectories(
        theory, 1.4e9,  # Start at 20M instead of 10M
        10, 0.01,
        theory_category='quantum',
        particle_names=['proton'],
        progress_type='none',
        verbose=False,
        no_cache=True,
        allow_horizon_crossing=False
    )
    
    if 'proton' in result and result['proton'].get('trajectory') is not None:
        traj = result['proton']['trajectory']
        r_geom = traj[:, 1] / (6.674e-11 * 1.989e30 / 2.998e8**2)
        print(f"Proton at 20M initial: {len(traj)} points, final r={r_geom[-1]:.3f}M")
    else:
        print(f"Proton at 20M: {result.get('proton', {}).get('tag', 'No result')}")
        
except Exception as e:
    print(f"Proton at 20M failed: {e}") 