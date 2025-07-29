#!/usr/bin/env python3
"""Debug the engine unpacking error"""

import sys
sys.path.append('.')
import torch
import importlib.util

from physics_agent.theory_engine_core import TheoryEngine

# Load a theory that's failing
theory_path = "runs/run_20250728_225529_float64/Regularised_Core_QG_ε_1_0e-04/code/theory_source.py"
spec = importlib.util.spec_from_file_location("theory_module", theory_path)
theory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(theory_module)
theory = theory_module.EinsteinRegularisedCore()

print(f"Testing: {theory.name}")

# Create engine with verbose mode
engine = TheoryEngine(verbose=True)

# Try to run trajectory
try:
    print("\nCalling run_trajectory directly...")
    r0_si = 7e8  # 700,000 km
    hist, tag, kicks = engine.run_trajectory(theory, r0_si, 10, 0.01)
    print(f"Direct call successful: tag={tag}, hist shape={hist.shape if hist is not None else None}")
except Exception as e:
    print(f"Direct call failed: {e}")
    import traceback
    traceback.print_exc()

# Try the multi-particle version which is failing
print("\n\nCalling run_multi_particle_trajectories...")
try:
    result = engine.run_multi_particle_trajectories(
        theory, 7e8, 10, 0.01,
        theory_category='quantum',
        particle_names=['electron'],
        progress_type='none',
        verbose=True,
        no_cache=True
    )
    print(f"Multi-particle call successful: {list(result.keys())}")
except Exception as e:
    print(f"Multi-particle call failed: {e}")
    import traceback
    traceback.print_exc() 