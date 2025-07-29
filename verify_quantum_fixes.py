#!/usr/bin/env python3
"""Verify all quantum theory fixes are working correctly"""

import sys
sys.path.append('.')
import torch
import numpy as np
import importlib.util
import os

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.unified_trajectory_calculator import UnifiedTrajectoryCalculator
from physics_agent.constants import c, G, SOLAR_MASS

print("QUANTUM THEORY FIXES VERIFICATION")
print("="*80)

# Clear relevant cache files first
print("\nClearing cache...")
cache_dir = "physics_agent/cache/trajectories/1.0.0/"
if os.path.exists(cache_dir):
    for f in os.listdir(cache_dir):
        if 'QG' in f or 'Gravity' in f:
            os.remove(os.path.join(cache_dir, f))
            print(f"  Removed: {f}")

# Test theories
test_cases = [
    {
        'path': "runs/run_20250728_225529_float64/Regularised_Core_QG_ε_1_0e-04/code/theory_source.py",
        'class': "EinsteinRegularisedCore",
        'params': {},
        'name': "Regularised Core QG"
    },
    {
        'path': "runs/run_20250728_225529_float64/Participatory_QG_ω_0_00/code/theory_source.py",
        'class': "Participatory",
        'params': {"omega": 0.0},
        'name': "Participatory QG"
    },
    {
        'path': "runs/run_20250728_225529_float64/fail/Emergent_Gravity_η_0_00/code/theory_source.py",
        'class': "Emergent",
        'params': {"eta": 0.0},
        'name': "Emergent Gravity"
    },
    {
        'path': "runs/run_20250728_225529_float64/fail/Log-Corrected_QG_γ_0_000/code/theory_source.py",
        'class': "LogCorrected",
        'params': {"gamma": 0.0},
        'name': "Log-Corrected QG"
    }
]

engine = TheoryEngine(verbose=False)
results = []

for test in test_cases:
    print(f"\n{'='*80}")
    print(f"TESTING: {test['name']}")
    print(f"{'='*80}")
    
    try:
        # Load theory
        spec = importlib.util.spec_from_file_location("theory_module", test['path'])
        theory_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theory_module)
        
        theory_class = getattr(theory_module, test['class'])
        theory = theory_class(**test['params'])
        
        print(f"\n1. Theory Properties:")
        print(f"   Name: {theory.name}")
        print(f"   Category: {theory.category}")
        print(f"   Is quantum: {theory.category == 'quantum'}")
        
        # Test 1: Metric sanity check
        print(f"\n2. Metric Test:")
        r_test = torch.tensor([10.0, 20.0, 50.0], dtype=torch.float64)  # Multiple radii
        M = torch.tensor(1.0, dtype=torch.float64)
        c_t = torch.tensor(1.0, dtype=torch.float64)
        G_t = torch.tensor(1.0, dtype=torch.float64)
        
        g_tt, g_rr, g_pp, g_tp = theory.get_metric(r_test, M, c_t, G_t)
        
        # Check for NaN/Inf
        has_nan = torch.any(torch.isnan(g_tt)) or torch.any(torch.isnan(g_rr))
        has_inf = torch.any(torch.isinf(g_tt)) or torch.any(torch.isinf(g_rr))
        
        print(f"   Has NaN: {has_nan}")
        print(f"   Has Inf: {has_inf}")
        print(f"   g_tt at r=10M: {g_tt[0].item():.6f}")
        print(f"   g_rr at r=10M: {g_rr[0].item():.6f}")
        
        # Check if flat
        is_flat = all(abs(g_tt[i] + 1.0) < 1e-10 and abs(g_rr[i] - 1.0) < 1e-10 for i in range(len(r_test)))
        print(f"   Is flat: {is_flat}")
        
        metric_ok = not has_nan and not has_inf and not is_flat
        print(f"   Metric OK: {metric_ok}")
        
        # Test 2: Solver selection
        print(f"\n3. Solver Test:")
        calc = UnifiedTrajectoryCalculator(
            theory=theory,
            M=SOLAR_MASS,
            c=c,
            G=G,
            enable_classical=True,
            enable_quantum=True
        )
        
        print(f"   Classical solver type: {type(calc.classical_solver).__name__}")
        
        # For quantum theories, check if quantum solver is created
        from physics_agent.geodesic_integrator_stable import is_quantum_theory
        should_use_quantum = is_quantum_theory(theory)
        print(f"   Should use quantum solver: {should_use_quantum}")
        
        # Test 3: Direct trajectory computation
        print(f"\n4. Direct Trajectory Test:")
        initial_conditions = {
            'r': 12.0,  # Geometric units
            't': 0.0,
            'phi': 0.0,
            'E': 0.95,
            'Lz': 3.9,
            'u_t': 1.14,
            'u_r': 0.0,
            'u_phi': 0.0278,
            'particle_name': 'electron',
            'particle_mass': 9.109e-31,
            'particle_charge': -1.602e-19,
            'particle_spin': 0.5
        }
        
        traj_result = calc.compute_classical_trajectory(
            initial_conditions,
            time_steps=20,
            step_size=0.01
        )
        
        if 'trajectory' in traj_result:
            traj = traj_result['trajectory']
            solver_used = traj_result.get('solver_type', 'Unknown')
            
            # Convert back to geometric units for analysis
            r_geom = traj[:, 1] / (G * SOLAR_MASS / c**2)
            
            r_initial = r_geom[0]
            r_final = r_geom[-1]
            r_change = abs(r_final - r_initial)
            r_std = np.std(r_geom)
            
            print(f"   Solver used: {solver_used}")
            print(f"   Using quantum solver: {'Quantum' in solver_used}")
            print(f"   Steps computed: {len(traj)}")
            print(f"   Initial r: {r_initial:.3f}M")
            print(f"   Final r: {r_final:.3f}M")
            print(f"   Radial change: {r_change:.6f}M")
            print(f"   Radial std dev: {r_std:.6f}M")
            print(f"   Shows motion: {r_std > 1e-6}")
            
            trajectory_ok = r_std > 1e-6
        else:
            print(f"   ERROR: {traj_result.get('error', 'Unknown error')}")
            trajectory_ok = False
            solver_used = 'Error'
            
        # Test 4: Full engine test
        print(f"\n5. Full Engine Test:")
        r0_si = 7e8  # 700,000 km
        
        try:
            engine_result = engine.run_multi_particle_trajectories(
                theory, r0_si, 50, 0.01,
                theory_category=theory.category,
                particle_names=['electron'],
                progress_type='none',
                verbose=False,
                no_cache=True  # Force recalculation
            )
            
            if 'electron' in engine_result:
                engine_traj = engine_result['electron']['trajectory']
                engine_solver = engine_result['electron'].get('solver_type', 'Unknown')
                
                # Check radial evolution
                r_engine = engine_traj[:, 1]
                r_engine_std = np.std(r_engine)
                
                print(f"   Engine solver: {engine_solver}")
                print(f"   Trajectory points: {len(engine_traj)}")
                print(f"   Initial r: {r_engine[0]/1e6:.1f} Mm")
                print(f"   Final r: {r_engine[-1]/1e6:.1f} Mm")
                print(f"   Radial std: {r_engine_std/1e3:.3f} km")
                print(f"   Shows motion: {r_engine_std > 100}")  # More than 100m
                
                engine_ok = r_engine_std > 100
            else:
                print(f"   ERROR: No trajectory generated")
                engine_ok = False
                
        except Exception as e:
            print(f"   ERROR: {e}")
            engine_ok = False
            
        # Summary for this theory
        all_ok = metric_ok and trajectory_ok and engine_ok
        using_quantum = 'Quantum' in solver_used if 'solver_used' in locals() else False
        
        results.append({
            'name': test['name'],
            'metric_ok': metric_ok,
            'trajectory_ok': trajectory_ok,
            'engine_ok': engine_ok,
            'using_quantum_solver': using_quantum,
            'all_ok': all_ok
        })
        
        print(f"\n6. Summary:")
        print(f"   Metric OK: {metric_ok}")
        print(f"   Direct trajectory OK: {trajectory_ok}")
        print(f"   Engine trajectory OK: {engine_ok}")
        print(f"   Using quantum solver: {using_quantum}")
        print(f"   ALL TESTS PASSED: {all_ok}")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'name': test['name'],
            'metric_ok': False,
            'trajectory_ok': False,
            'engine_ok': False,
            'using_quantum_solver': False,
            'all_ok': False,
            'error': str(e)
        })

# Final summary
print(f"\n{'='*80}")
print("FINAL VERIFICATION SUMMARY")
print(f"{'='*80}")

for result in results:
    print(f"\n{result['name']}:")
    if 'error' in result:
        print(f"  FATAL ERROR: {result['error']}")
    else:
        print(f"  Metric: {'✓' if result['metric_ok'] else '✗'}")
        print(f"  Trajectory: {'✓' if result['trajectory_ok'] else '✗'}")
        print(f"  Engine: {'✓' if result['engine_ok'] else '✗'}")
        print(f"  Quantum Solver: {'✓' if result['using_quantum_solver'] else '✗'}")
        print(f"  Overall: {'✓ PASS' if result['all_ok'] else '✗ FAIL'}")

passed = sum(1 for r in results if r['all_ok'])
total = len(results)
print(f"\n{passed}/{total} theories passed all tests")

if passed < total:
    print("\nFAILED THEORIES NEED ATTENTION!") 