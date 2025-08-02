#!/usr/bin/env python3
"""
Debug solver timing issues and run trajectory tests with progress bar.
Tests each theory for 1000 steps and shows loss against Kerr baseline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple

# Import engine and constants
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT

# Import all theories to test
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.theories.defaults.baselines.kerr import Kerr
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
from physics_agent.theories.string.theory import StringTheory
from physics_agent.theories.asymptotic_safety.theory import AsymptoticSafetyTheory
from physics_agent.theories.loop_quantum_gravity.theory import LoopQuantumGravity
from physics_agent.theories.yukawa.theory import Yukawa
from physics_agent.theories.spinor_conformal.theory import SpinorConformal

# Test theories
TEST_THEORIES = [
    ("Schwarzschild", Schwarzschild, "baseline"),
    ("Kerr", Kerr, "baseline"),
    ("Quantum Corrected", QuantumCorrected, "quantum"),
    ("String Theory", StringTheory, "quantum"),
    ("Asymptotic Safety", AsymptoticSafetyTheory, "quantum"),
    ("Loop Quantum Gravity", LoopQuantumGravity, "quantum"),
    ("Yukawa", Yukawa, "classical"),
    ("Spinor Conformal", SpinorConformal, "classical"),
]

def run_trajectory_test(theory_name, theory_class, category, kerr_trajectory=None, n_steps=1000):
    """
    Run trajectory integration test for a theory.
    
    Returns:
        trajectory: The computed trajectory
        timing_info: Dict with timing details
        loss_history: Loss vs Kerr at each step (if kerr_trajectory provided)
    """
    print(f"\n{'='*60}")
    print(f"Testing: {theory_name} [{category}]")
    print(f"{'='*60}")
    
    # Initialize theory
    try:
        if theory_name == "Kerr":
            theory = theory_class(a=0.0)  # Non-rotating
        else:
            theory = theory_class()
        print(f"Initialized: {theory.name}")
    except Exception as e:
        print(f"ERROR: Failed to initialize - {e}")
        return None, None, None
    
    # Create engine and setup initial conditions
    engine = TheoryEngine(verbose=False)
    
    # Initial conditions for circular orbit at r=10M
    r0_si = 10 * 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
    dtau_si = 0.01  # Proper time step
    
    # Convert to geometric units
    M_phys = torch.tensor(SOLAR_MASS, dtype=torch.float64)
    r0_geom = r0_si / (GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2)
    dtau_geom = dtau_si / (GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**3)
    
    # Initialize trajectory storage
    trajectory = []
    timing_info = {
        'total_time': 0.0,
        'solver_time': 0.0,
        'steps': 0,
        'time_per_step': 0.0
    }
    loss_history = []
    
    # Get initial conditions
    try:
        # get_initial_conditions returns (y0_symmetric, y0_general, solver_info)
        y0_symmetric, y0_general, solver_info = engine.get_initial_conditions(theory, torch.tensor(r0_geom))
        
        # Create appropriate solver
        if hasattr(theory, 'has_conserved_quantities') and theory.has_conserved_quantities:
            from physics_agent.geodesic_integrator import ConservedQuantityGeodesicSolver
            solver = GeodesicRK4Solver(theory, M_phys)
            # Extract E and Lz from solver_info
            if isinstance(solver_info, dict):
                solver.E = solver_info.get('E', 0.95)
                solver.Lz = solver_info.get('Lz', 4.0)
            elif isinstance(solver_info, torch.Tensor):
                # Fallback if solver_info is a tensor
                solver.E = 0.95
                solver.Lz = 4.0
            # Use symmetric initial conditions for 4D solver
            y0 = y0_symmetric  # Full 4D state: [t, r, phi, dr_dtau]
        else:
            from physics_agent.geodesic_integrator import GeneralRelativisticGeodesicSolver
            solver = GeneralGeodesicRK4Solver(theory, M_phys)
            # Use general initial conditions for 6D solver
            y0 = y0_general
            
    except Exception as e:
        print(f"ERROR: Failed to initialize solver - {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # Run integration with progress bar
    print(f"\nIntegrating {n_steps} steps...")
    start_time = time.time()
    solver_time_total = 0.0
    
    y = y0.clone()
    trajectory.append(y.clone())
    
    print(f"Initial state shape: {y0.shape}, type: {type(solver).__name__}")
    print(f"Initial state values: {y0}")
    
    with tqdm(total=n_steps, desc=f"{theory_name[:20]:<20}") as pbar:
        for step in range(n_steps):
            try:
                # Time the solver step
                solver_start = time.time()
                y_new = solver.rk4_step(y, dtau_geom)
                solver_time = time.time() - solver_start
                solver_time_total += solver_time
                
                if y_new is None:
                    print(f"\nIntegration failed at step {step}")
                    break
                    
                y = y_new
                trajectory.append(y.clone())
                
                # Calculate loss vs Kerr if available
                if kerr_trajectory is not None and step < len(kerr_trajectory) - 1:
                    # Compare radial positions
                    try:
                        # Debug: print state vector info
                        if step == 0:
                            print(f"  State vector shape: {y.shape}, values: {y}")
                        
                        # Handle different state vector formats:
                        # 4D: [t, r, phi, dr_dtau]
                        # 6D: [t, r, phi, u^t, u^r, u^phi]
                        r_theory = y[1]  # r is always at index 1
                        r_kerr = kerr_trajectory[step+1][1]  # r is always at index 1
                        loss = float((r_theory - r_kerr)**2)
                        loss_history.append(loss)
                    except Exception as e:
                        if step == 0:
                            print(f"  Loss calculation error: {e}")
                
                # Check for horizon crossing
                r_current = y[1]  # r is always at index 1 for both 4D and 6D
                if r_current <= 2.1:
                    print(f"\nReached horizon at step {step}")
                    break
                    
                # Update progress bar
                if step % 10 == 0:
                    pbar.set_postfix({
                        'r': f"{r_current:.3f}",
                        'ms/step': f"{solver_time*1000:.3f}"
                    })
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError at step {step}: {e}")
                if step == 0:
                    import traceback
                    traceback.print_exc()
                break
    
    total_time = time.time() - start_time
    
    # Calculate timing statistics
    timing_info['total_time'] = total_time
    timing_info['solver_time'] = solver_time_total
    timing_info['steps'] = len(trajectory) - 1
    timing_info['time_per_step'] = solver_time_total / max(1, len(trajectory) - 1)
    
    print(f"\nCompleted {len(trajectory)-1} steps")
    print(f"Total time: {total_time:.3f}s")
    print(f"Solver time: {solver_time_total:.3f}s ({solver_time_total/total_time*100:.1f}%)")
    print(f"Average: {timing_info['time_per_step']*1000:.3f}ms/step")
    
    return trajectory, timing_info, loss_history

def test_quantum_path_integrator_timing():
    """Debug quantum path integrator timing"""
    print("\n" + "="*80)
    print("QUANTUM PATH INTEGRATOR TIMING DEBUG")
    print("="*80)
    
    # Test with a simple quantum theory
    theory = QuantumCorrected()
    theory.quantum_integrator._debug_timing = True  # Enable debug output
    
    # Test path computation
    start = (0.0, 10.0, np.pi/2, 0.0)
    end = (1.0, 9.5, np.pi/2, 0.1)
    
    print(f"\nTesting quantum path computation...")
    start_time = time.time()
    
    # Try different methods
    methods = ['geodesic', 'monte_carlo', 'wkb']
    
    for method in methods:
        print(f"\n{method.upper()} method:")
        method_start = time.time()
        
        try:
            if method == 'geodesic':
                path = theory.quantum_integrator._compute_geodesic_path(start, end, num_points=100)
            elif method == 'monte_carlo':
                path = theory.quantum_integrator.sample_path_monte_carlo(start, end, num_points=100)
            elif method == 'wkb':
                amplitude = theory.quantum_integrator.compute_amplitude_wkb(start, end)
                print(f"  WKB amplitude: {amplitude}")
                continue
                
            method_time = time.time() - method_start
            print(f"  Computed {len(path)} points in {method_time:.3f}s ({method_time/len(path)*1000:.3f}ms/point)")
            
        except Exception as e:
            print(f"  Failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\nTotal debug time: {total_time:.3f}s")

def plot_trajectory_comparison(results: Dict[str, Tuple], save_path: str = "trajectory_comparison.png"):
    """Plot trajectory comparisons and loss evolution"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot trajectories
    ax1.set_title("Radial Evolution Comparison", fontsize=14)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("r/M")
    
    for theory_name, (trajectory, timing, loss) in results.items():
        if trajectory is None:
            continue
            
        # Extract radial coordinates
        r_values = []
        for state in trajectory:
            r = state[1]  # r is always at index 1
            r_values.append(float(r))
        
        # Plot with timing info in label
        time_per_step = timing['time_per_step'] * 1000  # Convert to ms
        label = f"{theory_name} ({time_per_step:.2f}ms/step)"
        ax1.plot(r_values, label=label, alpha=0.8)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss evolution vs Kerr
    ax2.set_title("Loss vs Kerr Baseline", fontsize=14)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Squared Radial Difference")
    ax2.set_yscale('log')
    
    for theory_name, (trajectory, timing, loss) in results.items():
        if loss and len(loss) > 0:
            ax2.plot(loss, label=theory_name, alpha=0.8)
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")

def main():
    """Run all tests"""
    # First debug quantum path integrator
    test_quantum_path_integrator_timing()
    
    # Run trajectory tests
    print("\n" + "="*80)
    print("TRAJECTORY INTEGRATION TESTS (1000 steps)")
    print("="*80)
    
    results = {}
    kerr_trajectory = None
    
    # Run tests for each theory
    for theory_name, theory_class, category in TEST_THEORIES:
        trajectory, timing, loss = run_trajectory_test(
            theory_name, theory_class, category, 
            kerr_trajectory=kerr_trajectory,
            n_steps=1000
        )
        
        if trajectory is not None:
            results[theory_name] = (trajectory, timing, loss)
            
            # Save Kerr as baseline
            if theory_name == "Kerr":
                kerr_trajectory = trajectory
    
    # Print summary
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    print(f"\n{'Theory':<25} {'Category':<10} {'Steps':<8} {'Total(s)':<10} {'Solver(s)':<10} {'ms/step':<10}")
    print("-"*80)
    
    for theory_name, (trajectory, timing, loss) in results.items():
        category = next(cat for name, cls, cat in TEST_THEORIES if name == theory_name)
        print(f"{theory_name:<25} {category:<10} {timing['steps']:<8} "
              f"{timing['total_time']:<10.3f} {timing['solver_time']:<10.3f} "
              f"{timing['time_per_step']*1000:<10.3f}")
    
    # Plot comparison
    plot_trajectory_comparison(results)
    
    # Highlight suspicious timings
    print("\n" + "="*80)
    print("SUSPICIOUS TIMING ANALYSIS")
    print("="*80)
    
    suspicious = []
    for theory_name, (trajectory, timing, loss) in results.items():
        if timing['time_per_step'] < 1e-6:  # Less than 1 microsecond
            suspicious.append((theory_name, timing['time_per_step']*1000))
    
    if suspicious:
        print("\nTheories with suspiciously fast timing (<1μs/step):")
        for name, ms_per_step in suspicious:
            print(f"  - {name}: {ms_per_step:.6f}ms/step")
        print("\nThese likely indicate fallback to approximations rather than proper integration.")
    else:
        print("\nAll theories show reasonable timing (>1μs/step)")
    
    return results

if __name__ == "__main__":
    results = main()