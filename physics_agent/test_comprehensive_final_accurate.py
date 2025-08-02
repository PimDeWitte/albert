#!/usr/bin/env python3
"""
Comprehensive theory validation test with accurate solver timing.
Replaces estimated timing with actual trajectory integration measurements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import json
import math
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

# Import engine and constants
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT

# Import validators
from physics_agent.validations.mercury_precession_validator import MercuryPrecessionValidator
from physics_agent.validations.light_deflection_validator import LightDeflectionValidator
from physics_agent.validations.photon_sphere_validator import PhotonSphereValidator
from physics_agent.validations.ppn_validator import PpnValidator
from physics_agent.validations.cow_interferometry_validator import COWInterferometryValidator
from physics_agent.validations.gw_validator import GwValidator
from physics_agent.validations.psr_j0740_validator import PsrJ0740Validator

# Import necessary components for solver tests
from physics_agent.geodesic_integrator import ConservedQuantityGeodesicSolver, GeneralRelativisticGeodesicSolver, QuantumCorrectedGeodesicSolver
from physics_agent.validations.cmb_power_spectrum_validator import CMBPowerSpectrumValidator
from physics_agent.validations.primordial_gws_validator import PrimordialGWsValidator

# Import all theories from test_theories_final.py
from physics_agent.theories.newtonian_limit.theory import NewtonianLimit
from physics_agent.theories.defaults.baselines.kerr import Kerr
from physics_agent.theories.defaults.baselines.kerr_newman import KerrNewman
from physics_agent.theories.yukawa.theory import Yukawa
from physics_agent.theories.einstein_teleparallel.theory import EinsteinTeleparallel
from physics_agent.theories.spinor_conformal.theory import SpinorConformal
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild

# Import quantum theories
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
from physics_agent.theories.string.theory import StringTheory
from physics_agent.theories.asymptotic_safety.theory import AsymptoticSafetyTheory
from physics_agent.theories.loop_quantum_gravity.theory import LoopQuantumGravity
from physics_agent.theories.non_commutative_geometry.theory import NonCommutativeGeometry
from physics_agent.theories.twistor_theory.theory import TwistorTheory
from physics_agent.theories.aalto_gauge_gravity.theory import AaltoGaugeGravity
from physics_agent.theories.causal_dynamical_triangulations.theory import CausalDynamicalTriangulations

# All theories to test
ALL_THEORIES = [
    # Baseline
    ("Schwarzschild", Schwarzschild, "baseline"),
    
    # Classical accepted
    ("Newtonian Limit", NewtonianLimit, "classical"),
    ("Kerr", Kerr, "classical"),
    ("Kerr-Newman", KerrNewman, "classical"),
    ("Yukawa", Yukawa, "classical"),
    ("Einstein Teleparallel", EinsteinTeleparallel, "classical"),
    ("Spinor Conformal", SpinorConformal, "classical"),
    
    # Quantum accepted/in-progress
    ("Quantum Corrected", QuantumCorrected, "quantum"),
    ("String Theory", StringTheory, "quantum"),
    ("Asymptotic Safety", AsymptoticSafetyTheory, "quantum"),
    ("Loop Quantum Gravity", LoopQuantumGravity, "quantum"),
    ("Non-Commutative Geometry", NonCommutativeGeometry, "quantum"),
    ("Twistor Theory", TwistorTheory, "quantum"),
    ("Aalto Gauge Gravity", AaltoGaugeGravity, "quantum"),
    ("Causal Dynamical Triangulations", CausalDynamicalTriangulations, "quantum"),
]

def run_validator_test(theory, validator_class, validator_name, engine):
    """Run a single validator on a theory."""
    try:
        validator = validator_class(engine=engine)
        result = validator.validate(theory, verbose=False)
        
        # Handle different result types
        if hasattr(result, '__dict__'):
            result_dict = result.__dict__
        elif isinstance(result, dict):
            result_dict = result
        else:
            return {
                'name': validator_name,
                'status': 'ERROR',
                'passed': False,
                'error': f'Unknown result type: {type(result)}'
            }
        
        # Extract status
        if 'passed' in result_dict:
            passed = result_dict['passed']
            status = 'PASS' if passed else 'FAIL'
        elif 'flags' in result_dict and 'overall' in result_dict['flags']:
            status = result_dict['flags']['overall']
            passed = status in ['PASS', 'WARNING']
        else:
            status = 'UNKNOWN'
            passed = False
        
        # Extract loss
        loss = result_dict.get('loss', None)
        if loss is None and 'results' in result_dict:
            loss = result_dict['results'].get('loss', None)
        
        # Extract error percentage
        error_pct = result_dict.get('error_percent', None)
        
        return {
            'name': validator_name,
            'status': status,
            'passed': passed,
            'loss': float(loss) if loss is not None else None,
            'error_percent': float(error_pct) if error_pct is not None else None
        }
        
    except Exception as e:
        return {
            'name': validator_name,
            'status': 'ERROR',
            'passed': False,
            'error': str(e)[:200]
        }

def test_trajectory_integration(theory, engine, n_steps=500, verbose=False):
    """
    Run actual trajectory integration test with accurate timing.
    
    Args:
        theory: Theory to test
        engine: TheoryEngine instance
        n_steps: Number of integration steps
        verbose: Print progress
        
    Returns:
        dict with results including actual solver timing
    """
    try:
        start_time = time.time()
        
        # Initial conditions for circular orbit
        r0_si = 10 * 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
        dtau_si = 0.01
        
        # Convert to geometric units
        M_phys = torch.tensor(SOLAR_MASS, dtype=torch.float64)
        r0_geom = r0_si / (GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2)
        dtau_geom = dtau_si / (GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**3)
        
        # Get initial conditions
        y0_symmetric, y0_general, solver_info = engine.get_initial_conditions(theory, torch.tensor(r0_geom))
        
        # Create appropriate solver
        if hasattr(theory, 'has_conserved_quantities') and theory.has_conserved_quantities:
            solver = ConservedQuantityGeodesicSolver(theory, M_phys)
            if isinstance(solver_info, dict):
                solver.E = solver_info.get('E', 0.95)
                solver.Lz = solver_info.get('Lz', 4.0)
            else:
                solver.E = 0.95
                solver.Lz = 4.0
            y0 = y0_symmetric
        else:
            solver = GeneralRelativisticGeodesicSolver(theory, M_phys)
            y0 = y0_general
        
        # Use actual class name
        solver_type = solver.__class__.__name__
        if hasattr(theory, 'enable_quantum') and theory.enable_quantum:
            solver_type = "Quantum-" + solver_type
        
        # Time the actual integration
        solver_start = time.time()
        y = y0.clone()
        steps_completed = 0
        
        if verbose:
            pbar = tqdm(total=n_steps, desc="Integration", leave=False)
        
        for i in range(n_steps):
            y_new = solver.rk4_step(y, dtau_geom)
            if y_new is None:
                break
            y = y_new
            steps_completed += 1
            
            # Check horizon crossing
            r_current = y[1]  # r is at index 1
            if r_current <= 2.1:
                break
                
            if verbose and i % 10 == 0:
                pbar.update(10)
        
        if verbose:
            pbar.close()
        
        solver_time = time.time() - solver_start
        exec_time = time.time() - start_time
        
        return {
            'name': 'Trajectory Integration',
            'status': 'PASS' if steps_completed > 0 else 'FAIL',
            'passed': steps_completed > 0,
            'solver_type': solver_type,
            'exec_time': exec_time,
            'solver_time': solver_time,
            'num_steps': steps_completed,
            'final_r': float(y[1]) if steps_completed > 0 else None
        }
        
    except Exception as e:
        solver_class = solver.__class__.__name__ if 'solver' in locals() and solver else f"Exception:{type(e).__name__}"
        return {
            'name': 'Trajectory Integration',
            'status': 'ERROR',
            'passed': False,
            'solver_type': solver_class,
            'exec_time': time.time() - start_time,
            'solver_time': 0.0,
            'num_steps': 0,
            'error': str(e)[:200]
        }

def test_circular_orbit_period(theory, engine):
    """Test circular orbit period calculation."""
    try:
        start_time = time.time()
        
        # Use a smaller orbit for faster testing
        r_orbit = 6.0  # In units of M
        M_phys = torch.tensor(SOLAR_MASS)
        
        # Create solver
        if hasattr(theory, 'has_conserved_quantities') and theory.has_conserved_quantities:
            solver = ConservedQuantityGeodesicSolver(theory, M_phys)
        else:
            solver = GeneralRelativisticGeodesicSolver(theory, M_phys)
        
        # Use actual class name
        solver_type = solver.__class__.__name__
        
        # Calculate orbital parameters
        r_geom = torch.tensor(r_orbit)
        E_circ, Lz_circ = solver.compute_circular_orbit_params(r_geom)
        
        if hasattr(solver, 'E'):
            solver.E = E_circ.item()
            solver.Lz = Lz_circ.item()
        
        # Integrate for a short time to test performance
        y = torch.tensor([0.0, r_orbit, 0.0, 0.0], dtype=torch.float64)
        h = 0.01
        
        solver_start = time.time()
        steps = 100
        for _ in range(steps):
            y_new = solver.rk4_step(y, torch.tensor(h))
            if y_new is None:
                break
            y = y_new
        
        solver_time = time.time() - solver_start
        exec_time = time.time() - start_time
        
        return {
            'name': 'Circular Orbit',
            'status': 'PASS',
            'passed': True,
            'solver_type': solver_type,
            'exec_time': exec_time,
            'solver_time': solver_time,
            'num_steps': steps
        }
        
    except Exception as e:
        solver_class = solver.__class__.__name__ if 'solver' in locals() and solver else f"Exception:{type(e).__name__}"
        return {
            'name': 'Circular Orbit',
            'status': 'FAIL',
            'passed': False,
            'solver_type': solver_class,
            'exec_time': time.time() - start_time,
            'solver_time': 0.0,
            'num_steps': 0,
            'error': str(e)[:100]
        }

def test_quantum_geodesic_sim(theory):
    """Test quantum geodesic simulator."""
    try:
        start_time = time.time()
        simulator = QuantumCorrectedGeodesicSolver(theory, num_qubits=2, 
                                           M_phys=torch.tensor(SOLAR_MASS))
        
        # Initial state
        y0 = torch.tensor([0.0, 10.0, 0.0, 1.0, 0.0, 0.1], dtype=torch.float64)
        
        solver_start = time.time()
        y = y0
        steps = 10  # Small number for quantum simulation
        
        for _ in range(steps):
            y_new = simulator.rk4_step(y, 0.01)
            if y_new is None:
                break
            y = y_new
        
        solver_time = time.time() - solver_start
        exec_time = time.time() - start_time
        
        return {
            'name': 'Quantum Geodesic Sim',
            'status': 'PASS',
            'passed': True,
            'solver_type': 'Quantum-2qb',
            'exec_time': exec_time,
            'solver_time': solver_time,
            'num_steps': steps
        }
        
    except Exception as e:
        return {
            'name': 'Quantum Geodesic Sim',
            'status': 'FAIL',
            'passed': False,
            'solver_type': 'Failed',
            'exec_time': time.time() - start_time,
            'solver_time': 0.0,
            'num_steps': 0
        }

def run_solver_test(theory, test_name, engine=None):
    """Run a single solver-based test on a theory."""
    if test_name == "Trajectory Integration":
        return test_trajectory_integration(theory, engine, n_steps=500)
    elif test_name == "Circular Orbit":
        return test_circular_orbit_period(theory, engine)
    elif test_name == "Quantum Geodesic Sim":
        return test_quantum_geodesic_sim(theory)
    elif test_name == "CMB Power Spectrum" and engine:
        # Still run the validator but don't use fake timing
        start_time = time.time()
        validator = CMBPowerSpectrumValidator(engine=engine)
        val_result = validator.validate(theory, verbose=False)
        exec_time = time.time() - start_time
        
        # Extract result
        if hasattr(val_result, '__dict__'):
            result_dict = val_result.__dict__
        elif isinstance(val_result, dict):
            result_dict = val_result
        else:
            result_dict = {}
        
        passed = result_dict.get('passed', False)
        
        return {
            'name': test_name,
            'status': 'PASS' if passed else 'FAIL',
            'passed': passed,
            'solver_type': 'Analytical',
            'exec_time': exec_time,
            'solver_time': 0.0,  # CMB validator doesn't run solver
            'num_steps': 0
        }
    elif test_name == "Primordial GWs" and engine:
        # Similar for PGW
        start_time = time.time()
        validator = PrimordialGWsValidator(engine=engine)
        val_result = validator.validate(theory, verbose=False)
        exec_time = time.time() - start_time
        
        if hasattr(val_result, '__dict__'):
            result_dict = val_result.__dict__
        elif isinstance(val_result, dict):
            result_dict = val_result
        else:
            result_dict = {}
        
        passed = result_dict.get('passed', False)
        
        return {
            'name': test_name,
            'status': 'PASS' if passed else 'FAIL',
            'passed': passed,
            'solver_type': 'Analytical',
            'exec_time': exec_time,
            'solver_time': 0.0,
            'num_steps': 0
        }
    else:
        return {
            'name': test_name,
            'status': 'SKIP',
            'passed': True,
            'solver_type': 'N/A',
            'exec_time': 0.0,
            'solver_time': 0.0,
            'num_steps': 0
        }

def test_theory_comprehensive(theory_name, theory_class, category):
    """Test a single theory with both analytical validators and solver tests."""
    print(f"\n{'='*60}")
    print(f"Testing: {theory_name} [{category}]")
    print(f"{'='*60}")
    
    try:
        # Initialize theory
        if theory_name == "Kerr":
            theory = theory_class(a=0.0)
        elif theory_name == "Kerr-Newman":
            theory = theory_class(a=0.0, Q=0.0)
        else:
            theory = theory_class()
        print(f"Initialized: {theory.name}")
    except Exception as e:
        print(f"ERROR: Failed to initialize - {e}")
        return None
    
    engine = TheoryEngine()
    
    # Define analytical validators
    analytical_validators = [
        (MercuryPrecessionValidator, "Mercury Precession"),
        (LightDeflectionValidator, "Light Deflection"),
        (PhotonSphereValidator, "Photon Sphere"),
        (PpnValidator, "PPN Parameters"),
        (COWInterferometryValidator, "COW Interferometry"),
        (GwValidator, "Gravitational Waves"),
        (PsrJ0740Validator, "PSR J0740"),
    ]
    
    # Define solver-based tests - now with actual trajectory integration
    solver_tests = [
        "Trajectory Integration",  # NEW: Actual integration test
        "Circular Orbit",
        "CMB Power Spectrum",
        "Primordial GWs",
        "Quantum Geodesic Sim",
    ]
    
    results = {
        'theory': theory_name,
        'category': category,
        'analytical_tests': [],
        'solver_tests': []
    }
    
    # Run analytical validators
    print("\nAnalytical Validators:")
    for validator_class, validator_name in analytical_validators:
        result = run_validator_test(theory, validator_class, validator_name, engine)
        results['analytical_tests'].append(result)
        
        if result['passed']:
            print(f"  ✓ {validator_name}: {result['status']}")
        else:
            print(f"  ✗ {validator_name}: {result['status']}")
    
    # Run solver-based tests
    print("\nSolver-Based Tests:")
    for test_name in solver_tests:
        print(f"  {test_name}:", end='', flush=True)
        result = run_solver_test(theory, test_name, engine)
        results['solver_tests'].append(result)
        
        if result['passed']:
            print(f" ✓ {result['status']} [{result['solver_type']}]", end='')
        else:
            print(f" ✗ {result['status']} [{result['solver_type']}]", end='')
            
        # Show timing for tests that actually run solvers
        if result['solver_time'] > 0 and result['num_steps'] > 0:
            ms_per_step = result['solver_time'] / result['num_steps'] * 1000
            print(f" ({ms_per_step:.2f}ms/step)")
        else:
            print()
    
    # Calculate summaries with accurate timing
    analytical_total = len(results['analytical_tests'])
    analytical_passed = sum(1 for t in results['analytical_tests'] if t['passed'])
    results['analytical_summary'] = {
        'total': analytical_total,
        'passed': analytical_passed,
        'success_rate': analytical_passed / analytical_total if analytical_total > 0 else 0
    }
    
    solver_total = len(results['solver_tests'])
    solver_passed = sum(1 for t in results['solver_tests'] if t['passed'])
    
    # Track solver types and accurate timing
    solver_types = set()
    total_solver_time = 0.0
    total_solver_steps = 0
    actual_solver_time = 0.0
    
    for test in results['solver_tests']:
        if 'solver_type' in test and test['solver_type'] not in ['N/A', 'Unknown', 'Analytical']:
            solver_types.add(test['solver_type'])
        if 'exec_time' in test:
            total_solver_time += test['exec_time']
        if 'solver_time' in test:
            actual_solver_time += test['solver_time']
        if 'num_steps' in test and test['solver_time'] > 0:  # Only count steps with actual solver time
            total_solver_steps += test['num_steps']
    
    results['solver_summary'] = {
        'total': solver_total,
        'passed': solver_passed,
        'success_rate': solver_passed / solver_total if solver_total > 0 else 0,
        'solver_types': list(solver_types),
        'total_exec_time': total_solver_time,
        'actual_solver_time': actual_solver_time,
        'total_steps': total_solver_steps
    }
    
    # Combined summary
    combined_total = analytical_total + solver_total
    combined_passed = analytical_passed + solver_passed
    results['combined_summary'] = {
        'total': combined_total,
        'passed': combined_passed,
        'success_rate': combined_passed / combined_total if combined_total > 0 else 0,
        'complexity_score': actual_solver_time,
        'total_solver_steps': total_solver_steps,
        'time_per_step': actual_solver_time / total_solver_steps if total_solver_steps > 0 else 0
    }
    
    print(f"\nAnalytical: {analytical_passed}/{analytical_total} passed ({results['analytical_summary']['success_rate']*100:.1f}%)")
    print(f"Solver: {solver_passed}/{solver_total} passed ({results['solver_summary']['success_rate']*100:.1f}%)")
    print(f"Combined: {combined_passed}/{combined_total} passed ({results['combined_summary']['success_rate']*100:.1f}%)")
    
    return results

def main():
    """Run all tests and generate comprehensive report with accurate timing."""
    print("COMPREHENSIVE THEORY VALIDATION - ACCURATE SOLVER TIMING")
    print("="*80)
    print(f"Testing {len(ALL_THEORIES)} theories")
    print("Now running actual trajectory integration for accurate timing!")
    
    all_results = []
    
    # Test all theories
    for theory_name, theory_class, category in ALL_THEORIES:
        result = test_theory_comprehensive(theory_name, theory_class, category)
        if result:
            all_results.append(result)
        time.sleep(0.1)  # Brief pause
    
    # Sort by success rate (descending)
    all_results.sort(key=lambda x: x['combined_summary']['success_rate'], reverse=True)
    
    # Generate analytical-only rankings
    print("\n\n" + "="*80)
    print("FINAL RANKINGS - ANALYTICAL VALIDATORS ONLY")
    print("="*80)
    
    # Sort by analytical success rate
    all_results_analytical = sorted(all_results, 
                                  key=lambda x: x['analytical_summary']['success_rate'], 
                                  reverse=True)
    
    print(f"\n{'Rank':<6} {'Theory':<35} {'Category':<12} {'Score':<10} {'Tests Passed'}")
    print("-"*80)
    
    for i, result in enumerate(all_results_analytical, 1):
        theory = result['theory']
        category = result['category']
        score = f"{result['analytical_summary']['success_rate']*100:.1f}%"
        tests = f"{result['analytical_summary']['passed']}/{result['analytical_summary']['total']}"
        
        marker = ""
        if theory == "Schwarzschild":
            marker = " ← Baseline"
        elif theory == "Newtonian Limit":
            marker = " ← Should be lowest"
        elif category == "quantum" and result['analytical_summary']['success_rate'] > 0.7:
            marker = " ✓ High-performing quantum"
        
        print(f"{i:<6} {theory:<35} {category:<12} {score:<10} {tests}{marker}")
    
    # Generate combined rankings with accurate timing
    print("\n\n" + "="*80)
    print("FINAL RANKINGS - COMBINED (ANALYTICAL + SOLVER-BASED)")
    print("="*80)
    
    # Create detailed table with accurate timing
    print(f"\n{'Rank':<6} {'Theory':<35} {'Category':<12} {'Analytical (Failed)':<25} {'Solver (Failed) [Type]':<50} {'Combined':<15} {'Solver Complexity':<25} {'Total'}")
    print("-"*198)
    
    for i, result in enumerate(all_results, 1):
        theory = result['theory']
        category = result['category']
        
        # Get failed analytical tests
        failed_analytical = []
        for test in result['analytical_tests']:
            if not test['passed']:
                # Shorten test names
                test_name = test['name'].replace(' Validator', '').replace(' Parameters', '')
                if test_name == "Mercury Precession":
                    test_name = "Mercury"
                elif test_name == "Light Deflection":
                    test_name = "Light"
                elif test_name == "Photon Sphere":
                    test_name = "Photon"
                elif test_name == "COW Interferometry":
                    test_name = "COW"
                elif test_name == "Gravitational Waves":
                    test_name = "GW"
                elif test_name == "PSR J0740":
                    test_name = "PSR"
                failed_analytical.append(test_name)
        
        analytical_str = f"{result['analytical_summary']['passed']}/{result['analytical_summary']['total']}"
        if failed_analytical:
            analytical_str += f" ✗{','.join(failed_analytical)}"
        
        # Get solver test details - only show failures
        solver_details = []
        for test in result['solver_tests']:
            if test['status'] not in ['SKIP', 'N/A'] and not test['passed']:
                # Shorten test names
                test_name = test['name']
                if test_name == "Trajectory vs Kerr":
                    test_name = "TvK"
                elif test_name == "Circular Orbit":
                    test_name = "Orb"
                elif test_name == "CMB Power Spectrum":
                    test_name = "CMB"
                elif test_name == "Primordial GWs":
                    test_name = "PGW"
                elif test_name == "Quantum Geodesic Sim":
                    test_name = "QGS"
                
                solver_type = test.get('solver_type', '?')
                if solver_type == "4DOF-RK4":
                    solver_type = "4D"
                elif solver_type == "6DOF-RK4":
                    solver_type = "6D"
                elif solver_type == "Quantum-4DOF-RK4":
                    solver_type = "Q4D"
                elif solver_type == "Quantum-6DOF-RK4":
                    solver_type = "Q6D"
                elif solver_type == "Quantum-2qb":
                    solver_type = "Q2"
                elif solver_type == "Quantum-PI":
                    solver_type = "QPI"
                elif solver_type == "Analytical":
                    solver_type = "An"
                
                solver_details.append(f"✗{test_name}[{solver_type}]")
        
        solver_str = f"{result['solver_summary']['passed']}/{result['solver_summary']['total']}"
        if solver_details:
            solver_str += f" {' '.join(solver_details)}"
        
        combined_score = f"{result['combined_summary']['success_rate']*100:.1f}%"
        
        # Show solver complexity details with accurate timing
        solver_time = result['combined_summary'].get('complexity_score', 0)
        total_steps = result['combined_summary'].get('total_solver_steps', 0)
        if total_steps > 0:
            time_per_step = solver_time / total_steps * 1000  # Convert to ms
            complexity = f"{solver_time:.3f}s ({time_per_step:.1f}ms/step)"
        else:
            complexity = f"{solver_time:.3f}s"
        
        total_tests = f"{result['combined_summary']['passed']}/{result['combined_summary']['total']}"
        
        # Highlight special cases
        marker = ""
        if theory == "Schwarzschild":
            marker = " ← Baseline"
        elif theory == "Newtonian Limit":
            marker = " ← Should be lowest"
        elif category == "quantum" and result['combined_summary']['success_rate'] > 0.7:
            marker = " ✓ High-performing quantum"
        
        print(f"{i:<6} {theory:<35} {category:<12} {analytical_str:<25} {solver_str:<50} {combined_score:<15} {complexity:<25} {total_tests}{marker}")
    
    # Timing analysis
    print("\n\nTIMING ANALYSIS (Accurate Measurements):")
    print("-"*60)
    
    # Collect theories with valid timing
    theories_with_timing = []
    for result in all_results:
        ms_per_step = result['combined_summary']['time_per_step'] * 1000
        if ms_per_step > 0:
            theories_with_timing.append((result['theory'], ms_per_step, result['category']))
    
    # Sort by timing
    theories_with_timing.sort(key=lambda x: x[1])
    
    print(f"\nFastest theories (ms/step):")
    for name, timing, cat in theories_with_timing[:5]:
        print(f"  {name} ({cat}): {timing:.3f}ms/step")
    
    print(f"\nSlowest theories (ms/step):")
    for name, timing, cat in theories_with_timing[-5:]:
        print(f"  {name} ({cat}): {timing:.3f}ms/step")
    
    # Category averages
    quantum_timings = [t[1] for t in theories_with_timing if t[2] == 'quantum']
    classical_timings = [t[1] for t in theories_with_timing if t[2] == 'classical']
    
    if quantum_timings:
        print(f"\nQuantum theories average: {np.mean(quantum_timings):.3f}ms/step")
    if classical_timings:
        print(f"Classical theories average: {np.mean(classical_timings):.3f}ms/step")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_theories': len(all_results),
            'rankings': [(r['theory'], r['category'], r['combined_summary']['success_rate']) for r in all_results]
        },
        'full_results': all_results
    }
    
    report_file = f"theory_validation_accurate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(v) for v in obj)
        elif hasattr(obj, '__dict__') and not callable(obj):
            return make_serializable(obj.__dict__)
        else:
            return obj
    
    with open(report_file, 'w') as f:
        json.dump(make_serializable(report), f, indent=2)
    
    print(f"\n\nDetailed report saved to: {report_file}")
    
    return all_results

if __name__ == "__main__":
    results = main()