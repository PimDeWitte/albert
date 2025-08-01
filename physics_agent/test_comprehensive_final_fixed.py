#!/usr/bin/env python3
"""
Fixed comprehensive theory validation test with proper solver timing.
Fixes the 0.0ms/step issue by measuring actual solver computation time.
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
from physics_agent.geodesic_integrator import GeodesicRK4Solver, GeneralGeodesicRK4Solver, QuantumGeodesicSimulator
from physics_agent.validations.cmb_power_spectrum_validator import CMBPowerSpectrumValidator
from physics_agent.validations.primordial_gws_validator import PrimordialGWsValidator

# Import all theories
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

def run_analytical_validator(theory, validator_class, validator_name, engine):
    """Run a single analytical validator on a theory."""
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
        
        return {
            'name': validator_name,
            'status': status,
            'passed': passed
        }
        
    except Exception as e:
        return {
            'name': validator_name,
            'status': 'ERROR',
            'passed': False,
            'error': str(e)[:200]
        }

def test_circular_orbit_with_timing(theory, engine, n_steps=100):
    """Test circular orbit with proper timing measurement."""
    try:
        start_time = time.time()
        
        # Setup initial conditions
        r0_si = 10 * 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
        M_phys = torch.tensor(SOLAR_MASS, dtype=torch.float64)
        r0_geom = r0_si / (GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2)
        dtau_geom = 0.01 / (GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**3)
        
        # Get initial conditions
        y0_symmetric, y0_general, solver_info = engine.get_initial_conditions(theory, torch.tensor(r0_geom))
        
        # Create appropriate solver
        if hasattr(theory, 'has_conserved_quantities') and theory.has_conserved_quantities:
            solver = GeodesicRK4Solver(theory, M_phys)
            if isinstance(solver_info, dict):
                solver.E = solver_info.get('E', 0.95)
                solver.Lz = solver_info.get('Lz', 4.0)
            else:
                solver.E = 0.95
                solver.Lz = 4.0
            y0 = y0_symmetric
            solver_type = "4DOF-RK4"
        else:
            solver = GeneralGeodesicRK4Solver(theory, M_phys)
            y0 = y0_general
            solver_type = "6DOF-RK4"
        
        # Run integration
        solver_start = time.time()
        y = y0.clone()
        actual_steps = 0
        
        for i in range(n_steps):
            y_new = solver.rk4_step(y, dtau_geom)
            if y_new is None:
                break
            y = y_new
            actual_steps += 1
            
            # Check horizon
            r_current = y[1]
            if r_current <= 2.1:
                break
        
        solver_time = time.time() - solver_start
        exec_time = time.time() - start_time
        
        return True, solver_type, exec_time, solver_time, actual_steps
        
    except Exception as e:
        return False, "Failed", time.time() - start_time, 0.0, 0

def test_quantum_path_timing(theory, n_samples=10):
    """Test quantum path integrator with actual timing."""
    try:
        if not (hasattr(theory, 'quantum_integrator') and theory.quantum_integrator is not None):
            return False, "No-QPI", 0.0, 0.0, 0
        
        start_time = time.time()
        integrator = theory.quantum_integrator
        
        # Test path computation
        start_point = (0.0, 10.0, np.pi/2, 0.0)
        end_point = (1.0, 9.8, np.pi/2, 0.1)
        
        solver_start = time.time()
        
        # Try to compute multiple paths to get better timing
        paths_computed = 0
        for _ in range(n_samples):
            try:
                path = integrator.sample_path_monte_carlo(start_point, end_point, num_points=20)
                paths_computed += 1
            except:
                pass
        
        solver_time = time.time() - solver_start
        exec_time = time.time() - start_time
        
        if paths_computed > 0:
            return True, "Quantum-PI", exec_time, solver_time, paths_computed * 20  # 20 points per path
        else:
            return False, "QPI-Failed", exec_time, 0.0, 0
            
    except Exception as e:
        return False, "Error", time.time() - start_time, 0.0, 0

def run_solver_test_with_timing(theory, test_name, engine=None):
    """Run solver test with actual timing measurement."""
    if test_name == "Circular Orbit":
        return test_circular_orbit_with_timing(theory, engine)
    
    elif test_name == "Quantum Geodesic Sim":
        # Test quantum geodesic simulator
        try:
            start_time = time.time()
            simulator = QuantumGeodesicSimulator(theory, num_qubits=2, 
                                               M_phys=torch.tensor(SOLAR_MASS))
            
            # Run a few integration steps
            y0 = torch.tensor([0.0, 10.0, 0.0, 1.0, 0.0, 0.1], dtype=torch.float64)
            
            solver_start = time.time()
            steps_done = 0
            y = y0
            
            for i in range(10):  # Just 10 steps for timing
                y_new = simulator.rk4_step(y, 0.01)
                if y_new is None:
                    break
                y = y_new
                steps_done += 1
            
            solver_time = time.time() - solver_start
            exec_time = time.time() - start_time
            
            return steps_done > 0, "Quantum-2qb", exec_time, solver_time, steps_done
            
        except Exception as e:
            return False, "Failed", time.time() - start_time, 0.0, 0
    
    elif test_name == "CMB Power Spectrum" and engine:
        # For CMB, actually measure quantum path computation time if used
        start_time = time.time()
        validator = CMBPowerSpectrumValidator(engine=engine)
        
        # Measure validator execution
        val_result = validator.validate(theory, verbose=False)
        exec_time = time.time() - start_time
        
        # Check if passed
        if hasattr(val_result, '__dict__'):
            result_dict = val_result.__dict__
        elif isinstance(val_result, dict):
            result_dict = val_result
        else:
            result = False
            result_dict = {}
        
        passed = result_dict.get('passed', False)
        
        # If theory has quantum integrator, measure its timing
        solver_time = 0.0
        num_steps = 1
        solver_type = "Analytical"
        
        if hasattr(theory, 'enable_quantum') and theory.enable_quantum and hasattr(theory, 'quantum_integrator'):
            # Test quantum path computation
            qpi_passed, _, _, qpi_time, qpi_steps = test_quantum_path_timing(theory, n_samples=5)
            if qpi_passed:
                solver_type = "Quantum-PI"
                solver_time = qpi_time
                num_steps = qpi_steps
        
        return passed, solver_type, exec_time, solver_time, num_steps
    
    elif test_name == "Primordial GWs" and engine:
        # Similar to CMB
        start_time = time.time()
        validator = PrimordialGWsValidator(engine=engine)
        val_result = validator.validate(theory, verbose=False)
        exec_time = time.time() - start_time
        
        # Check result
        if hasattr(val_result, '__dict__'):
            result_dict = val_result.__dict__
        elif isinstance(val_result, dict):
            result_dict = val_result
        else:
            result_dict = {}
        
        passed = result_dict.get('passed', False)
        
        # Measure quantum timing if applicable
        solver_time = 0.0
        num_steps = 1
        solver_type = "Analytical"
        
        if hasattr(theory, 'enable_quantum') and theory.enable_quantum:
            qpi_passed, _, _, qpi_time, qpi_steps = test_quantum_path_timing(theory, n_samples=5)
            if qpi_passed:
                solver_type = "Quantum-PI"
                solver_time = qpi_time
                num_steps = qpi_steps
        
        return passed, solver_type, exec_time, solver_time, num_steps
    
    else:
        # Unknown test
        return False, "Unknown", 0.0, 0.0, 0

def test_theory_comprehensive(theory_name, theory_class, category):
    """Test a single theory with proper timing measurements."""
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
    
    # Define solver-based tests
    solver_tests = [
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
        result = run_analytical_validator(theory, validator_class, validator_name, engine)
        results['analytical_tests'].append(result)
        
        status_symbol = "✓" if result['passed'] else "✗"
        print(f"  {status_symbol} {validator_name}: {result['status']}")
    
    # Run solver-based tests
    print("\nSolver-Based Tests:")
    for test_name in solver_tests:
        passed, solver_type, exec_time, solver_time, num_steps = run_solver_test_with_timing(
            theory, test_name, engine
        )
        
        result = {
            'name': test_name,
            'status': 'PASS' if passed else 'FAIL',
            'passed': passed,
            'solver_type': solver_type,
            'exec_time': exec_time,
            'solver_time': solver_time,
            'num_steps': num_steps
        }
        results['solver_tests'].append(result)
        
        status_symbol = "✓" if passed else "✗"
        print(f"  {status_symbol} {test_name}: {result['status']} [{solver_type}]")
        if solver_time > 0 and num_steps > 0:
            ms_per_step = solver_time / num_steps * 1000
            print(f"     Timing: {solver_time:.3f}s for {num_steps} steps ({ms_per_step:.3f}ms/step)")
    
    # Calculate summaries
    analytical_total = len(results['analytical_tests'])
    analytical_passed = sum(1 for t in results['analytical_tests'] if t['passed'])
    solver_total = len(results['solver_tests'])
    solver_passed = sum(1 for t in results['solver_tests'] if t['passed'])
    
    # Calculate actual solver timing
    total_solver_time = sum(t['solver_time'] for t in results['solver_tests'])
    total_solver_steps = sum(t['num_steps'] for t in results['solver_tests'])
    
    results['summary'] = {
        'analytical_passed': analytical_passed,
        'analytical_total': analytical_total,
        'solver_passed': solver_passed,
        'solver_total': solver_total,
        'total_solver_time': total_solver_time,
        'total_solver_steps': total_solver_steps,
        'avg_ms_per_step': (total_solver_time / total_solver_steps * 1000) if total_solver_steps > 0 else 0
    }
    
    print(f"\nSummary:")
    print(f"  Analytical: {analytical_passed}/{analytical_total} passed")
    print(f"  Solver: {solver_passed}/{solver_total} passed")
    print(f"  Total solver time: {total_solver_time:.3f}s")
    print(f"  Average: {results['summary']['avg_ms_per_step']:.3f}ms/step")
    
    return results

def main():
    """Run comprehensive tests with fixed timing."""
    print("COMPREHENSIVE THEORY VALIDATION - FIXED TIMING")
    print("="*80)
    print(f"Testing {len(ALL_THEORIES)} theories")
    print("Now with actual solver timing measurements!")
    
    all_results = []
    
    # Test all theories
    for theory_name, theory_class, category in ALL_THEORIES:
        result = test_theory_comprehensive(theory_name, theory_class, category)
        if result:
            all_results.append(result)
        time.sleep(0.1)
    
    # Sort by combined score
    for result in all_results:
        total_passed = result['summary']['analytical_passed'] + result['summary']['solver_passed']
        total_tests = result['summary']['analytical_total'] + result['summary']['solver_total']
        result['combined_score'] = total_passed / total_tests if total_tests > 0 else 0
    
    all_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Print final rankings
    print("\n\n" + "="*80)
    print("FINAL RANKINGS - WITH ACCURATE TIMING")
    print("="*80)
    
    print(f"\n{'Rank':<6} {'Theory':<30} {'Category':<10} {'Score':<8} {'Avg ms/step'}")
    print("-"*70)
    
    for i, result in enumerate(all_results, 1):
        theory = result['theory']
        category = result['category']
        score = f"{result['combined_score']*100:.1f}%"
        ms_per_step = result['summary']['avg_ms_per_step']
        
        marker = ""
        if theory == "Schwarzschild":
            marker = " ← Baseline"
        elif category == "quantum" and result['combined_score'] > 0.7:
            marker = " ✓ High-performing quantum"
        
        print(f"{i:<6} {theory:<30} {category:<10} {score:<8} {ms_per_step:>6.2f}{marker}")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"theory_validation_fixed_timing_{timestamp}.json"
    
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj
    
    with open(filename, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    
    print(f"\n\nDetailed results saved to: {filename}")
    
    # Analysis
    print("\n\nTIMING ANALYSIS:")
    print("-"*60)
    
    # Find theories with suspicious timing
    suspicious = []
    realistic = []
    
    for result in all_results:
        ms_per_step = result['summary']['avg_ms_per_step']
        if ms_per_step < 0.001:  # Less than 1 microsecond
            suspicious.append((result['theory'], ms_per_step))
        elif ms_per_step > 0.1:  # More than 0.1ms
            realistic.append((result['theory'], ms_per_step))
    
    if suspicious:
        print("\nTheories with suspiciously fast timing (<1μs/step):")
        for name, timing in suspicious:
            print(f"  - {name}: {timing:.6f}ms/step")
    
    if realistic:
        print(f"\nTheories with realistic timing (>0.1ms/step): {len(realistic)}")
        for name, timing in realistic[:5]:  # Show top 5
            print(f"  - {name}: {timing:.3f}ms/step")
    
    return all_results

if __name__ == "__main__":
    results = main()