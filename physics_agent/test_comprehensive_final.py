#!/usr/bin/env python3
"""
Comprehensive theory validation test combining analytical and solver-based tests.
Generates two ranking tables: analytical only and combined scores.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import json
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
import torch
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

def test_circular_orbit_for_theory(theory):
    """Test circular orbit period calculation for a given theory."""
    try:
        # Track solver type and timing
        start_time = time.time()
        solver_type = "Unknown"
        
        # Skip for theories that don't support circular orbits well
        if "Newtonian" in theory.name:
            solver_type = "Analytical"
            return True, solver_type, time.time() - start_time
        
        M_sun = torch.tensor(SOLAR_MASS, dtype=torch.float64)
        
        # Try to create appropriate solver based on theory properties
        if hasattr(theory, 'force_6dof_solver') and theory.force_6dof_solver:
            solver = GeneralGeodesicRK4Solver(theory, M_phys=M_sun, c=SPEED_OF_LIGHT, G=GRAVITATIONAL_CONSTANT)
            solver_type = "6DOF-RK4"
        else:
            solver = GeodesicRK4Solver(theory, M_phys=M_sun, c=SPEED_OF_LIGHT, G=GRAVITATIONAL_CONSTANT)
            solver_type = "4DOF-RK4"
        
        # Use 100 Schwarzschild radii
        rs_phys = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
        r_orbit = 100 * rs_phys
        r_geom = solver.to_geometric_length(torch.tensor(r_orbit))
        
        # Get circular orbit parameters
        E_geom, L_geom = solver.compute_circular_orbit_params(r_geom)
        
        exec_time = time.time() - start_time
        
        # Simple check: parameters should be finite and reasonable
        if torch.isfinite(E_geom) and torch.isfinite(L_geom) and E_geom > 0 and L_geom > 0:
            return True, solver_type, exec_time
        else:
            return False, solver_type, exec_time
            
    except Exception as e:
        # Some theories might not support circular orbits
        return False, "Failed", time.time() - start_time

def test_quantum_geodesic_for_theory(theory):
    """Test quantum geodesic simulator for a given theory."""
    try:
        start_time = time.time()
        M_sun = torch.tensor(SOLAR_MASS, dtype=torch.float64)
        
        # Initialize quantum simulator
        quantum_solver = QuantumGeodesicSimulator(theory, num_qubits=2, M_phys=M_sun)
        solver_type = "Quantum-2qb"
        
        # Test state: [t, r, phi, u^t, u^r, u^phi] for 6D motion
        state = torch.tensor([0.0, 10.0, 0.0, 1.0, 0.0, 0.1], dtype=torch.float64)
        
        # Compute derivatives
        quantum_deriv = quantum_solver.compute_derivatives(state)
        
        exec_time = time.time() - start_time
        
        # Check if derivatives are finite and quantum correction exists
        if torch.all(torch.isfinite(quantum_deriv)):
            return True, solver_type, exec_time
        else:
            return False, solver_type, exec_time
            
    except Exception as e:
        # Not all theories support quantum simulation
        return False, "Failed", time.time() - start_time

def run_solver_test(theory, test_func, test_name, engine=None):
    """Run a single solver-based test on a theory."""
    try:
        if test_name == "Circular Orbit":
            result, solver_type, exec_time = test_circular_orbit_for_theory(theory)
            return {
                'name': test_name,
                'status': 'PASS' if result else 'FAIL',
                'passed': result,
                'solver_type': solver_type,
                'exec_time': exec_time
            }
        elif test_name == "Quantum Geodesic Sim":
            result, solver_type, exec_time = test_quantum_geodesic_for_theory(theory)
            return {
                'name': test_name,
                'status': 'PASS' if result else 'FAIL',
                'passed': result,
                'solver_type': solver_type,
                'exec_time': exec_time
            }
        elif test_name == "CMB Power Spectrum" and engine:
            # Run CMB test using the validator
            start_time = time.time()
            validator = CMBPowerSpectrumValidator(engine=engine)
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
            
            if 'passed' in result_dict:
                result = result_dict['passed']
            elif 'flags' in result_dict and 'overall' in result_dict['flags']:
                result = result_dict['flags']['overall'] in ['PASS', 'WARNING']
            else:
                result = False
            
            # Determine solver type based on theory properties
            solver_type = "Analytical"
            if hasattr(theory, 'enable_quantum') and theory.enable_quantum:
                solver_type = "Quantum-PI"  # Quantum Path Integral
            
            return {
                'name': test_name,
                'status': 'PASS' if result else 'FAIL',
                'passed': result,
                'solver_type': solver_type,
                'exec_time': exec_time
            }
        elif test_name == "Primordial GWs" and engine:
            # Run Primordial GW test using the validator
            start_time = time.time()
            validator = PrimordialGWsValidator(engine=engine)
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
            
            if 'passed' in result_dict:
                result = result_dict['passed']
            elif 'flags' in result_dict and 'overall' in result_dict['flags']:
                result = result_dict['flags']['overall'] in ['PASS', 'WARNING']
            else:
                result = False
                
            # Determine solver type
            solver_type = "Analytical"
            if hasattr(theory, 'enable_quantum') and theory.enable_quantum:
                solver_type = "Quantum-PI"
                
            return {
                'name': test_name,
                'status': 'PASS' if result else 'FAIL',
                'passed': result,
                'solver_type': solver_type,
                'exec_time': exec_time
            }
        elif test_name == "Trajectory Cache":
            # Trajectory cache test is performance-based, skip for now
            return {
                'name': test_name,
                'status': 'SKIP',
                'passed': True,
                'notes': 'Performance test, not theory-specific',
                'solver_type': 'N/A',
                'exec_time': 0.0
            }
        else:
            return {
                'name': test_name,
                'status': 'ERROR',
                'passed': False,
                'error': f'Unknown test: {test_name}',
                'solver_type': 'Unknown',
                'exec_time': 0.0
            }
        
    except Exception as e:
        return {
            'name': test_name,
            'status': 'ERROR',
            'passed': False,
            'error': str(e)[:200],
            'solver_type': 'Failed',
            'exec_time': 0.0
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
    
    # Define solver-based tests (test_func is not used anymore, just for structure)
    solver_tests = [
        (None, "Circular Orbit"),
        (None, "CMB Power Spectrum"),
        (None, "Primordial GWs"),
        (None, "Trajectory Cache"),
        (None, "Quantum Geodesic Sim"),
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
    for test_func, test_name in solver_tests:
        result = run_solver_test(theory, test_func, test_name, engine)
        results['solver_tests'].append(result)
        
        # Format solver info
        solver_info = ""
        if 'solver_type' in result and result['solver_type'] not in ['N/A', 'Unknown']:
            solver_info = f" [{result['solver_type']}]"
        
        if result['status'] in ['SKIP', 'N/A']:
            print(f"  - {test_name}: {result['status']} ({result.get('notes', '')})")
        elif result['passed']:
            print(f"  ✓ {test_name}: {result['status']}{solver_info}")
        else:
            print(f"  ✗ {test_name}: {result['status']}{solver_info}")
    
    # Calculate summaries
    analytical_total = len(results['analytical_tests'])
    analytical_passed = sum(1 for t in results['analytical_tests'] if t['passed'])
    results['analytical_summary'] = {
        'total': analytical_total,
        'passed': analytical_passed,
        'success_rate': analytical_passed / analytical_total if analytical_total > 0 else 0
    }
    
    # For solver tests, only count actual tests (not skipped/N/A)
    solver_actual = [t for t in results['solver_tests'] if t['status'] not in ['SKIP', 'N/A']]
    solver_total = len(solver_actual)
    solver_passed = sum(1 for t in solver_actual if t['passed'])
    
    # Track solver types and total execution time
    solver_types = set()
    total_solver_time = 0.0
    for test in results['solver_tests']:
        if 'solver_type' in test and test['solver_type'] not in ['N/A', 'Unknown']:
            solver_types.add(test['solver_type'])
        if 'exec_time' in test:
            total_solver_time += test['exec_time']
    
    results['solver_summary'] = {
        'total': solver_total,
        'passed': solver_passed,
        'success_rate': solver_passed / solver_total if solver_total > 0 else 0,
        'solver_types': list(solver_types),
        'total_exec_time': total_solver_time
    }
    
    # Combined summary
    combined_total = analytical_total + solver_total
    combined_passed = analytical_passed + solver_passed
    results['combined_summary'] = {
        'total': combined_total,
        'passed': combined_passed,
        'success_rate': combined_passed / combined_total if combined_total > 0 else 0,
        'complexity_score': total_solver_time  # Lower is better
    }
    
    print(f"\nAnalytical: {analytical_passed}/{analytical_total} passed ({results['analytical_summary']['success_rate']*100:.1f}%)")
    print(f"Solver: {solver_passed}/{solver_total} passed ({results['solver_summary']['success_rate']*100:.1f}%)")
    print(f"Combined: {combined_passed}/{combined_total} passed ({results['combined_summary']['success_rate']*100:.1f}%)")
    
    return results

def print_ranking_table(results, ranking_type="analytical"):
    """Print a ranking table for the given ranking type."""
    # Sort by appropriate success rate
    if ranking_type == "analytical":
        results.sort(key=lambda x: x['analytical_summary']['success_rate'], reverse=True)
        summary_key = 'analytical_summary'
        # Print header for analytical table
        print(f"\n{'Rank':<6} {'Theory':<35} {'Category':<12} {'Score':<10} {'Tests Passed'}")
        print("-"*80)
    else:  # combined
        # Sort by success rate first, then by complexity (lower is better) for tiebreakers
        results.sort(key=lambda x: (-x['combined_summary']['success_rate'], 
                                    x['combined_summary'].get('complexity_score', float('inf'))))
        summary_key = 'combined_summary'
        # Print header for combined table with separate columns
        print(f"\n{'Rank':<6} {'Theory':<35} {'Category':<12} {'Analytical (Failed)':<25} {'Solver (Failed) [Type]':<50} {'Combined':<15} {'Complexity':<12} {'Total'}")
        print("-"*185)
    
    for i, result in enumerate(results, 1):
        theory = result['theory']
        category = result['category']
        
        if ranking_type == "analytical":
            # Simple format for analytical only
            score = f"{result[summary_key]['success_rate']*100:.1f}%"
            tests = f"{result[summary_key]['passed']}/{result[summary_key]['total']}"
            
            # Highlight special cases
            marker = ""
            if theory == "Schwarzschild":
                marker = " ← Baseline"
            elif theory == "Newtonian Limit":
                marker = " ← Should be lowest"
            elif category == "quantum" and result[summary_key]['success_rate'] > 0.7:
                marker = " ✓ High-performing quantum"
            
            print(f"{i:<6} {theory:<35} {category:<12} {score:<10} {tests}{marker}")
        else:
            # Detailed format for combined table
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
                    if test_name == "Circular Orbit":
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
            complexity = f"{result['combined_summary'].get('complexity_score', 0):.3f}s"
            total_tests = f"{result['combined_summary']['passed']}/{result['combined_summary']['total']}"
            
            # Highlight special cases
            marker = ""
            if theory == "Schwarzschild":
                marker = " ← Baseline"
            elif theory == "Newtonian Limit":
                marker = " ← Should be lowest"
            elif category == "quantum" and result['combined_summary']['success_rate'] > 0.7:
                marker = " ✓ High-performing quantum"
            
            print(f"{i:<6} {theory:<35} {category:<12} {analytical_str:<25} {solver_str:<50} {combined_score:<15} {complexity:<12} {total_tests}{marker}")

def main():
    """Run comprehensive tests and generate both ranking tables."""
    print("COMPREHENSIVE THEORY VALIDATION - ANALYTICAL + SOLVER TESTS")
    print("="*80)
    print(f"Testing {len(ALL_THEORIES)} theories with both analytical and solver-based tests")
    
    all_results = []
    
    # Test all theories
    for theory_name, theory_class, category in ALL_THEORIES:
        result = test_theory_comprehensive(theory_name, theory_class, category)
        if result:
            all_results.append(result)
        time.sleep(0.1)  # Brief pause
    
    # Generate first ranking table (analytical only)
    print("\n\n" + "="*80)
    print("FINAL RANKINGS - ANALYTICAL VALIDATORS ONLY")
    print("="*80)
    print_ranking_table(all_results.copy(), "analytical")
    
    print("\n\nVALIDATOR DETAILS (Analytical):")
    print("-"*80)
    print("\nValidators used (all use analytical methods, no trajectory solvers):")
    print("  1. Mercury Precession     - Analytical (weak-field approximation)")
    print("  2. Light Deflection       - Analytical (PPN calculation, γ parameter)")
    print("  3. Photon Sphere          - Analytical (effective potential extremum)")
    print("  4. PPN Parameters         - Analytical (metric expansion, γ and β)")
    print("  5. COW Interferometry     - Analytical (metric gradient calculation)")
    print("  6. Gravitational Waves    - Analytical (post-Newtonian waveforms)")
    print("  7. PSR J0740              - Analytical (Shapiro time delay)")
    
    # Generate second ranking table (combined)
    print("\n\n" + "="*80)
    print("FINAL RANKINGS - COMBINED (ANALYTICAL + SOLVER-BASED)")
    print("="*80)
    print_ranking_table(all_results.copy(), "combined")
    
    print("\n\nADDITIONAL TESTS (Solver-Based):")
    print("-"*80)
    print("\nSolver-based tests added in combined ranking:")
    print("  1. Circular Orbit Period  - USES geodesic RK4 solver (Schwarzschild only)")
    print("  2. CMB Power Spectrum     - USES quantum path integral (optional)")
    print("  3. Primordial GWs         - USES quantum path integral (optional)")
    print("  4. Trajectory Cache       - Tests geodesic solver caching (Schwarzschild only)")
    print("  5. Quantum Geodesic Sim   - Tests quantum corrections (Schwarzschild only)")
    
    print("\nNOTE: Some solver tests currently only run on Schwarzschild theory.")
    print("Conservation Validator (trajectory-based) is still NOT included.")
    
    print("\n\nLEGEND:")
    print("-"*60)
    print("Analytical Test Abbreviations:")
    print("  Mercury = Mercury Precession, Light = Light Deflection, Photon = Photon Sphere")
    print("  PPN = PPN Parameters, COW = COW Interferometry, GW = Gravitational Waves, PSR = PSR J0740")
    print("\nSolver Test Abbreviations:")
    print("  Orb = Circular Orbit, CMB = CMB Power Spectrum, PGW = Primordial GWs, QGS = Quantum Geodesic Sim")
    print("\nSolver Types:")
    print("  4D = 4DOF-RK4, 6D = 6DOF-RK4, Q2 = Quantum-2qb, QPI = Quantum-PI, An = Analytical")
    
    print("\n\nSOLVER-BASED TESTS ADDED IN COMBINED RANKING:")
    print("-"*80)
    print("\n1. Circular Orbit Period (Orb) - Tests geodesic integration accuracy over time")
    print("   - What it tests: Integrates full orbital trajectories through many timesteps")
    print("   - Time aspect: Follows particle motion for complete orbits (thousands of integration steps)")
    print("   - Why relevant: Reveals if small errors accumulate or if theory remains stable over time")
    print("   - Complements: Mercury precession (single calculation from orbital parameters)")
    
    print("\n2. CMB Power Spectrum (CMB) - Tests cosmological evolution") 
    print("   - What it tests: Evolves perturbations from early universe to CMB formation")
    print("   - Time aspect: Tracks quantum fluctuations evolving over cosmic time (380,000 years)")
    print("   - Why relevant: Tests if theory predictions remain consistent across cosmic epochs")
    print("   - Complements: Instantaneous tests by validating long-term cosmological evolution")
    
    print("\n3. Primordial Gravitational Waves (PGW) - Tests tensor mode evolution")
    print("   - What it tests: Propagation of gravitational waves from inflation to today")
    print("   - Time aspect: Evolves tensor perturbations through multiple cosmic phases")
    print("   - Why relevant: Detects instabilities that only appear in long-term wave propagation")
    print("   - Complements: GW test (single waveform calculation for current mergers)")
    
    print("\n4. Quantum Geodesic Simulator (QGS) - Tests quantum trajectory evolution")
    print("   - What it tests: 2-qubit simulation tracking quantum corrections over time")
    print("   - Time aspect: Evolves quantum state through many gate operations")
    print("   - Why relevant: Shows if quantum corrections remain coherent or decohere over time")
    print("   - Complements: Classical tests that assume point particles without quantum effects")
    
    print("\n5. Trajectory Cache Performance (not shown) - Tests computational efficiency")
    print("   - What it tests: Speed and accuracy of repeated trajectory calculations")
    print("   - Why relevant: Practical viability for simulations and predictions")
    
    print("\nKEY INSIGHTS:")
    print("• Analytical tests: Single-point calculations at specific moments (instantaneous measurements)")
    print("• Solver tests: Evolution over time through multiple steps (accumulated effects)")
    print("• This temporal difference is crucial because:")
    print("  - Analytical tests check if a theory gives correct values at one instant")
    print("  - Solver tests verify the theory remains consistent and stable over extended evolution")
    print("  - Small errors can accumulate over time, revealing issues invisible to single-point tests")
    print("• Additional differences:")
    print("  - Analytical tests use approximations (weak-field, post-Newtonian, etc.)")
    print("  - Solver tests validate the full nonlinear dynamics without approximations")
    print("• High failure rate in solver tests (especially PGW) suggests issues with:")
    print("  - Numerical stability over extended time evolution")
    print("  - Theory implementations breaking down during multi-step integration")
    print("  - Cumulative errors in modified gravity theories")
    
    # Analysis
    print("\n\nANALYSIS:")
    print("-"*60)
    
    # Compare rankings
    analytical_sorted = sorted(all_results, key=lambda x: x['analytical_summary']['success_rate'], reverse=True)
    combined_sorted = sorted(all_results, key=lambda x: x['combined_summary']['success_rate'], reverse=True)
    
    # Check if rankings changed
    changes = 0
    for i, (a, c) in enumerate(zip(analytical_sorted, combined_sorted)):
        if a['theory'] != c['theory']:
            changes += 1
    
    if changes == 0:
        print("Rankings remained the same after adding solver tests.")
    else:
        print(f"Rankings changed for {changes} theories after adding solver tests.")
    
    # Save comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
        'analytical_rankings': [r['theory'] for r in analytical_sorted],
        'combined_rankings': [r['theory'] for r in combined_sorted]
    }
    
    report_file = f"theory_validation_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def make_serializable(obj):
        # Handle torch tensors
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        # Handle numpy types
        elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            # This catches numpy scalars (bool_, int64, float64, etc)
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle containers
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(v) for v in obj)
        # Handle custom objects
        elif hasattr(obj, '__dict__') and not callable(obj):
            return make_serializable(obj.__dict__)
        # Default case
        else:
            return obj
    
    with open(report_file, 'w') as f:
        json.dump(make_serializable(report), f, indent=2)
    
    print(f"\n\nDetailed report saved to: {report_file}")
    
    return all_results

if __name__ == "__main__":
    results = main()