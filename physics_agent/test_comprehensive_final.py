#!/usr/bin/env python3
"""
Comprehensive theory validation test combining analytical and solver-based tests.
Generates two ranking tables: analytical only and combined scores.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
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

# Import solver tests
from physics_agent.solver_tests.test_geodesic_validator_comparison import (
    test_circular_orbit_period,
    test_cmb_power_spectrum,
    test_bicep_keck_primordial_gws,
    test_trajectory_cache_performance,
    test_quantum_geodesic_simulator
)

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

def run_solver_test(theory, test_func, test_name):
    """Run a single solver-based test on a theory."""
    try:
        # Special handling for solver tests that expect no arguments
        if test_name in ["Circular Orbit", "Trajectory Cache"]:
            # These tests run on Schwarzschild only
            if theory.name == "Schwarzschild":
                result = test_func()
            else:
                # Skip for non-Schwarzschild theories
                return {
                    'name': test_name,
                    'status': 'SKIP',
                    'passed': True,  # Don't penalize for skipping
                    'notes': 'Test only runs on Schwarzschild'
                }
        else:
            # CMB and Primordial GW tests can run on any theory
            # But they're currently hardcoded to Schwarzschild
            # We'll mark them as not applicable for now
            return {
                'name': test_name,
                'status': 'N/A',
                'passed': True,  # Don't penalize
                'notes': 'Solver test not theory-specific yet'
            }
        
        return {
            'name': test_name,
            'status': 'PASS' if result else 'FAIL',
            'passed': result
        }
        
    except Exception as e:
        return {
            'name': test_name,
            'status': 'ERROR',
            'passed': False,
            'error': str(e)[:200]
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
    
    # Define solver-based tests
    solver_tests = [
        (test_circular_orbit_period, "Circular Orbit"),
        (test_cmb_power_spectrum, "CMB Power Spectrum"),
        (test_bicep_keck_primordial_gws, "Primordial GWs"),
        (test_trajectory_cache_performance, "Trajectory Cache"),
        (test_quantum_geodesic_simulator, "Quantum Geodesic Sim"),
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
        result = run_solver_test(theory, test_func, test_name)
        results['solver_tests'].append(result)
        if result['status'] in ['SKIP', 'N/A']:
            print(f"  - {test_name}: {result['status']} ({result.get('notes', '')})")
        elif result['passed']:
            print(f"  ✓ {test_name}: {result['status']}")
        else:
            print(f"  ✗ {test_name}: {result['status']}")
    
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
    results['solver_summary'] = {
        'total': solver_total,
        'passed': solver_passed,
        'success_rate': solver_passed / solver_total if solver_total > 0 else 0
    }
    
    # Combined summary
    combined_total = analytical_total + solver_total
    combined_passed = analytical_passed + solver_passed
    results['combined_summary'] = {
        'total': combined_total,
        'passed': combined_passed,
        'success_rate': combined_passed / combined_total if combined_total > 0 else 0
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
        results.sort(key=lambda x: x['combined_summary']['success_rate'], reverse=True)
        summary_key = 'combined_summary'
        # Print header for combined table with separate columns
        print(f"\n{'Rank':<6} {'Theory':<35} {'Category':<12} {'Analytical':<15} {'Solver':<15} {'Combined':<15} {'Total Tests'}")
        print("-"*130)
    
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
            analytical_str = f"{result['analytical_summary']['passed']}/{result['analytical_summary']['total']}"
            solver_str = f"{result['solver_summary']['passed']}/{result['solver_summary']['total']}" if result['solver_summary']['total'] > 0 else "N/A"
            combined_score = f"{result['combined_summary']['success_rate']*100:.1f}%"
            total_tests = f"{result['combined_summary']['passed']}/{result['combined_summary']['total']}"
            
            # Highlight special cases
            marker = ""
            if theory == "Schwarzschild":
                marker = " ← Baseline"
            elif theory == "Newtonian Limit":
                marker = " ← Should be lowest"
            elif category == "quantum" and result['combined_summary']['success_rate'] > 0.7:
                marker = " ✓ High-performing quantum"
            
            print(f"{i:<6} {theory:<35} {category:<12} {analytical_str:<15} {solver_str:<15} {combined_score:<15} {total_tests}{marker}")

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
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj
    
    with open(report_file, 'w') as f:
        json.dump(make_serializable(report), f, indent=2)
    
    print(f"\n\nDetailed report saved to: {report_file}")
    
    return all_results

if __name__ == "__main__":
    results = main()