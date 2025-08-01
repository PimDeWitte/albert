#!/usr/bin/env python3
"""
Final comprehensive theory validation test.
Tests all theories and ranks them by performance.
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

# Import validators directly
from physics_agent.validations.mercury_precession_validator import MercuryPrecessionValidator
from physics_agent.validations.light_deflection_validator import LightDeflectionValidator
from physics_agent.validations.photon_sphere_validator import PhotonSphereValidator
from physics_agent.validations.ppn_validator import PpnValidator
from physics_agent.validations.cow_interferometry_validator import COWInterferometryValidator
from physics_agent.validations.gw_validator import GwValidator
from physics_agent.validations.psr_j0740_validator import PsrJ0740Validator

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
            # ValidationResult object
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
            # Some validators put loss in results
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

def test_theory(theory_name, theory_class, category):
    """Test a single theory with all validators."""
    print(f"\n{'='*60}")
    print(f"Testing: {theory_name} [{category}]")
    print(f"{'='*60}")
    
    try:
        # Initialize theory with default parameters
        if theory_name == "Kerr":
            theory = theory_class(a=0.0)  # Non-rotating for simpler testing
        elif theory_name == "Kerr-Newman":
            theory = theory_class(a=0.0, Q=0.0)  # Non-rotating, uncharged
        else:
            theory = theory_class()
        print(f"Initialized: {theory.name}")
    except Exception as e:
        print(f"ERROR: Failed to initialize - {e}")
        return None
    
    engine = TheoryEngine()
    
    # Define validators to test
    validators = [
        (MercuryPrecessionValidator, "Mercury Precession"),
        (LightDeflectionValidator, "Light Deflection"),
        (PhotonSphereValidator, "Photon Sphere"),
        (PpnValidator, "PPN Parameters"),
        (COWInterferometryValidator, "COW Interferometry"),
        (GwValidator, "Gravitational Waves"),
        (PsrJ0740Validator, "PSR J0740"),
    ]
    
    results = {
        'theory': theory_name,
        'category': category,
        'tests': []
    }
    
    # Run each validator
    for validator_class, validator_name in validators:
        print(f"\n{validator_name}:")
        result = run_validator_test(theory, validator_class, validator_name, engine)
        results['tests'].append(result)
        
        if result['passed']:
            print(f"  ✓ {result['status']}")
        else:
            print(f"  ✗ {result['status']}")
        
        if result.get('loss') is not None:
            print(f"    Loss: {result['loss']:.6f}")
        if result.get('error_percent') is not None:
            print(f"    Error: {result['error_percent']:.2f}%")
        if result.get('error'):
            print(f"    Error: {result['error']}")
    
    # Calculate summary
    total_tests = len(results['tests'])
    passed_tests = sum(1 for t in results['tests'] if t['passed'])
    results['summary'] = {
        'total': total_tests,
        'passed': passed_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0
    }
    
    print(f"\nSummary: {passed_tests}/{total_tests} passed ({results['summary']['success_rate']*100:.1f}%)")
    
    return results

def main():
    """Run all tests and generate comprehensive report."""
    print("COMPREHENSIVE THEORY VALIDATION - FINAL TEST")
    print("="*80)
    print(f"Testing {len(ALL_THEORIES)} theories")
    print("Expected order: Quantum theories > Schwarzschild > Newtonian Limit")
    
    all_results = []
    
    # Test all theories
    for theory_name, theory_class, category in ALL_THEORIES:
        result = test_theory(theory_name, theory_class, category)
        if result:
            all_results.append(result)
        time.sleep(0.1)  # Brief pause
    
    # Sort by success rate (descending)
    all_results.sort(key=lambda x: x['summary']['success_rate'], reverse=True)
    
    # Generate report
    print("\n\n" + "="*80)
    print("FINAL RANKINGS")
    print("="*80)
    
    print(f"\n{'Rank':<6} {'Theory':<35} {'Category':<12} {'Score':<10} {'Tests Passed'}")
    print("-"*80)
    
    for i, result in enumerate(all_results, 1):
        theory = result['theory']
        category = result['category']
        score = f"{result['summary']['success_rate']*100:.1f}%"
        tests = f"{result['summary']['passed']}/{result['summary']['total']}"
        
        # Highlight special cases
        marker = ""
        if theory == "Schwarzschild":
            marker = " ← Baseline"
        elif theory == "Newtonian Limit":
            marker = " ← Should be lowest"
        elif category == "quantum" and result['summary']['success_rate'] > 0.7:
            marker = " ✓ High-performing quantum"
        
        print(f"{i:<6} {theory:<35} {category:<12} {score:<10} {tests}{marker}")
    
    # Validator and Solver Information
    print("\n\nVALIDATOR DETAILS:")
    print("-"*80)
    print("\nValidators used in these tests (all use analytical methods, no trajectory solvers):")
    print("  1. Mercury Precession     - Analytical (weak-field approximation)")
    print("  2. Light Deflection       - Analytical (PPN calculation, γ parameter)")
    print("  3. Photon Sphere          - Analytical (effective potential extremum)")
    print("  4. PPN Parameters         - Analytical (metric expansion, γ and β)")
    print("  5. COW Interferometry     - Analytical (metric gradient calculation)")
    print("  6. Gravitational Waves    - Analytical (post-Newtonian waveforms)")
    print("  7. PSR J0740              - Analytical (Shapiro time delay)")
    
    print("\nNOTE: No geodesic trajectory integration is used in these validators.")
    print("Conservation Validator (which uses trajectories) is NOT included in this test suite.")
    
    print("\nAdditional solver-based tests available in test_geodesic_validator_comparison.py:")
    print("  • Circular Orbit Period   - USES geodesic RK4 solver")
    print("  • CMB Power Spectrum      - USES quantum path integral (optional)")
    print("  • Primordial GWs          - USES quantum path integral (optional)")
    print("  • Trajectory Cache Test   - Tests geodesic solver caching")
    print("  • Quantum Geodesic Sim    - Tests quantum corrections to trajectories")
    
    # Analysis
    print("\n\nANALYSIS:")
    print("-"*60)
    
    # Check if quantum theories are at top
    top_5 = all_results[:5]
    quantum_in_top_5 = sum(1 for r in top_5 if r['category'] == 'quantum')
    print(f"\nQuantum theories in top 5: {quantum_in_top_5}/5")
    
    # Find specific theories
    schwarzschild_rank = next((i for i, r in enumerate(all_results, 1) if r['theory'] == 'Schwarzschild'), None)
    newtonian_rank = next((i for i, r in enumerate(all_results, 1) if r['theory'] == 'Newtonian Limit'), None)
    
    print(f"Schwarzschild rank: #{schwarzschild_rank}")
    print(f"Newtonian Limit rank: #{newtonian_rank}")
    
    # Check expectations
    print("\nExpectation Check:")
    
    # Best quantum theory
    best_quantum = next((r for r in all_results if r['category'] == 'quantum'), None)
    if best_quantum and schwarzschild_rank:
        schwarzschild_score = next((r['summary']['success_rate'] for r in all_results if r['theory'] == 'Schwarzschild'), 0)
        if best_quantum['summary']['success_rate'] > schwarzschild_score:
            print(f"✓ Best quantum theory ({best_quantum['theory']}) outperforms Schwarzschild")
        else:
            print(f"✗ No quantum theory outperforms Schwarzschild yet")
    
    # Newtonian should be at bottom
    if newtonian_rank and newtonian_rank == len(all_results):
        print("✓ Newtonian Limit is at the bottom as expected")
    elif newtonian_rank and newtonian_rank > len(all_results) - 3:
        print("~ Newtonian Limit is near bottom")
    else:
        print("✗ Newtonian Limit is not at bottom")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_theories': len(all_results),
            'rankings': [(r['theory'], r['category'], r['summary']['success_rate']) for r in all_results]
        },
        'full_results': all_results
    }
    
    report_file = f"theory_validation_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
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
    
    with open(report_file, 'w') as f:
        json.dump(make_serializable(report), f, indent=2)
    
    print(f"\n\nDetailed report saved to: {report_file}")
    
    return all_results

if __name__ == "__main__":
    results = main()