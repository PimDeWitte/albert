#!/usr/bin/env python3
"""
Test all theories against the comprehensive validator test suite.
This will run each theory through the tests from test_geodesic_validator_comparison.py
and generate a summary report.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Import all theories
from physics_agent.theories.newtonian_limit.theory import NewtonianLimit
# Kerr and KerrNewman are imported from baselines via their theory.py files
from physics_agent.theories.defaults.baselines.kerr import Kerr
from physics_agent.theories.defaults.baselines.kerr_newman import KerrNewman
from physics_agent.theories.yukawa.theory import Yukawa
from physics_agent.theories.einstein_teleparallel.theory import EinsteinTeleparallel
from physics_agent.theories.spinor_conformal.theory import SpinorConformal
from physics_agent.theories.stochastic_noise.theory import StochasticNoise
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
from physics_agent.theories.asymptotic_safety.theory import AsymptoticSafetyTheory
from physics_agent.theories.gauge_gravity.theory import GaugeGravity
from physics_agent.theories.emergent.theory import Emergent
from physics_agent.theories.string.theory import StringTheory
from physics_agent.theories.entropic_gravity.theory import EntropicGravity
from physics_agent.theories.surfaceology.theory import Surfaceology
from physics_agent.theories.log_corrected.theory import LogCorrected
from physics_agent.theories.fractal.theory import Fractal
from physics_agent.theories.alena_tensor.theory import AlenaTensor
from physics_agent.theories.ugm.theory import UnifiedGaugeModel
from physics_agent.theories.einstein_asymmetric.theory import EinsteinAsymmetric
from physics_agent.theories.post_quantum_gravity.theory import PostQuantumGravityTheory
from physics_agent.theories.phase_transition.theory import PhaseTransition
from physics_agent.theories.einstein_regularised_core.theory import EinsteinRegularisedCore
from physics_agent.theories.non_commutative_geometry.theory import NonCommutativeGeometry
from physics_agent.theories.twistor_theory.theory import TwistorTheory
from physics_agent.theories.aalto_gauge_gravity.theory import AaltoGaugeGravity
from physics_agent.theories.loop_quantum_gravity.theory import LoopQuantumGravity
from physics_agent.theories.causal_dynamical_triangulations.theory import CausalDynamicalTriangulations

# Import test framework
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.validations.mercury_precession_validator import MercuryPrecessionValidator
from physics_agent.validations.light_deflection_validator import LightDeflectionValidator
from physics_agent.validations.photon_sphere_validator import PhotonSphereValidator
from physics_agent.validations.ppn_validator import PpnValidator
from physics_agent.validations.cow_interferometry_validator import COWInterferometryValidator
from physics_agent.validations.gw_validator import GwValidator
from physics_agent.validations.cmb_power_spectrum_validator import CMBPowerSpectrumValidator
from physics_agent.validations.primordial_gws_validator import PrimordialGWsValidator
from physics_agent.validations.psr_j0740_validator import PsrJ0740Validator

# Also test Schwarzschild as baseline
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild

# Categories of theories
ACCEPTED_THEORIES = [
    ("Newtonian Limit", NewtonianLimit),
    ("Kerr", Kerr),
    ("Kerr-Newman", KerrNewman),
    ("Yukawa", Yukawa),
    ("Einstein Teleparallel", EinsteinTeleparallel),
    ("Spinor Conformal", SpinorConformal),
]

IN_PROGRESS_THEORIES = [
    ("Einstein Asymmetric", EinsteinAsymmetric),
    ("Einstein Regularised Core", EinsteinRegularisedCore),
    ("Stochastic Noise", StochasticNoise),
    ("Quantum Corrected", QuantumCorrected),
    ("Post Quantum Gravity", PostQuantumGravityTheory),
    ("Log Corrected", LogCorrected),
    ("Phase Transition", PhaseTransition),
    ("Asymptotic Safety", AsymptoticSafetyTheory),
    ("Gauge Gravity", GaugeGravity),
    ("Surfaceology", Surfaceology),
    ("String Theory", StringTheory),
    ("Emergent", Emergent),
    ("Entropic Gravity", EntropicGravity),
    ("Alena Tensor", AlenaTensor),
    ("UGM", UnifiedGaugeModel),
    ("Non-Commutative Geometry", NonCommutativeGeometry),
    ("Twistor Theory", TwistorTheory),
    ("Aalto Gauge Gravity", AaltoGaugeGravity),
    ("Loop Quantum Gravity", LoopQuantumGravity),
    ("Causal Dynamical Triangulations", CausalDynamicalTriangulations),
    ("Fractal Gravity", Fractal),
]

BASELINE_THEORIES = [
    ("Schwarzschild", Schwarzschild),
]

# Test validators
VALIDATORS = [
    ("Mercury Precession", MercuryPrecessionValidator, "Classical test of GR"),
    ("Light Deflection", LightDeflectionValidator, "Solar light bending"),
    ("Photon Sphere", PhotonSphereValidator, "Black hole photon orbit"),
    ("PPN Parameters", PpnValidator, "Parameterized post-Newtonian"),
    ("COW Interferometry", COWInterferometryValidator, "Quantum gravity test"),
    ("Gravitational Waves", GwValidator, "LIGO/Virgo waveforms"),
    ("CMB Power Spectrum", CMBPowerSpectrumValidator, "Cosmological test"),
    ("Primordial GWs", PrimordialGWsValidator, "Early universe"),
    ("PSR J0740", PsrJ0740Validator, "Pulsar timing"),
]


def test_theory(theory_name: str, theory_class, engine: TheoryEngine) -> Dict:
    """Test a single theory against all validators."""
    print(f"\n{'='*60}")
    print(f"Testing: {theory_name}")
    print(f"{'='*60}")
    
    results = {
        'theory_name': theory_name,
        'tests': {},
        'summary': {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0
        }
    }
    
    try:
        # Initialize theory with default parameters
        theory = theory_class()
        print(f"Initialized: {theory.name}")
    except Exception as e:
        print(f"ERROR: Failed to initialize theory: {e}")
        results['initialization_error'] = str(e)
        return results
    
    # Run each validator
    for val_name, val_class, description in VALIDATORS:
        print(f"\n{val_name}: {description}")
        
        start_time = time.time()
        try:
            validator = val_class(engine=engine)
            result = validator.validate(theory, verbose=False)
            elapsed = time.time() - start_time
            
            # Extract key information
            # Handle ValidationResult objects and dictionaries
            if hasattr(result, 'passed'):
                passed = result.passed
                status = 'PASS' if passed else 'FAIL'
            elif hasattr(result, '__dict__') and 'flags' in result.__dict__ and 'overall' in result.__dict__['flags']:
                # For objects with flags attribute
                status = result.__dict__['flags']['overall']
                passed = status in ['PASS', 'WARNING']
            elif isinstance(result, dict) and 'flags' in result and 'overall' in result['flags']:
                # For dictionary results
                status = result['flags']['overall']
                passed = status in ['PASS', 'WARNING']
            else:
                passed = False
                status = 'UNKNOWN'
            
            # Get loss value
            loss = None
            if hasattr(result, 'loss'):
                loss = result.loss
            elif hasattr(result, '__dict__') and 'loss' in result.__dict__:
                loss = result.__dict__['loss']
            elif isinstance(result, dict) and 'loss' in result:
                loss = result['loss']
            
            # Get error percentage if available
            error_pct = None
            if hasattr(result, 'error_percent'):
                error_pct = result.error_percent
            elif hasattr(result, '__dict__') and 'error_percent' in result.__dict__:
                error_pct = result.__dict__['error_percent']
            
            results['tests'][val_name] = {
                'passed': passed,
                'status': status,
                'loss': loss,
                'error_percent': error_pct,
                'time': elapsed
            }
            
            results['summary']['total'] += 1
            if passed:
                results['summary']['passed'] += 1
                print(f"  ✓ {status} (time: {elapsed:.3f}s)")
            else:
                results['summary']['failed'] += 1
                print(f"  ✗ {status} (time: {elapsed:.3f}s)")
            
            if loss is not None:
                print(f"    Loss: {loss:.6f}")
            if error_pct is not None:
                print(f"    Error: {error_pct:.2f}%")
                
        except Exception as e:
            print(f"  ⚠ ERROR: {str(e)[:100]}...")
            results['tests'][val_name] = {
                'passed': False,
                'status': 'ERROR',
                'error': str(e),
                'time': time.time() - start_time
            }
            results['summary']['errors'] += 1
            results['summary']['total'] += 1
    
    # Calculate success rate
    if results['summary']['total'] > 0:
        results['summary']['success_rate'] = results['summary']['passed'] / results['summary']['total']
    else:
        results['summary']['success_rate'] = 0.0
    
    print(f"\nSummary: {results['summary']['passed']}/{results['summary']['total']} passed " +
          f"({results['summary']['success_rate']*100:.1f}%)")
    
    return results


def generate_report(all_results: List[Dict], output_file: str = "theory_test_report.json"):
    """Generate a comprehensive test report."""
    
    # Group results by category
    accepted_results = [r for r in all_results if r['theory_name'] in [t[0] for t in ACCEPTED_THEORIES]]
    in_progress_results = [r for r in all_results if r['theory_name'] in [t[0] for t in IN_PROGRESS_THEORIES]]
    baseline_results = [r for r in all_results if r['theory_name'] in [t[0] for t in BASELINE_THEORIES]]
    
    # Summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'categories': {
            'accepted': {
                'theories': [r['theory_name'] for r in accepted_results],
                'avg_success_rate': sum(r['summary']['success_rate'] for r in accepted_results) / len(accepted_results) if accepted_results else 0
            },
            'in_progress': {
                'theories': [r['theory_name'] for r in in_progress_results],
                'avg_success_rate': sum(r['summary']['success_rate'] for r in in_progress_results) / len(in_progress_results) if in_progress_results else 0
            },
            'baseline': {
                'theories': [r['theory_name'] for r in baseline_results],
                'avg_success_rate': sum(r['summary']['success_rate'] for r in baseline_results) / len(baseline_results) if baseline_results else 0
            }
        },
        'all_results': all_results
    }
    
    # Save JSON report
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    print(f"\nACCEPTED THEORIES (Expected: Near perfect scores)")
    print("-"*60)
    for result in accepted_results:
        print(f"{result['theory_name']:<30} {result['summary']['passed']:>2}/{result['summary']['total']:>2} " +
              f"({result['summary']['success_rate']*100:>5.1f}%)")
    
    print(f"\nIN PROGRESS THEORIES")
    print("-"*60)
    for result in in_progress_results:
        print(f"{result['theory_name']:<30} {result['summary']['passed']:>2}/{result['summary']['total']:>2} " +
              f"({result['summary']['success_rate']*100:>5.1f}%)")
    
    print(f"\nBASELINE THEORIES")
    print("-"*60)
    for result in baseline_results:
        print(f"{result['theory_name']:<30} {result['summary']['passed']:>2}/{result['summary']['total']:>2} " +
              f"({result['summary']['success_rate']*100:>5.1f}%)")
    
    # Test-by-test breakdown
    print(f"\n\nTEST-BY-TEST BREAKDOWN")
    print("="*80)
    
    # Create a matrix view
    test_names = [v[0] for v in VALIDATORS]
    
    # Header
    print(f"{'Theory':<25}", end='')
    for i, test in enumerate(test_names):
        print(f" {i+1:<3}", end='')
    print("  Success")
    print("-"*80)
    
    # Legend
    print("\nTest Legend:")
    for i, (test, _, desc) in enumerate(VALIDATORS):
        print(f"  {i+1}. {test}: {desc}")
    
    print(f"\n{'Theory':<25}", end='')
    for i in range(len(test_names)):
        print(f" {i+1:<3}", end='')
    print("  Success")
    print("-"*80)
    
    # Results matrix
    for result in all_results:
        print(f"{result['theory_name']:<25}", end='')
        for test in test_names:
            if test in result['tests']:
                if result['tests'][test]['passed']:
                    print(" ✓  ", end='')
                elif result['tests'][test]['status'] == 'ERROR':
                    print(" E  ", end='')
                else:
                    print(" ✗  ", end='')
            else:
                print(" -  ", end='')
        print(f"  {result['summary']['success_rate']*100:>5.1f}%")
    
    return summary


def main():
    """Run all tests."""
    print("="*80)
    print("COMPREHENSIVE THEORY VALIDATION TEST SUITE")
    print("="*80)
    print(f"\nTesting {len(ACCEPTED_THEORIES)} accepted theories")
    print(f"Testing {len(IN_PROGRESS_THEORIES)} in-progress theories")
    print(f"Testing {len(BASELINE_THEORIES)} baseline theories")
    print(f"Using {len(VALIDATORS)} validators")
    
    # Initialize engine
    engine = TheoryEngine()
    
    # Test all theories
    all_results = []
    
    # Test baseline first
    print("\n\nTESTING BASELINE THEORIES")
    print("="*80)
    for name, theory_class in BASELINE_THEORIES:
        results = test_theory(name, theory_class, engine)
        all_results.append(results)
    
    # Test accepted theories
    print("\n\nTESTING ACCEPTED THEORIES")
    print("="*80)
    for name, theory_class in ACCEPTED_THEORIES:
        results = test_theory(name, theory_class, engine)
        all_results.append(results)
    
    # Test in-progress theories
    print("\n\nTESTING IN-PROGRESS THEORIES")
    print("="*80)
    for name, theory_class in IN_PROGRESS_THEORIES:
        results = test_theory(name, theory_class, engine)
        all_results.append(results)
    
    # Generate report
    report_file = f"theory_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = generate_report(all_results, report_file)
    
    print(f"\n\nDetailed report saved to: {report_file}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    accepted_avg = summary['categories']['accepted']['avg_success_rate']
    in_progress_avg = summary['categories']['in_progress']['avg_success_rate']
    baseline_avg = summary['categories']['baseline']['avg_success_rate']
    
    print(f"\nAccepted theories average: {accepted_avg*100:.1f}%")
    print(f"In-progress theories average: {in_progress_avg*100:.1f}%")
    print(f"Baseline (Schwarzschild) average: {baseline_avg*100:.1f}%")
    
    # Check expectations
    print("\nExpectation Check:")
    if accepted_avg > 0.8:
        print("✓ Accepted theories performing well (>80%)")
    else:
        print("✗ Accepted theories underperforming (<80%)")
    
    if baseline_avg > 0.7:
        print("✓ Schwarzschild baseline performing as expected")
    else:
        print("✗ Schwarzschild baseline has issues")
    
    return all_results


if __name__ == "__main__":
    results = main()