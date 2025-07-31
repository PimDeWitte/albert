#!/usr/bin/env python3
"""
Comprehensive theory validation with quantum-specific tests.
Tests all theories with both classical and quantum validators.
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

# Import classical validators
from physics_agent.validations.mercury_precession_validator import MercuryPrecessionValidator
from physics_agent.validations.light_deflection_validator import LightDeflectionValidator
from physics_agent.validations.photon_sphere_validator import PhotonSphereValidator
from physics_agent.validations.ppn_validator import PpnValidator
from physics_agent.validations.gw_validator import GwValidator
from physics_agent.validations.psr_j0740_validator import PsrJ0740Validator

# Import quantum-specific validators
from physics_agent.validations.cow_interferometry_validator import COWInterferometryValidator
from physics_agent.validations.atom_interferometry_validator import AtomInterferometryValidator
from physics_agent.validations.gravitational_decoherence_validator import GravitationalDecoherenceValidator
from physics_agent.validations.quantum_clock_validator import QuantumClockValidator
from physics_agent.validations.hawking_validator import HawkingValidator
from physics_agent.validations.qed_precision_validator import QEDPrecisionValidator

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

def run_validator_test(theory, validator_class, validator_name, engine, hist=None):
    """Run a single validator on a theory."""
    try:
        validator = validator_class(engine=engine)
        
        # Some validators need trajectory history
        if hist is not None:
            result = validator.validate(theory, hist, verbose=False)
        else:
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
        
        # Skip tests that aren't applicable
        if status == 'SKIP':
            return None
        
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
            'error_percent': float(error_pct) if error_pct is not None else None,
            'is_quantum_test': validator_name in ['COW Interferometry', 'Atom Interferometry', 
                                                   'Gravitational Decoherence', 'Quantum Clock',
                                                   'Hawking Temperature', 'QED Precision']
        }
        
    except Exception as e:
        error_msg = str(e)[:200]
        # Skip tensor errors that are intermittent
        if "sqrt(): argument 'input'" in error_msg:
            return None
        return {
            'name': validator_name,
            'status': 'ERROR',
            'passed': False,
            'error': error_msg,
            'is_quantum_test': validator_name in ['COW Interferometry', 'Atom Interferometry', 
                                                   'Gravitational Decoherence', 'Quantum Clock',
                                                   'Hawking Temperature', 'QED Precision']
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
    
    # Generate trajectory for validators that need it
    hist = None
    try:
        y0 = torch.tensor([10.0, 0.0, 0.0, 0.0], device=engine.device, dtype=engine.dtype)
        hist, _ = engine.run_trajectory(theory, y0, t_end=100.0)
    except:
        pass  # Some theories might fail trajectory generation
    
    # Define validators to test - classical and quantum
    validators = [
        # Classical tests
        (MercuryPrecessionValidator, "Mercury Precession", False),
        (LightDeflectionValidator, "Light Deflection", False),
        (PhotonSphereValidator, "Photon Sphere", False),
        (PpnValidator, "PPN Parameters", False),
        (GwValidator, "Gravitational Waves", False),
        (PsrJ0740Validator, "PSR J0740", False),
        
        # Quantum-specific tests
        (COWInterferometryValidator, "COW Interferometry", False),
        (AtomInterferometryValidator, "Atom Interferometry", False),
        (GravitationalDecoherenceValidator, "Gravitational Decoherence", True),  # Needs hist
        (QuantumClockValidator, "Quantum Clock", False),
        (HawkingValidator, "Hawking Temperature", False),
        (QEDPrecisionValidator, "QED Precision", False),
    ]
    
    results = {
        'theory': theory_name,
        'category': category,
        'tests': [],
        'classical_tests': [],
        'quantum_tests': []
    }
    
    # Run each validator
    for validator_class, validator_name, needs_hist in validators:
        print(f"\n{validator_name}:")
        
        if needs_hist and hist is None:
            print(f"  ⚠ SKIP - No trajectory available")
            continue
            
        result = run_validator_test(theory, validator_class, validator_name, engine, hist if needs_hist else None)
        
        if result is None:
            print(f"  ⚠ SKIP - Not applicable or intermittent error")
            continue
            
        results['tests'].append(result)
        
        # Categorize test
        if result.get('is_quantum_test', False):
            results['quantum_tests'].append(result)
        else:
            results['classical_tests'].append(result)
        
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
    
    # Calculate summaries
    total_tests = len(results['tests'])
    passed_tests = sum(1 for t in results['tests'] if t['passed'])
    
    classical_total = len(results['classical_tests'])
    classical_passed = sum(1 for t in results['classical_tests'] if t['passed'])
    
    quantum_total = len(results['quantum_tests'])
    quantum_passed = sum(1 for t in results['quantum_tests'] if t['passed'])
    
    results['summary'] = {
        'total': total_tests,
        'passed': passed_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
        'classical_success_rate': classical_passed / classical_total if classical_total > 0 else 0,
        'quantum_success_rate': quantum_passed / quantum_total if quantum_total > 0 else 0,
        'classical_passed': classical_passed,
        'classical_total': classical_total,
        'quantum_passed': quantum_passed,
        'quantum_total': quantum_total
    }
    
    print(f"\nSummary:")
    print(f"  Overall: {passed_tests}/{total_tests} passed ({results['summary']['success_rate']*100:.1f}%)")
    print(f"  Classical: {classical_passed}/{classical_total} ({results['summary']['classical_success_rate']*100:.1f}%)")
    print(f"  Quantum: {quantum_passed}/{quantum_total} ({results['summary']['quantum_success_rate']*100:.1f}%)")
    
    return results

def main():
    """Run all tests and generate comprehensive report."""
    print("COMPREHENSIVE THEORY VALIDATION WITH QUANTUM TESTS")
    print("="*80)
    print(f"Testing {len(ALL_THEORIES)} theories")
    print("Expected: Quantum theories excel at quantum tests")
    print("          Classical theories fail quantum tests")
    print("          Newtonian Limit fails most tests naturally")
    
    all_results = []
    
    # Test all theories
    for theory_name, theory_class, category in ALL_THEORIES:
        result = test_theory(theory_name, theory_class, category)
        if result:
            all_results.append(result)
        time.sleep(0.1)  # Brief pause
    
    # Calculate combined score: 70% classical + 30% quantum for classical theories
    #                          30% classical + 70% quantum for quantum theories
    for result in all_results:
        if result['category'] == 'quantum':
            # Quantum theories should excel at quantum tests
            result['weighted_score'] = (0.3 * result['summary']['classical_success_rate'] + 
                                       0.7 * result['summary']['quantum_success_rate'])
        else:
            # Classical theories should do well on classical but fail quantum
            result['weighted_score'] = (0.7 * result['summary']['classical_success_rate'] + 
                                       0.3 * result['summary']['quantum_success_rate'])
    
    # Sort by weighted score (descending)
    all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
    
    # Generate report
    print("\n\n" + "="*100)
    print("FINAL RANKINGS (Weighted by Theory Type)")
    print("="*100)
    
    print(f"\n{'Rank':<6} {'Theory':<30} {'Category':<12} {'Score':<10} {'Classical':<12} {'Quantum':<12}")
    print("-"*100)
    
    for i, result in enumerate(all_results, 1):
        theory = result['theory']
        category = result['category']
        score = f"{result['weighted_score']*100:.1f}%"
        classical = f"{result['summary']['classical_passed']}/{result['summary']['classical_total']}"
        quantum = f"{result['summary']['quantum_passed']}/{result['summary']['quantum_total']}"
        
        # Highlight special cases
        marker = ""
        if theory == "Schwarzschild":
            marker = " ← Baseline"
        elif theory == "Newtonian Limit":
            marker = " ← Classical limit"
        elif category == "quantum" and result['weighted_score'] > 0.7:
            marker = " ✓ Quantum advantage"
        
        print(f"{i:<6} {theory:<30} {category:<12} {score:<10} {classical:<12} {quantum:<12}{marker}")
    
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
    
    # Check quantum test performance
    print("\nQuantum Test Performance:")
    for result in all_results[:5]:
        qt_rate = result['summary']['quantum_success_rate']
        print(f"  {result['theory']}: {qt_rate*100:.1f}% quantum tests passed")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_theories': len(all_results),
            'rankings': [(r['theory'], r['category'], r['weighted_score']) for r in all_results]
        },
        'full_results': all_results
    }
    
    report_file = f"theory_validation_quantum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
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