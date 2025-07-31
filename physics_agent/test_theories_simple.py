#!/usr/bin/env python3
"""
Simplified theory validation test - runs all theories through all validators.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime
import json

# Import engine
from physics_agent.theory_engine_core import TheoryEngine

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

# Categories
CLASSICAL_ACCEPTED = [
    ("Schwarzschild", Schwarzschild),  # Baseline
    ("Newtonian Limit", NewtonianLimit),
    ("Kerr", Kerr),
    ("Kerr-Newman", KerrNewman),
    ("Yukawa", Yukawa),
    ("Einstein Teleparallel", EinsteinTeleparallel),
    ("Spinor Conformal", SpinorConformal),
]

QUANTUM_SAMPLE = [
    ("Quantum Corrected", QuantumCorrected),
    ("String Theory", StringTheory), 
    ("Asymptotic Safety", AsymptoticSafetyTheory),
    ("Loop Quantum Gravity", LoopQuantumGravity),
    ("Non-Commutative Geometry", NonCommutativeGeometry),
]

def run_theory_tests(theory_name, theory_class):
    """Run all standard tests on a theory."""
    print(f"\n{'='*60}")
    print(f"Testing: {theory_name}")
    print(f"{'='*60}")
    
    try:
        theory = theory_class()
    except Exception as e:
        print(f"ERROR: Failed to initialize - {e}")
        return None
        
    engine = TheoryEngine()
    results = {}
    
    # List of validators to test
    validators = [
        ('mercury_precession', "Mercury orbit precession"),
        ('light_deflection', "Solar light deflection"),
        ('photon_sphere', "Black hole photon sphere"),
        ('ppn', "PPN parameters"),
        ('cow_interferometry', "COW quantum test"),
        ('gw', "Gravitational waves"),
        ('psr_j0740', "Pulsar timing"),
    ]
    
    for val_name, description in validators:
        print(f"\n{description}:")
        try:
            # Run validation
            val_result = engine.run_validation(theory, val_name)
            
            # Extract results based on validator output structure
            if isinstance(val_result, dict):
                flags = val_result.get('flags', {})
                overall = flags.get('overall', 'UNKNOWN')
                loss = val_result.get('loss', None)
                
                # Determine if passed
                passed = overall in ['PASS', 'WARNING']
                
                results[val_name] = {
                    'status': overall,
                    'passed': passed,
                    'loss': loss
                }
                
                print(f"  Status: {overall}")
                if loss is not None:
                    print(f"  Loss: {loss:.6f}")
                    
            else:
                print(f"  ERROR: Unexpected result type {type(val_result)}")
                results[val_name] = {'status': 'ERROR', 'passed': False}
                
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}...")
            results[val_name] = {'status': 'ERROR', 'passed': False}
    
    # Summary
    total = len(validators)
    passed = sum(1 for r in results.values() if r.get('passed', False))
    success_rate = passed / total if total > 0 else 0
    
    print(f"\nSummary: {passed}/{total} passed ({success_rate*100:.1f}%)")
    
    return {
        'theory': theory_name,
        'results': results,
        'summary': {
            'total': total,
            'passed': passed,
            'success_rate': success_rate
        }
    }

def main():
    """Run all tests and generate report."""
    print("COMPREHENSIVE THEORY VALIDATION")
    print("="*60)
    print(f"Testing {len(CLASSICAL_ACCEPTED)} classical theories")
    print(f"Testing {len(QUANTUM_SAMPLE)} quantum theories (sample)")
    
    all_results = []
    
    # Test classical theories
    print("\n\nCLASSICAL THEORIES")
    print("="*60)
    for name, theory_class in CLASSICAL_ACCEPTED:
        result = run_theory_tests(name, theory_class)
        if result:
            all_results.append(result)
    
    # Test quantum theories
    print("\n\nQUANTUM THEORIES (SAMPLE)")
    print("="*60)
    for name, theory_class in QUANTUM_SAMPLE:
        result = run_theory_tests(name, theory_class)
        if result:
            all_results.append(result)
    
    # Generate summary report
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    # Sort by success rate
    all_results.sort(key=lambda x: x['summary']['success_rate'], reverse=True)
    
    print(f"\n{'Theory':<30} {'Passed':<10} {'Success Rate':<15}")
    print("-"*60)
    
    for result in all_results:
        theory = result['theory']
        summary = result['summary']
        print(f"{theory:<30} {summary['passed']}/{summary['total']:<10} {summary['success_rate']*100:>6.1f}%")
    
    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
        'summary': {
            'total_theories': len(all_results),
            'avg_success_rate': sum(r['summary']['success_rate'] for r in all_results) / len(all_results) if all_results else 0
        }
    }
    
    report_file = f"theory_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n\nDetailed report saved to: {report_file}")
    
    # Analysis
    print("\n\nANALYSIS:")
    print("-"*60)
    
    # Classical theories
    classical_results = [r for r in all_results if r['theory'] in [t[0] for t in CLASSICAL_ACCEPTED]]
    if classical_results:
        avg_classical = sum(r['summary']['success_rate'] for r in classical_results) / len(classical_results)
        print(f"Classical theories average: {avg_classical*100:.1f}%")
        
        # Check specific expectations
        schwarzschild = next((r for r in classical_results if r['theory'] == 'Schwarzschild'), None)
        if schwarzschild:
            print(f"  Schwarzschild (baseline): {schwarzschild['summary']['success_rate']*100:.1f}%")
            
        newtonian = next((r for r in classical_results if r['theory'] == 'Newtonian Limit'), None)
        if newtonian:
            print(f"  Newtonian Limit: {newtonian['summary']['success_rate']*100:.1f}%")
            print(f"    Note: Expected to fail some relativistic tests")
    
    # Quantum theories
    quantum_results = [r for r in all_results if r['theory'] in [t[0] for t in QUANTUM_SAMPLE]]
    if quantum_results:
        avg_quantum = sum(r['summary']['success_rate'] for r in quantum_results) / len(quantum_results)
        print(f"\nQuantum theories average: {avg_quantum*100:.1f}%")
    
    return all_results

if __name__ == "__main__":
    results = main()