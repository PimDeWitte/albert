#!/usr/bin/env python3
"""
Proper theory validation test using the framework's validation system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import json
from datetime import datetime

# Import engine and constants
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT

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

def run_theory_validation(theory_name, theory_class):
    """Run comprehensive validation on a theory."""
    print(f"\n{'='*60}")
    print(f"Testing: {theory_name}")
    print(f"{'='*60}")
    
    try:
        theory = theory_class()
        print(f"Initialized: {theory.name}")
    except Exception as e:
        print(f"ERROR: Failed to initialize - {e}")
        return None
    
    # Create engine
    engine = TheoryEngine()
    
    # Setup validation parameters
    rs = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
    r0_val = 6.0 * rs  # 6 Schwarzschild radii
    r0 = torch.tensor([r0_val], dtype=torch.float64)
    
    # Run validation trajectory
    print(f"\nRunning validation trajectory...")
    dtau = torch.tensor(0.1, dtype=torch.float64)
    n_steps = 100
    
    try:
        # Run trajectory
        hist, tag, kicks = engine.run_trajectory(
            theory, r0, n_steps, dtau,
            no_cache=True,
            verbose=False
        )
        
        if hist is None:
            print("ERROR: Trajectory computation failed")
            return None
            
        print(f"  Trajectory computed successfully ({tag})")
        
        # Get initial conditions
        y0_sym, y0_gen, _ = engine.get_initial_conditions(theory, r0)
        
        # Run all validations
        print(f"\nRunning validations...")
        validation_results = engine.run_all_validations(
            theory, hist, y0_gen,
            categories=["constraint", "observational"]
        )
        
        # Process results
        results = process_validation_results(theory_name, validation_results)
        
        # Print summary
        print_validation_summary(results)
        
        return results
        
    except Exception as e:
        print(f"ERROR during validation: {str(e)[:200]}...")
        import traceback
        traceback.print_exc()
        return None

def process_validation_results(theory_name, validation_results):
    """Process raw validation results into structured format."""
    validations = validation_results.get('validations', [])
    
    # Categorize results
    constraints = []
    observational = []
    
    for val in validations:
        val_type = val.get('type', 'unknown')
        validator_name = val.get('validator', 'Unknown')
        flags = val.get('flags', {})
        overall = flags.get('overall', 'UNKNOWN')
        loss = val.get('loss', None)
        
        result = {
            'name': validator_name,
            'status': overall,
            'passed': overall in ['PASS', 'WARNING'],
            'loss': loss
        }
        
        if val_type == 'constraint':
            constraints.append(result)
        elif val_type == 'observational':
            observational.append(result)
    
    # Calculate summary statistics
    total_constraints = len(constraints)
    passed_constraints = sum(1 for c in constraints if c['passed'])
    
    total_observational = len(observational)
    passed_observational = sum(1 for o in observational if o['passed'])
    
    total_all = total_constraints + total_observational
    passed_all = passed_constraints + passed_observational
    
    return {
        'theory': theory_name,
        'constraints': {
            'results': constraints,
            'total': total_constraints,
            'passed': passed_constraints,
            'success_rate': passed_constraints / total_constraints if total_constraints > 0 else 0
        },
        'observational': {
            'results': observational,
            'total': total_observational,
            'passed': passed_observational,
            'success_rate': passed_observational / total_observational if total_observational > 0 else 0
        },
        'overall': {
            'total': total_all,
            'passed': passed_all,
            'success_rate': passed_all / total_all if total_all > 0 else 0
        }
    }

def print_validation_summary(results):
    """Print a nice summary of validation results."""
    print(f"\nValidation Summary for {results['theory']}:")
    print("-" * 50)
    
    # Constraints
    print(f"\nConstraints ({results['constraints']['passed']}/{results['constraints']['total']}):")
    for c in results['constraints']['results']:
        status = "✓" if c['passed'] else "✗"
        print(f"  {status} {c['name']}: {c['status']}")
        if c['loss'] is not None:
            print(f"     Loss: {c['loss']:.6f}")
    
    # Observational
    print(f"\nObservational ({results['observational']['passed']}/{results['observational']['total']}):")
    for o in results['observational']['results']:
        status = "✓" if o['passed'] else "✗"
        print(f"  {status} {o['name']}: {o['status']}")
        if o['loss'] is not None:
            print(f"     Loss: {o['loss']:.6f}")
    
    # Overall
    print(f"\nOverall: {results['overall']['passed']}/{results['overall']['total']} passed " +
          f"({results['overall']['success_rate']*100:.1f}%)")

def main():
    """Run all tests and generate comprehensive report."""
    print("COMPREHENSIVE THEORY VALIDATION")
    print("="*80)
    print(f"Testing {len(CLASSICAL_ACCEPTED)} classical theories")
    print(f"Testing {len(QUANTUM_SAMPLE)} quantum theories")
    print(f"\nUsing framework's built-in validation system")
    
    all_results = []
    
    # Test classical theories
    print("\n\nCLASSICAL THEORIES")
    print("="*80)
    for name, theory_class in CLASSICAL_ACCEPTED:
        result = run_theory_validation(name, theory_class)
        if result:
            all_results.append(result)
        time.sleep(0.1)  # Brief pause between theories
    
    # Test quantum theories
    print("\n\nQUANTUM THEORIES")
    print("="*80)
    for name, theory_class in QUANTUM_SAMPLE:
        result = run_theory_validation(name, theory_class)
        if result:
            all_results.append(result)
        time.sleep(0.1)
    
    # Generate final report
    print("\n\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    
    # Sort by overall success rate
    all_results.sort(key=lambda x: x['overall']['success_rate'], reverse=True)
    
    # Print summary table
    print(f"\n{'Theory':<30} {'Constraints':<15} {'Observational':<15} {'Overall':<15}")
    print("-"*80)
    
    for result in all_results:
        theory = result['theory']
        c_rate = f"{result['constraints']['passed']}/{result['constraints']['total']}"
        o_rate = f"{result['observational']['passed']}/{result['observational']['total']}"
        overall = f"{result['overall']['success_rate']*100:.1f}%"
        
        print(f"{theory:<30} {c_rate:<15} {o_rate:<15} {overall:<15}")
    
    # Analysis by category
    print("\n\nANALYSIS BY CATEGORY:")
    print("-"*60)
    
    # Classical theories
    classical_results = [r for r in all_results if r['theory'] in [t[0] for t in CLASSICAL_ACCEPTED]]
    if classical_results:
        avg_rate = sum(r['overall']['success_rate'] for r in classical_results) / len(classical_results)
        print(f"\nClassical theories:")
        print(f"  Average success rate: {avg_rate*100:.1f}%")
        
        # Specific expectations
        schwarzschild = next((r for r in classical_results if r['theory'] == 'Schwarzschild'), None)
        if schwarzschild:
            print(f"  Schwarzschild (baseline): {schwarzschild['overall']['success_rate']*100:.1f}%")
            
        newtonian = next((r for r in classical_results if r['theory'] == 'Newtonian Limit'), None)
        if newtonian:
            print(f"  Newtonian Limit: {newtonian['overall']['success_rate']*100:.1f}%")
            if newtonian['overall']['success_rate'] < schwarzschild['overall']['success_rate']:
                print(f"    ✓ As expected, scores lower than Schwarzschild")
            else:
                print(f"    ⚠ Unexpected: Should score lower than Schwarzschild")
    
    # Quantum theories
    quantum_results = [r for r in all_results if r['theory'] in [t[0] for t in QUANTUM_SAMPLE]]
    if quantum_results:
        avg_rate = sum(r['overall']['success_rate'] for r in quantum_results) / len(quantum_results)
        print(f"\nQuantum theories:")
        print(f"  Average success rate: {avg_rate*100:.1f}%")
    
    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_theories_tested': len(all_results),
            'classical_theories': len(classical_results),
            'quantum_theories': len(quantum_results)
        },
        'results': all_results
    }
    
    report_file = f"theory_validation_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n\nDetailed report saved to: {report_file}")
    
    # Final verdict
    print("\n\nFINAL VERDICT:")
    print("-"*60)
    
    # Check accepted theories
    accepted_names = ['Kerr', 'Kerr-Newman', 'Yukawa', 'Einstein Teleparallel', 'Spinor Conformal']
    accepted_results = [r for r in all_results if r['theory'] in accepted_names]
    
    if accepted_results:
        avg_accepted = sum(r['overall']['success_rate'] for r in accepted_results) / len(accepted_results)
        print(f"Accepted theories average: {avg_accepted*100:.1f}%")
        if avg_accepted > 0.8:
            print("  ✓ Accepted theories performing well (>80%)")
        else:
            print("  ⚠ Accepted theories need improvement")
    
    return all_results

if __name__ == "__main__":
    results = main()