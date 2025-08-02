#!/usr/bin/env python3
"""
Run comprehensive test with quantum validators to check report outputs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from datetime import datetime
from theory_engine_core import TheoryEngine
from theories.newtonian_limit.theory import NewtonianLimit
from theories.quantum_corrected.theory import QuantumCorrected
from theories.defaults.baselines.schwarzschild import Schwarzschild
from theories.string.theory import StringTheory
from comprehensive_test_report_generator import ComprehensiveTestReportGenerator

def run_comprehensive_test():
    """Run comprehensive tests on selected theories."""
    print("Running Comprehensive Theory Tests with Quantum Validators")
    print("="*60)
    
    engine = TheoryEngine(verbose=False)
    report_generator = ComprehensiveTestReportGenerator()
    
    # Test theories
    theories = [
        NewtonianLimit(),
        Schwarzschild(),
        QuantumCorrected(),
        StringTheory()
    ]
    
    # Dummy trajectory data
    dummy_hist = torch.zeros((100, 4))
    dummy_y0 = torch.tensor([0.0, 10.0, 0.0, 0.0])
    
    results = []
    
    for theory in theories:
        print(f"\nTesting {theory.name}...")
        
        # Run validations
        validation_results = engine.run_all_validations(
            theory,
            dummy_hist,
            dummy_y0,
            categories=["observational"]
        )
        
        # Format results for report
        theory_result = {
            'theory_name': theory.name,
            'category': getattr(theory, 'category', 'unknown'),
            'tests': [],
            'score': 0,
            'total_tests': 0
        }
        
        # Process validation results
        for val in validation_results.get('validations', []):
            test_result = {
                'name': val.get('validator', 'Unknown'),
                'status': val.get('flags', {}).get('overall', 'ERROR'),
                'passed': val.get('flags', {}).get('overall', 'ERROR') == 'PASS',
                'notes': val.get('flags', {}).get('details', ''),
                'loss': val.get('loss', 1.0)
            }
            
            # Add specific details for quantum validators
            if 'g-2' in test_result['name'].lower():
                test_result['notes'] = f"Quantum correction test: {test_result['status']}"
            elif 'scattering' in test_result['name'].lower():
                test_result['notes'] = f"Particle physics test: {test_result['status']}"
            
            theory_result['tests'].append(test_result)
            theory_result['total_tests'] += 1
            if test_result['passed']:
                theory_result['score'] += 1
        
        results.append(theory_result)
        
        # Print summary
        print(f"  Score: {theory_result['score']}/{theory_result['total_tests']}")
        for test in theory_result['tests']:
            if 'g-2' in test['name'].lower() or 'scattering' in test['name'].lower():
                print(f"  {test['name']}: {test['status']}")
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"quantum_validator_test_report_{timestamp}.html"
    
    print(f"\nGenerating report: {report_file}")
    report_generator.generate_report(results, report_file)
    
    # Check quantum validator results
    print("\n" + "="*60)
    print("QUANTUM VALIDATOR RESULTS SUMMARY")
    print("="*60)
    
    for result in results:
        theory_name = result['theory_name']
        g2_result = next((t for t in result['tests'] if 'g-2' in t['name'].lower()), None)
        scat_result = next((t for t in result['tests'] if 'scattering' in t['name'].lower()), None)
        
        print(f"\n{theory_name}:")
        if g2_result:
            print(f"  g-2: {g2_result['status']} - {g2_result.get('notes', '')}")
        if scat_result:
            print(f"  Scattering: {scat_result['status']} - {scat_result.get('notes', '')}")
        
        # Check if results match expectations
        if 'newtonian' in theory_name.lower():
            if g2_result and g2_result['status'] == 'FAIL' and scat_result and scat_result['status'] == 'FAIL':
                print("  ✓ Correctly fails quantum tests (as expected)")
            else:
                print("  ✗ ERROR: Should fail quantum tests!")
        else:
            if g2_result and g2_result['status'] == 'PASS' and scat_result and scat_result['status'] == 'PASS':
                print("  ✓ Passes quantum tests")
            else:
                print("  ⚠ Some quantum tests failed")

if __name__ == "__main__":
    run_comprehensive_test()