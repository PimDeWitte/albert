#!/usr/bin/env python3
"""
Test script to verify quantum validator integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from theory_engine_core import TheoryEngine
from theories.newtonian_limit.theory import NewtonianLimit
from theories.quantum_corrected.theory import QuantumCorrected
from theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.validations import GMinus2Validator, ScatteringAmplitudeValidator

def test_validators_directly():
    """Test validators directly on theories."""
    print("="*60)
    print("DIRECT VALIDATOR TESTS")
    print("="*60)
    
    theories = [
        NewtonianLimit(),
        Schwarzschild(),
        QuantumCorrected()
    ]
    
    g2_validator = GMinus2Validator()
    scat_validator = ScatteringAmplitudeValidator()
    
    for theory in theories:
        print(f"\n{theory.name}:")
        
        # Test g-2
        g2_result = g2_validator.validate(theory, lepton='muon')
        print(f"  g-2: {'PASS' if g2_result.passed else 'FAIL'}")
        print(f"    Notes: {g2_result.notes}")
        
        # Test scattering
        scat_result = scat_validator.validate(theory, process='ee_to_mumu')
        print(f"  Scattering: {'PASS' if scat_result.passed else 'FAIL'}")
        print(f"    Notes: {scat_result.notes}")

def test_theory_engine_integration():
    """Test validators through theory engine."""
    print("\n" + "="*60)
    print("THEORY ENGINE INTEGRATION TEST")
    print("="*60)
    
    engine = TheoryEngine(verbose=True)
    
    # Test Newtonian theory
    print("\nTesting Newtonian Limit...")
    newtonian = NewtonianLimit()
    
    # Run just the observational validations
    import torch
    dummy_hist = torch.zeros((100, 4))  # Dummy trajectory
    dummy_y0 = torch.tensor([0.0, 10.0, 0.0, 0.0])
    
    results = engine.run_all_validations(
        newtonian, 
        dummy_hist, 
        dummy_y0,
        categories=["observational"]
    )
    
    # Check quantum validator results
    print("\nQuantum validator results for Newtonian theory:")
    for val in results.get('validations', []):
        if 'g-2' in val['validator'] or 'Scattering' in val['validator']:
            print(f"  {val['validator']}: {val['flags']['overall']}")
            if 'details' in val['flags']:
                print(f"    Details: {val['flags']['details']}")

def main():
    """Run all tests."""
    test_validators_directly()
    test_theory_engine_integration()
    print("\n✓ Integration test complete")

if __name__ == "__main__":
    main()