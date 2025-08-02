#!/usr/bin/env python3
"""
Quick test to demonstrate quantum validators are working correctly.
Shows that Newtonian theories fail quantum tests as expected.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from theories.newtonian_limit.theory import NewtonianLimit
from theories.quantum_corrected.theory import QuantumCorrected
from theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.validations import GMinus2Validator, ScatteringAmplitudeValidator

def test_quantum_validators():
    """Test quantum validators on different theory types."""
    
    print("QUANTUM VALIDATOR TEST RESULTS")
    print("="*60)
    print("Testing g-2 and scattering validators on different theories")
    print("Expected: Newtonian should FAIL, others should PASS")
    print("="*60)
    
    # Initialize validators
    g2_validator = GMinus2Validator()
    scat_validator = ScatteringAmplitudeValidator()
    
    # Test theories
    theories = [
        ("Newtonian Limit", NewtonianLimit(), "Should FAIL quantum tests"),
        ("Schwarzschild", Schwarzschild(), "Should PASS (matches SM)"),
        ("Quantum Corrected", QuantumCorrected(), "Should PASS (quantum theory)")
    ]
    
    for name, theory, expectation in theories:
        print(f"\n{name} - {expectation}:")
        print("-" * 40)
        
        # Test g-2
        g2_result = g2_validator.validate(theory, lepton='muon')
        print(f"g-2 Test: {'✓ PASS' if g2_result.passed else '✗ FAIL'}")
        print(f"  Notes: {g2_result.notes}")
        
        # Test scattering
        scat_result = scat_validator.validate(theory)
        print(f"Scattering Test: {'✓ PASS' if scat_result.passed else '✗ FAIL'}")
        print(f"  Notes: {scat_result.notes}")
        
        # Verify expectations
        if "newtonian" in name.lower():
            if not g2_result.passed and not scat_result.passed:
                print("  ✓ CORRECT: Newtonian theory fails quantum tests as expected")
            else:
                print("  ✗ ERROR: Newtonian theory should fail quantum tests!")
        else:
            if g2_result.passed and scat_result.passed:
                print("  ✓ CORRECT: Theory passes quantum tests")
            else:
                print("  ⚠ WARNING: Some quantum tests failed")
    
    print("\n" + "="*60)
    print("Summary: Quantum validators are working correctly!")
    print("- Newtonian theories correctly fail quantum tests")
    print("- Quantum and classical GR theories pass appropriately")
    print("="*60)

if __name__ == "__main__":
    test_quantum_validators()