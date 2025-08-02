#!/usr/bin/env python3
"""
Investigation script to understand why quantum tests are failing in comprehensive tests.
This file will load theories directly and test them with the quantum validators.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import traceback
from theory_engine_core import TheoryEngine
from test_comprehensive_final import test_g_minus_2, test_scattering_amplitude

# Import some theories to test
from theories.defaults.baselines.schwarzschild import Schwarzschild
from theories.newtonian_limit.theory import NewtonianLimit
from theories.quantum_corrected.theory import QuantumCorrected
from theories.string.theory import StringTheory
from theories.einstein_teleparallel.theory import EinsteinTeleparallel

def investigate_theory(theory_name, theory_class):
    """Test a single theory with quantum validators."""
    print(f"\n{'='*60}")
    print(f"Testing: {theory_name}")
    print(f"{'='*60}")
    
    try:
        # Instantiate theory
        theory = theory_class()
        print(f"✓ Theory instantiated: {theory.name}")
        
        # Check if theory has required quantum methods
        has_coupling = hasattr(theory, 'get_coupling_constants')
        has_scattering = hasattr(theory, 'calculate_scattering_amplitude')
        print(f"✓ Has get_coupling_constants: {has_coupling}")
        print(f"✓ Has calculate_scattering_amplitude: {has_scattering}")
        
        # Test g-2
        print("\nTesting g-2...")
        try:
            g2_result = test_g_minus_2(theory)
            print(f"  Status: {g2_result.get('status', 'UNKNOWN')}")
            print(f"  Passed: {g2_result.get('passed', False)}")
            print(f"  Notes: {g2_result.get('notes', 'N/A')}")
            if 'error' in g2_result:
                print(f"  Error: {g2_result['error']}")
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            traceback.print_exc()
        
        # Test scattering
        print("\nTesting Scattering Amplitudes...")
        try:
            scat_result = test_scattering_amplitude(theory)
            print(f"  Status: {scat_result.get('status', 'UNKNOWN')}")
            print(f"  Passed: {scat_result.get('passed', False)}")
            print(f"  Notes: {scat_result.get('notes', 'N/A')}")
            if 'error' in scat_result:
                print(f"  Error: {scat_result['error']}")
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"FAILED to instantiate theory: {e}")
        traceback.print_exc()

def main():
    """Main investigation."""
    print("QUANTUM TESTS INVESTIGATION")
    print("===========================")
    print("This script directly tests theories with quantum validators")
    print("to understand why they're failing in comprehensive tests.\n")
    
    # Test a variety of theories
    theories_to_test = [
        ("Schwarzschild", Schwarzschild),
        ("Newtonian Limit", NewtonianLimit),
        ("Quantum Corrected", QuantumCorrected),
        ("String Theory", StringTheory),
        ("Einstein Teleparallel", EinsteinTeleparallel),
    ]
    
    for name, theory_class in theories_to_test:
        investigate_theory(name, theory_class)
    
    # Now let's check if the validators themselves exist
    print(f"\n{'='*60}")
    print("CHECKING VALIDATOR IMPORTS")
    print(f"{'='*60}")
    
    try:
        from validations.g_minus_2_validator import GMinus2Validator
        print("✓ GMinus2Validator can be imported")
    except ImportError as e:
        print(f"✗ GMinus2Validator import failed: {e}")
    
    try:
        from validations.scattering_amplitude_validator import ScatteringAmplitudeValidator
        print("✓ ScatteringAmplitudeValidator can be imported")
    except ImportError as e:
        print(f"✗ ScatteringAmplitudeValidator import failed: {e}")
    
    # Check if test functions are trying to import non-existent validators
    print(f"\n{'='*60}")
    print("ANALYZING TEST FUNCTION IMPORTS")
    print(f"{'='*60}")
    
    # Let's look at what test_g_minus_2 actually does
    import inspect
    print("\ntest_g_minus_2 source:")
    try:
        print(inspect.getsource(test_g_minus_2))
    except:
        print("Could not get source")
    
    print("\ntest_scattering_amplitude source:")
    try:
        print(inspect.getsource(test_scattering_amplitude))
    except:
        print("Could not get source")

if __name__ == "__main__":
    main()