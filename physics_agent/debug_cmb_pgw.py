#!/usr/bin/env python3
"""Debug CMB and PGW test failures"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
from physics_agent.validations.cmb_power_spectrum_validator import CMBPowerSpectrumValidator
from physics_agent.validations.primordial_gws_validator import PrimordialGWsValidator

def debug_validator(theory, validator_class, validator_name):
    """Debug a single validator"""
    print(f"\n{'='*60}")
    print(f"Testing {validator_name} on {theory.name}")
    print('='*60)
    
    engine = TheoryEngine()
    validator = validator_class(engine=engine)
    
    # Run validation with verbose output
    result = validator.validate(theory, verbose=True)
    
    # Print detailed results
    print(f"\nDetailed Results:")
    if hasattr(result, '__dict__'):
        for key, value in result.__dict__.items():
            print(f"  {key}: {value}")
    elif isinstance(result, dict):
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    return result

def main():
    """Debug CMB and PGW tests"""
    print("Debugging CMB and PGW Test Failures")
    
    # Test on a baseline theory
    schwarzschild = Schwarzschild()
    debug_validator(schwarzschild, CMBPowerSpectrumValidator, "CMB Power Spectrum")
    debug_validator(schwarzschild, PrimordialGWsValidator, "Primordial GWs")
    
    # Test on a quantum theory
    quantum = QuantumCorrected()
    debug_validator(quantum, CMBPowerSpectrumValidator, "CMB Power Spectrum")
    debug_validator(quantum, PrimordialGWsValidator, "Primordial GWs")

if __name__ == "__main__":
    main()