#!/usr/bin/env python3
"""
Standalone script to reproduce the prediction improvement.
Run this script to verify the result independently.
"""

import sys
import os

# Add the gravity_compression directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../')))

from physics_agent.validations.cmb_power_spectrum_validator import CMBPowerSpectrumValidator
from theory_implementation import theory

def main():
    print("Reproducing Prediction Improvement")
    print("=" * 60)
    
    # Create validator
    validator = CMBPowerSpectrumValidator()
    
    print(f"Theory: {theory.name}")
    print(f"Validator: {validator.name}")
    print()
    
    # Run validation
    result = validator.validate(theory, verbose=True)
    
    print()
    print("Results:")
    print(f"  Beats SOTA: {result.beats_sota}")
    print(f"  Predicted Value: {result.predicted_value}")
    print(f"  SOTA Value: {result.sota_value}")
    print(f"  Improvement: {result.error} ({result.error_percent:.1f}%)")
    
    if result.beats_sota:
        print()
        print("✅ REPRODUCTION SUCCESSFUL - Theory beats SOTA!")
    else:
        print()
        print("❌ REPRODUCTION FAILED - Theory does not beat SOTA")
    
    return result

if __name__ == "__main__":
    result = main()
