#!/usr/bin/env python3
"""Monitor validation results from the theory engine output."""

import re
import sys
import time

def monitor_output(filename="full_run_output.txt"):
    """Monitor the output file for validation results."""
    theories_validated = {}
    validation_phase = False
    current_theory = None
    
    print("Monitoring validation results...")
    print("=" * 80)
    print("Hierarchical Validation: Constraints → Observations → Predictions")
    print("=" * 80)
    
    try:
        with open(filename, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                    
                # Detect validation phase
                if "Phase 1: Validating All Theories" in line:
                    validation_phase = True
                    print("\n🔍 VALIDATION PHASE STARTED")
                    print("=" * 80)
                
                # Track current theory being validated
                if "--- Validating:" in line:
                    current_theory = line.split("Validating:")[1].strip().rstrip(" ---")
                    theories_validated[current_theory] = {
                        'constraints': None,
                        'observations': None, 
                        'predictions': None,
                        'overall': None
                    }
                    print(f"\n📊 Validating: {current_theory}")
                
                # Track test results
                if current_theory:
                    if "Constraint tests:" in line:
                        result = "PASSED" if "PASSED" in line else "FAILED"
                        theories_validated[current_theory]['constraints'] = result
                        print(f"  1️⃣ Constraints: {result}")
                        
                    elif "Observation tests:" in line:
                        result = "PASSED" if "PASSED" in line else "FAILED"
                        theories_validated[current_theory]['observations'] = result
                        print(f"  2️⃣ Observations: {result}")
                        
                    elif "Skipping observational tests" in line:
                        theories_validated[current_theory]['observations'] = "SKIPPED"
                        print(f"  2️⃣ Observations: SKIPPED (constraints failed)")
                        
                    elif "Prediction tests:" in line:
                        if "BEATS SOTA" in line:
                            result = "BEATS SOTA"
                        elif "Does not beat SOTA" in line:
                            result = "NO SOTA"
                        elif "SKIPPED" in line:
                            result = "SKIPPED"
                        else:
                            result = "UNKNOWN"
                        theories_validated[current_theory]['predictions'] = result
                        if "constraints failed" in line:
                            print(f"  3️⃣ Predictions: SKIPPED (constraints failed)")
                        elif "observations failed" in line:
                            print(f"  3️⃣ Predictions: SKIPPED (observations failed)")
                        else:
                            print(f"  3️⃣ Predictions: {result}")
                        
                    elif "Skipping prediction tests" in line:
                        theories_validated[current_theory]['predictions'] = "SKIPPED"
                        if "constraints failed" in line:
                            print(f"  3️⃣ Predictions: SKIPPED (constraints failed)")
                        else:
                            print(f"  3️⃣ Predictions: SKIPPED (observations failed)")
                        
                    elif "Overall validation:" in line:
                        result = "PASSED" if "PASSED" in line else "FAILED"
                        theories_validated[current_theory]['overall'] = result
                        print(f"  {'✅' if result == 'PASSED' else '❌'} Overall: {result}")
                        
                        # Show why theory passed or failed
                        theory_data = theories_validated[current_theory]
                        if result == "PASSED":
                            print(f"     → Passed all required tests (constraints + observations)")
                        else:
                            if theory_data['constraints'] == "FAILED":
                                print(f"     → Failed at constraint level")
                            elif theory_data['observations'] == "FAILED":
                                print(f"     → Failed at observation level")
                
                # Detect end of validation phase
                if "Theories that passed validation:" in line or "Validation Summary:" in line:
                    print("\n" + "=" * 80)
                    print("VALIDATION SUMMARY:")
                    print("=" * 80)
                    
                    passed_count = sum(1 for t in theories_validated.values() if t['overall'] == 'PASSED')
                    total_count = len(theories_validated)
                    
                    print(f"\nTotal theories validated: {total_count}")
                    print(f"Passed: {passed_count}")
                    print(f"Failed: {total_count - passed_count}")
                    
                    # Categorize failures
                    constraint_failures = [t for t, r in theories_validated.items() if r['constraints'] == 'FAILED']
                    observation_failures = [t for t, r in theories_validated.items() 
                                         if r['constraints'] == 'PASSED' and r['observations'] == 'FAILED']
                    
                    if constraint_failures:
                        print(f"\nFailed at constraint level ({len(constraint_failures)}):")
                        for theory in constraint_failures:
                            print(f"  ❌ {theory}")
                    
                    if observation_failures:
                        print(f"\nFailed at observation level ({len(observation_failures)}):")
                        for theory in observation_failures:
                            print(f"  ❌ {theory}")
                    
                    # Show theories that made it to predictions
                    prediction_tested = [t for t, r in theories_validated.items() 
                                       if r['predictions'] not in [None, 'SKIPPED']]
                    if prediction_tested:
                        print(f"\nTheories tested for predictions ({len(prediction_tested)}):")
                        for theory in prediction_tested:
                            pred_result = theories_validated[theory]['predictions']
                            print(f"  {'🌟' if pred_result == 'BEATS SOTA' else '📊'} {theory}: {pred_result}")
                    
                    print("\nDetailed results:")
                    for theory, results in theories_validated.items():
                        print(f"\n{theory}:")
                        print(f"  Constraints: {results['constraints']}")
                        print(f"  Observations: {results['observations']}")
                        print(f"  Predictions: {results['predictions']}")
                        print(f"  Overall: {results['overall']}")
                    
                    break
                    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        return theories_validated
    except Exception as e:
        print(f"\nError: {e}")
        return theories_validated

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "full_run_output.txt"
    monitor_output(filename) 