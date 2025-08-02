#!/usr/bin/env python3
"""
Run comprehensive validation tests on all theories.
This replaces the deleted test_comprehensive_final.py
"""

import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.theory_engine_core import main as run_theory_engine

def main():
    """Run comprehensive validation with proper arguments."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run comprehensive theory validation')
    parser.add_argument('--theory-filter', type=str, default=None,
                       help='Filter which theories to test (regex pattern)')
    parser.add_argument('--sweep', action='store_true',
                       help='Run parameter sweep for each theory')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable trajectory caching')
    
    args = parser.parse_args()
    
    print("="*60)
    print("COMPREHENSIVE THEORY VALIDATION")
    print("="*60)
    print(f"Timestamp: {datetime.now()}")
    print(f"Theory filter: {args.theory_filter or 'All theories'}")
    print(f"Parameter sweep: {args.sweep}")
    print(f"Verbose: {args.verbose}")
    print(f"Cache: {'disabled' if args.no_cache else 'enabled'}")
    print("="*60)
    print()
    
    # Build command line arguments for theory engine
    sys.argv = ['theory_engine_core.py']
    
    if args.theory_filter:
        sys.argv.extend(['--theory-filter', args.theory_filter])
    
    if args.sweep:
        sys.argv.append('--sweep')
        
    if args.verbose:
        sys.argv.append('--verbose')
        
    if args.no_cache:
        sys.argv.append('--no-cache')
    
    # Run the theory engine
    print("Starting comprehensive validation...")
    print("This will test theories with all validators including:")
    print("- Classical validators (Mercury precession, light deflection, etc.)")
    print("- Quantum validators (g-2, scattering amplitudes)")
    print("- Constraint validators (conservation, metric properties)")
    print()
    
    try:
        run_theory_engine()
        print("\n✓ Comprehensive validation completed successfully!")
        print(f"Check the latest report in: physics_agent/runs/")
    except Exception as e:
        print(f"\n✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())