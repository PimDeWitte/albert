#!/usr/bin/env python3
"""
Standalone script to run comprehensive theory validation.
This generates the scientific scorecard for all theories.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.test_comprehensive_final import run_comprehensive_tests

def main():
    """Run comprehensive validation and generate reports."""
    print("="*80)
    print("Running Comprehensive Theory Validation")
    print("This generates the scientific scorecard for all theories")
    print("="*80)
    
    try:
        # Run the tests
        test_results, json_file, html_file = run_comprehensive_tests()
        
        print("\n" + "="*80)
        print("Comprehensive Validation Complete!")
        print("="*80)
        print(f"\nReports generated:")
        print(f"  - JSON report: {json_file}")
        print(f"  - HTML report: {html_file}")
        print(f"  - Latest HTML: physics_agent/reports/latest_comprehensive_validation.html")
        
        print("\nOpen the HTML report in a browser to view the full scientific scorecard.")
        
        # Optionally open the HTML report
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(html_file)}")
        except:
            pass  # Silently fail if can't open browser
            
    except Exception as e:
        print(f"\nError running comprehensive validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()