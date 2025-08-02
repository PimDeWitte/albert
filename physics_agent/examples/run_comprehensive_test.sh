#!/bin/bash
# Example: Run comprehensive theory validation test

echo "=== Comprehensive Theory Validation Test Examples ==="
echo ""

# Method 1: Default behavior - runs comprehensive test only
echo "1. Running comprehensive test (default behavior)..."
python -m physics_agent.theory_engine_core

# Method 2: Run comprehensive test AND continue with simulation
# echo "2. Running comprehensive test + full simulation..."
# python -m physics_agent.theory_engine_core --continue-after-test

# Method 3: Skip comprehensive test (not recommended)
# echo "3. Skipping comprehensive test..."
# python -m physics_agent.theory_engine_core --skip-comprehensive-test

# Method 4: Run standalone script
# echo "4. Running standalone comprehensive validation..."
# python physics_agent/run_comprehensive_validation.py

echo ""
echo "Done! Check the generated HTML report for detailed results."
echo "Report location: physics_agent/reports/latest_comprehensive_validation.html"