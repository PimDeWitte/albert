#!/usr/bin/env python3
"""Generate HTML report for failed theory"""
import sys
import os
sys.path.insert(0, '/Users/p/dev/albert')

from physics_agent.validations.comprehensive_report_generator import ComprehensiveReportGenerator
from physics_agent.theory_engine_core import TheoryEngine
import json

# Find the latest run
runs_dir = "runs"
run_dirs = [d for d in os.listdir(runs_dir) if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d))]
latest_run = sorted(run_dirs)[-1]

# Path to failed theory
fail_dir = os.path.join(runs_dir, latest_run, "fail", "Quantum_Corrected_α_+0_01")
val_results_path = os.path.join(fail_dir, "validation_results.json")

if os.path.exists(val_results_path):
    # Load validation results
    with open(val_results_path) as f:
        val_results = json.load(f)
    
    # Generate report
    generator = ComprehensiveReportGenerator()
    generator.generate_report("Quantum Corrected (α=+0.01)", val_results, fail_dir)
    
    print(f"Generated report in: {fail_dir}/results.html")
    
    # Now run our color test
    import subprocess
    subprocess.run(["python", "test_html_colors.py"])
else:
    print(f"No validation results found at {val_results_path}") 