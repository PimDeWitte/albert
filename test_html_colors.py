#!/usr/bin/env python3
"""Test HTML report color generation"""
import os

# Find the latest run directory
runs_dir = "runs"
run_dirs = [d for d in os.listdir(runs_dir) if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d))]
latest_run = sorted(run_dirs)[-1] if run_dirs else None

if not latest_run:
    print("No run directories found")
    exit(1)

# Check if there's a fail directory with Quantum Corrected theory
fail_dir = os.path.join(runs_dir, latest_run, "fail", "Quantum_Corrected_α_+0_01")
results_html = os.path.join(fail_dir, "results.html")

if os.path.exists(results_html):
    print(f"Found results.html at: {results_html}")
    
    # Read and check for pass/fail colors
    with open(results_html, 'r') as f:
        content = f.read()
        
    # Count occurrences
    pass_count = content.count('<span class="pass">PASS</span>')
    fail_count = content.count('<span class="fail">FAIL</span>')
    
    print(f"\nColor usage in HTML:")
    print(f"  PASS (green): {pass_count} occurrences")
    print(f"  FAIL (red): {fail_count} occurrences")
    
    # Check CSS
    if '.pass { color: green;' in content:
        print("  ✓ CSS correctly defines .pass as green")
    if '.fail { color: red;' in content:
        print("  ✓ CSS correctly defines .fail as red")
        
    # Show some examples
    print("\nSample validators:")
    if "COWInterferometryValidator" in content:
        start = content.find("COWInterferometryValidator")
        end = content.find("</tr>", start)
        row = content[start:end]
        if "PASS" in row:
            print("  COW Interferometry: PASS (should be green)")
        elif "FAIL" in row:
            print("  COW Interferometry: FAIL (should be red)")
            
    if "MercuryPrecessionValidator" in content:
        start = content.find("MercuryPrecessionValidator")
        end = content.find("</tr>", start)
        row = content[start:end]
        if "PASS" in row:
            print("  Mercury Precession: PASS (should be green)")
        elif "FAIL" in row:
            print("  Mercury Precession: FAIL (should be red)")
else:
    print(f"No results.html found at {results_html}")
    print("Run 'python -m physics_agent run --theory-filter \"quantum_corrected\" --steps 100' first") 