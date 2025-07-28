#!/usr/bin/env python3
"""
Monitor validation chain: Schwarzschild → RN → Kerr → KN
Shows how different particles behave in each spacetime.
"""

import sys
import time
import re
from datetime import datetime

def parse_baseline_info(line):
    """Extract baseline information from log lines."""
    if "Loaded baseline" in line:
        match = re.search(r"Loaded baseline (\d+): (.+?) \((.+?)\)", line)
        if match:
            return {
                'number': int(match.group(1)),
                'name': match.group(2),
                'description': match.group(3)
            }
    return None

def parse_particle_test(line):
    """Extract particle test information."""
    if "Running trajectory for" in line:
        match = re.search(r"Running trajectory for (\w+)\.{3}", line)
        if match:
            return match.group(1)
    return None

def get_particle_properties(particle_name):
    """Return known particle properties."""
    particles = {
        'electron': {'charge': '-', 'mass': 'light', 'symbol': 'e⁻'},
        'proton': {'charge': '+', 'mass': 'heavy', 'symbol': 'p⁺'},
        'photon': {'charge': '0', 'mass': '0', 'symbol': 'γ'},
        'neutrino': {'charge': '0', 'mass': '≈0', 'symbol': 'ν'}
    }
    return particles.get(particle_name, {'charge': '?', 'mass': '?', 'symbol': '?'})

def main():
    print("🔬 VALIDATION CHAIN MONITOR")
    print("=" * 70)
    print("Tracking particle behavior through increasing spacetime complexity:")
    print()
    
    baselines = {}
    current_baseline = None
    current_particle = None
    
    print("📊 Expected Physics:")
    print("─" * 70)
    print("1. Schwarzschild: All particles follow same geodesics (charge ignored)")
    print("2. Reissner-Nordström: Charged particles (e⁻, p⁺) deviate from neutrals")
    print("3. Kerr: All particles experience frame-dragging")
    print("4. Kerr-Newman: Both charge and rotation effects combined")
    print()
    
    print("🚀 Monitoring baseline tests...")
    print("─" * 70)
    
    for line in sys.stdin:
        line = line.strip()
        
        # Parse baseline info
        baseline_info = parse_baseline_info(line)
        if baseline_info:
            baselines[baseline_info['number']] = baseline_info
            print(f"\n✓ Baseline {baseline_info['number']}: {baseline_info['name']}")
            print(f"  Description: {baseline_info['description']}")
        
        # Track which baseline is being run
        if "Running:" in line and not "Running trajectory" in line:
            for name in ['Schwarzschild', 'Reissner-Nordström', 'Kerr', 'Kerr-Newman']:
                if name in line:
                    current_baseline = name
                    print(f"\n🔄 Testing {current_baseline}...")
                    break
        
        # Track particle tests
        particle = parse_particle_test(line)
        if particle and current_baseline:
            props = get_particle_properties(particle)
            print(f"  • {props['symbol']} ({particle}): ", end='')
            
            # Predict behavior based on baseline and particle
            if 'Schwarzschild' in current_baseline:
                print("geodesic only")
            elif 'Reissner' in current_baseline or 'Newman' in current_baseline:
                if props['charge'] != '0':
                    print(f"geodesic + EM force (charge {props['charge']})")
                else:
                    print("geodesic only (neutral)")
            
            if 'Kerr' in current_baseline:
                if 'Reissner' not in current_baseline and 'Newman' not in current_baseline:
                    print("geodesic + frame-dragging")
        
        # Show validation results
        if "Status: PASSED" in line:
            print("    ✅ Validation PASSED")
        elif "Status: FAILED" in line:
            print("    ❌ Validation FAILED")
        
        # Show completion
        if "Phase 1 Summary" in line:
            print("\n" + "=" * 70)
            print("📈 VALIDATION CHAIN COMPLETE")
            print("=" * 70)

if __name__ == "__main__":
    main() 