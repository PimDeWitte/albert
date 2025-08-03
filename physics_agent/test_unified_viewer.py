#!/usr/bin/env python3
"""Test the unified multi-particle viewer generator."""

import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.ui.unified_multi_particle_viewer_generator import generate_unified_multi_particle_viewer

# Test with the latest comprehensive test run
run_dirs = [
    'physics_agent/runs/comprehensive_test_20250803_153145',
    'physics_agent/runs/comprehensive_test_20250803_152848',
    'physics_agent/runs/comprehensive_test_20250803_151738'
]

for run_dir in run_dirs:
    if os.path.exists(run_dir):
        print(f"\nTesting with {run_dir}")
        output_path = 'test_unified_viewer.html'
        
        result = generate_unified_multi_particle_viewer(
            run_dir=run_dir,
            output_path=output_path
        )
        
        if result:
            print(f"Successfully generated: {result}")
            print(f"File size: {os.path.getsize(result):,} bytes")
            # Try to open it
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(result)}')
            break
        else:
            print("Failed to generate viewer")
    else:
        print(f"Directory not found: {run_dir}")
else:
    print("No valid run directories found")