#!/usr/bin/env python3
"""
Test dual solver visualization: classical and quantum trajectories for quantum theories.
Ensures Kerr and Kerr-Newman are visible with circular orbits.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Run a test with dual solvers for quantum theories."""
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Test dual solver visualization')
    parser.add_argument('--theory', default='quantum_corrected', 
                       help='Theory to test (default: quantum_corrected)')
    parser.add_argument('--steps', type=int, default=100000,
                       help='Number of integration steps (default: 100000)')
    parser.add_argument('--radius', type=float, default=12.0,
                       help='Initial radius in Schwarzschild radii (default: 12.0)')
    parser.add_argument('--output-dir', default='test_dual_solver_viz',
                       help='Output directory (default: test_dual_solver_viz)')
    args = parser.parse_args()
    
    # Import after parsing args to avoid long startup
    from physics_agent.theory_engine_core import main as run_simulation
    
    # Prepare simulation arguments
    sim_args = [
        '--theories', args.theory,
        '--steps', str(args.steps),
        '--radius', str(args.radius),
        '--output-dir', args.output_dir,
        '--verbose',
        '--all-particles',  # Show all particle types
        '--no-parallel',    # Avoid multiprocessing issues
    ]
    
    print("="*60)
    print("Dual Solver Visualization Test")
    print("="*60)
    print(f"Theory: {args.theory}")
    print(f"Steps: {args.steps}")
    print(f"Initial radius: {args.radius} Schwarzschild radii")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    # Modify sys.argv for the simulation
    original_argv = sys.argv
    try:
        sys.argv = ['test_dual_solver_viz.py'] + sim_args
        run_simulation()
    finally:
        sys.argv = original_argv
    
    print("\n✓ Test completed. Check the output directory for visualizations.")
    
if __name__ == "__main__":
    main()