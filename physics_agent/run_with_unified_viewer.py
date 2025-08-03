#!/usr/bin/env python3
"""
Consolidated entry point for running theories with unified viewer generation.
This demonstrates the primary workflow for the albert command.
"""

import sys
import os
import subprocess

def main():
    """Run the complete workflow with unified viewer generation."""
    
    print("=" * 80)
    print("🌌 Albert: Running Comprehensive Theory Validation")
    print("=" * 80)
    print("\nThis will:")
    print("1. Run comprehensive tests on all theories")
    print("2. Generate individual trajectory viewers")
    print("3. Generate the unified multi-particle viewer")
    print("4. Create comprehensive reports and leaderboards")
    print("\n" + "=" * 80 + "\n")
    
    # Run using the albert command (or python -m physics_agent.theory_engine_core)
    try:
        # Check if albert command is available
        result = subprocess.run(['which', 'albert'], capture_output=True)
        if result.returncode == 0:
            # Use albert command
            print("Running: albert run")
            cmd = ['albert', 'run']
        else:
            # Fall back to python module
            print("Running: python -m physics_agent.theory_engine_core")
            cmd = [sys.executable, '-m', 'physics_agent.theory_engine_core']
        
        # Add any additional arguments passed to this script
        if len(sys.argv) > 1:
            cmd.extend(sys.argv[1:])
        
        # Run the command
        subprocess.run(cmd)
        
        # After completion, find and display the latest results
        print("\n" + "=" * 80)
        print("✅ Run complete!")
        print("=" * 80)
        
        # Find the latest run directory
        runs_dir = 'physics_agent/runs'
        if os.path.exists(runs_dir):
            run_dirs = [d for d in os.listdir(runs_dir) if d.startswith('comprehensive_test_')]
            if run_dirs:
                run_dirs.sort()
                latest_run = run_dirs[-1]
                run_path = os.path.join(runs_dir, latest_run)
                
                print(f"\nResults saved to: {run_path}")
                
                # Check for key files
                report_path = os.path.join(run_path, f'comprehensive_theory_validation_{latest_run.split("_", 2)[2]}.html')
                unified_viewer_path = os.path.join(run_path, 'trajectory_viewers', 'unified_multi_particle_viewer_advanced.html')
                
                if os.path.exists(report_path):
                    print(f"\n📊 Comprehensive Report:")
                    print(f"   {report_path}")
                
                if os.path.exists(unified_viewer_path):
                    print(f"\n🌍 Unified Multi-Particle Viewer:")
                    print(f"   {unified_viewer_path}")
                    
                    # On macOS, offer to open the viewer
                    if sys.platform == 'darwin':
                        response = input("\nOpen unified viewer in browser? (y/n): ")
                        if response.lower() == 'y':
                            subprocess.run(['open', unified_viewer_path])
                
                print("\n💡 Tip: The unified viewer shows all theories in one interactive 3D visualization!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()