#!/usr/bin/env python3
"""Test the complete unified viewer workflow."""

import os
import sys
import subprocess

def test_unified_workflow():
    """Test the complete workflow with unified viewer generation."""
    
    print("=" * 80)
    print("Testing Unified Multi-Particle Viewer Workflow")
    print("=" * 80)
    
    # Run a quick comprehensive test
    print("\nRunning comprehensive test with unified viewer generation...")
    print("This will generate the unified_multi_particle_viewer_advanced.html\n")
    
    # Use the albert command with quick test parameters
    cmd = [
        sys.executable, "-m", "physics_agent.evaluation",
        "--quick"  # If this flag exists, otherwise we'll run a minimal test
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error running comprehensive test:")
            print(result.stderr)
            return False
            
        # Check if the unified viewer was generated
        run_dirs = [d for d in os.listdir('physics_agent/runs') if d.startswith('comprehensive_test_')]
        if not run_dirs:
            print("No run directories found!")
            return False
            
        # Get the most recent run
        run_dirs.sort()
        latest_run = run_dirs[-1]
        run_dir = os.path.join('physics_agent/runs', latest_run)
        
        # Check for unified viewer
        unified_viewer_path = os.path.join(run_dir, 'trajectory_viewers', 'unified_multi_particle_viewer_advanced.html')
        
        if os.path.exists(unified_viewer_path):
            print(f"✅ Success! Unified viewer generated at:")
            print(f"   {unified_viewer_path}")
            print(f"\nFile size: {os.path.getsize(unified_viewer_path):,} bytes")
            
            # Check that it contains actual data
            with open(unified_viewer_path, 'r') as f:
                content = f.read()
                if 'runData' in content and 'theories' in content:
                    print("✅ Viewer contains theory data")
                else:
                    print("❌ Viewer appears to be missing data")
                    return False
                    
            return True
        else:
            print(f"❌ Unified viewer not found at expected location:")
            print(f"   {unified_viewer_path}")
            
            # List what's in the trajectory_viewers directory
            viewers_dir = os.path.join(run_dir, 'trajectory_viewers')
            if os.path.exists(viewers_dir):
                print(f"\nContents of {viewers_dir}:")
                for f in os.listdir(viewers_dir):
                    print(f"   - {f}")
            else:
                print(f"\nDirectory not found: {viewers_dir}")
                
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_unified_workflow()
    sys.exit(0 if success else 1)