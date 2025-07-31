#!/usr/bin/env python3
"""
Fix for particle initial radius calculation.

The issue: When args.radius = 12 (meaning 12 Schwarzschild radii), the code does:
  r0_val = args.radius * engine.rs
  
But engine.rs = 2.0 (geometric units), not the actual Schwarzschild radius!
This gives r0_val = 24 meters, which is only 0.008 Schwarzschild radii.

The fix: Use the actual Schwarzschild radius in meters.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.theory_engine_core import TheoryEngine


def demonstrate_issue():
    """Show the current radius calculation issue."""
    
    engine = TheoryEngine()
    
    print("CURRENT ISSUE:")
    print("="*60)
    
    # Current incorrect calculation
    radius_rs = 12.0  # User wants 12 Schwarzschild radii
    r0_val_wrong = radius_rs * engine.rs  # WRONG: uses rs=2.0 (geometric)
    
    print(f"User specifies: {radius_rs} Schwarzschild radii")
    print(f"engine.rs = {engine.rs} (geometric units where M=1)")
    print(f"Current calculation: r0_val = {radius_rs} * {engine.rs} = {r0_val_wrong} meters")
    
    # Convert to actual Schwarzschild radii
    rs_actual = 2 * engine.G_si * engine.M_si / engine.c_si**2
    r_in_rs = r0_val_wrong / rs_actual
    
    print(f"\nActual Schwarzschild radius = {rs_actual:.1f} meters")
    print(f"So r0_val = {r0_val_wrong} m = {r_in_rs:.4f} Schwarzschild radii")
    print(f"ERROR: User wanted 12 rs but got {r_in_rs:.4f} rs!")
    
    print("\n" + "="*60)
    print("CORRECT CALCULATION:")
    print("="*60)
    
    # Correct calculation
    r0_val_correct = radius_rs * rs_actual
    
    print(f"User specifies: {radius_rs} Schwarzschild radii")
    print(f"Actual rs = {rs_actual:.1f} meters")
    print(f"Correct: r0_val = {radius_rs} * {rs_actual:.1f} = {r0_val_correct:.1f} meters")
    print(f"Verification: {r0_val_correct:.1f} m / {rs_actual:.1f} m = {r0_val_correct/rs_actual:.1f} rs ✓")
    
    # Show geometric conversion
    r0_geom = r0_val_correct / engine.length_scale
    print(f"\nIn geometric units: r0_geom = {r0_val_correct:.1f} / {engine.length_scale:.1f} = {r0_geom:.1f}")
    
    print("\n" + "="*60)
    print("RECOMMENDED INITIAL RADII FOR QUANTUM EFFECTS:")
    print("="*60)
    
    radii_to_test = [6, 10, 20, 50, 100]
    
    for r_rs in radii_to_test:
        r_meters = r_rs * rs_actual
        r_geom = r_meters / engine.length_scale
        
        physics = ""
        if r_rs < 10:
            physics = "Strong field, significant quantum effects"
        elif r_rs < 50:
            physics = "Moderate field, some quantum effects"
        else:
            physics = "Weak field, minimal quantum effects"
            
        print(f"\n{r_rs:3d} rs = {r_meters:8.0f} m = {r_geom:6.1f} (geometric)")
        print(f"       {physics}")


def create_patch():
    """Create a patch file to fix the issue in theory_engine_core.py"""
    
    patch_content = '''
# Patch for theory_engine_core.py - Fix initial radius calculation

In the main() function around line 4030, replace:

OLD CODE:
--------
    if args.close_orbit:
        r0_val = 6.0 * engine.rs
    else:
        # <reason>chain: Use user-specified radius or default to 6.0 Rs</reason>
        r0_val = args.radius * engine.rs  # User-specified radius in Schwarzschild radii

NEW CODE:
--------
    # <reason>chain: Calculate actual Schwarzschild radius in meters</reason>
    rs_meters = 2 * engine.G_si * engine.M_si / engine.c_si**2
    
    if args.close_orbit:
        r0_val = 6.0 * rs_meters  # 6 Schwarzschild radii in meters
    else:
        # <reason>chain: Use user-specified radius in actual Schwarzschild radii</reason>
        r0_val = args.radius * rs_meters  # Convert rs to meters using actual rs value

This ensures particles start at the correct radius!
'''
    
    print("\n" + "="*60)
    print("PATCH TO FIX THE ISSUE:")
    print("="*60)
    print(patch_content)
    
    # Save patch file
    with open('fix_initial_radius.patch', 'w') as f:
        f.write(patch_content)
    
    print("\nPatch saved to: fix_initial_radius.patch")


def main():
    """Main function to demonstrate and fix the issue."""
    
    parser = argparse.ArgumentParser(description='Fix particle initial radius calculation')
    parser.add_argument('--create-patch', action='store_true', 
                       help='Create a patch file to fix the issue')
    args = parser.parse_args()
    
    demonstrate_issue()
    
    if args.create_patch:
        create_patch()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("The issue is that engine.rs = 2.0 (geometric units) is being used")
    print("instead of the actual Schwarzschild radius = 2953 meters.")
    print("\nThis causes particles to start 1477x closer than intended!")
    print("For quantum effects, we want particles at 6-100 Schwarzschild radii.")


if __name__ == "__main__":
    main()