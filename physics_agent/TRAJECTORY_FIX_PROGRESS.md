# Trajectory Fix Progress Report

## Issues Fixed So Far

### 1. ✅ Unit Conversion Bug
- **Fixed**: Removed incorrect scaling of phi values as velocities
- **File**: theory_engine_core.py
- **Lines**: 882 and 1474

### 2. ✅ State Vector Format Confusion  
- **Fixed**: Corrected 6D to 4D conversion to properly insert theta=π/2
- **File**: theory_engine_core.py  
- **Lines**: 1265 and 1419

### 3. ✅ Initial State Format
- **Fixed**: Changed 4D initial state from [t, r, phi, dr/dtau] to [t, r, theta, phi]
- **File**: theory_engine_core.py
- **Line**: 1108

## Current Status

After fixes, trajectories now show:
```
Step 0: t=0.000, r=0.000, theta=1.571, phi=0.000        ✓ Correct initial state
Step 1: t=0.000, r=0.000, theta=1.567, phi=-161240771381753568.000  ✗ Phi corrupted
```

### What's Working
- r = 7.426e-12 meters (correct for 10M in SI units)
- theta = 1.571 (π/2, correct for equatorial)
- Initial state is correct

### What's Still Broken
- phi values become huge negative numbers after first step
- This suggests the solver is outputting wrong values or in wrong format

## Next Steps

The ConservedQuantityGeodesicSolver might be:
1. Using a different state format internally
2. Outputting velocities instead of positions
3. Having numerical overflow issues

Need to check:
- What format does ConservedQuantityGeodesicSolver expect/return?
- Is there a mismatch between solver input/output format?
- Are there numerical stability issues causing overflow?

## Test Results
- Clearing cache and regenerating shows the issue persists
- Problem is in the computation, not just cached data
- The huge phi values (~1e17) suggest they might be accumulated velocities or numerical errors