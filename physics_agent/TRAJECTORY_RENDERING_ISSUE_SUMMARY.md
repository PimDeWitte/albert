# Trajectory Rendering Issue Summary

## Problem
The WebGL visualization is not showing trajectories because the trajectory data appears to be corrupted with:
- All zeros for radial coordinate (r = 0.000)
- Extremely large values for phi coordinate (e.g., -161240771381753600.000)
- This happens for both cached and freshly computed trajectories

## Investigation Results

### 1. Distance Traveled Calculation Added ✓
Successfully added distance traveled calculation to:
- `test_comprehensive_final.py`: Calculates 3D distance using spherical to Cartesian conversion
- `comprehensive_test_report_generator.py`: Shows "Distance Traveled (Theory / Kerr)" column in table

### 2. Trajectory Data Format Issue ❌
The trajectory data returned by `run_trajectory()` has format issues:
```
Trajectory shape: torch.Size([101, 4])
Features per step: 4
Step 0: t=0.000, r=0.000, theta=0.000, phi=0.000
Step 1: t=0.000, r=0.000, theta=-0.004, phi=-161240771381753568.000
```

Expected format should be:
- Column 0: time (tau)
- Column 1: radial distance (r) - should be ~10 for circular orbit
- Column 2: theta angle - should be ~π/2 for equatorial orbit
- Column 3: phi angle - should increase smoothly

### 3. Root Causes Identified

1. **Unit Conversion Issue**: The trajectory data may be in the wrong units or improperly converted
2. **Cache Corruption**: Both cached and fresh trajectories show the same problem
3. **Solver Output Format**: The ConservedQuantityGeodesicSolver may be returning data in a different format than expected

## Recommendations

### Immediate Fix Needed
1. Debug the `run_trajectory()` method in `theory_engine_core.py` to understand the actual output format
2. Fix the unit conversion between geometric and SI units
3. Ensure the solver returns proper spherical coordinates

### WebGL Visualization Update
Once trajectory data is fixed:
1. The viewer already expects [r, phi] for 2D projection
2. Loss calculation per step is already implemented
3. Distance traveled will show meaningful values

### Test With Simple Case
```python
# Expected circular orbit at r=10M:
# r should stay ~10
# theta should stay ~π/2
# phi should increase from 0 to ~2π over many steps
```

## Current Status
- ✅ Distance traveled calculation implemented
- ✅ Report table updated with distance column
- ❌ Trajectory data corrupted/wrong format
- ❌ WebGL viewer cannot render invalid data

The WebGL viewer code is ready - it just needs valid trajectory data to visualize.