# Trajectory Conversion Bug Fix

## Bug Description
The trajectory unit conversion code has a critical bug that corrupts phi values.

### Current Wrong Code

**When saving to cache (theory_engine_core.py:1474):**
```python
hist_si[:,3] *= self.velocity_scale / self.time_scale  # WRONG! Treats phi as velocity
```

**When loading from cache (theory_engine_core.py:882):**
```python
hist_geom[:,3] *= self.time_scale / self.velocity_scale  # WRONG! Treats phi as velocity
```

### The Problem
- Column 3 is phi (angle in radians), which is dimensionless
- The code treats it as dr/dtau (velocity) and scales it
- This causes phi values to become huge (~1e18) due to scale factors

### Correct Trajectory Format
For spherical coordinates: **[t, r, theta, phi]**
- Column 0: t (time) - needs scaling
- Column 1: r (radius) - needs scaling  
- Column 2: theta (polar angle) - dimensionless
- Column 3: phi (azimuthal angle) - dimensionless

## Fix Required

### When saving (line 1474):
```python
# Remove the line that scales column 3
# hist_si[:,3] *= self.velocity_scale / self.time_scale  # DELETE THIS
```

### When loading (line 882):
```python
# Remove the line that scales column 3
# hist_geom[:,3] *= self.time_scale / self.velocity_scale  # DELETE THIS
```

### Additional Fix
The comment on line 880 "phi is already dimensionless" is correct but refers to the wrong column. Both theta (column 2) and phi (column 3) are dimensionless angles.

## Impact
This bug causes:
1. Phi values to explode to ~1e18 (should be 0 to 2π)
2. WebGL visualization to fail (no visible trajectory)
3. Distance calculations to be wrong
4. Trajectory comparisons to be meaningless

## Verification
After fix, cached trajectories should show:
- t: small values in SI units (primordial BH has tiny timescale)
- r: ~7.4e-12 meters (10M for primordial BH)
- theta: ~1.571 (π/2 for equatorial orbit)
- phi: 0 to ~6.28 (0 to 2π range)