# Fix for CMB and PGW Test Failures

## The Problem

The CMB and PGW validators are designed to look for theories that **beat** the standard model (ΛCDM for CMB, standard inflation for PGW). This means:

1. **CMB Test**: Only passes if `delta_chi2 > 0.1` (improvement over ΛCDM)
2. **PGW Test**: Only passes if `beats_sota = true` AND `r < upper_limit`

This causes GR-consistent theories (Schwarzschild, Kerr, etc.) to FAIL because they match the standard model rather than beat it!

## The Solution

We need to modify the test interpretation to handle three cases:

1. **PASS**: Theory beats standard model OR matches it within tolerance
2. **FAIL**: Theory is worse than standard model by significant margin
3. **SKIP**: Data unavailable

## Temporary Workaround

For now, we can modify the comprehensive test to treat these specific failures differently:

```python
# In test_comprehensive_final.py
# For CMB and PGW tests, check if theory is GR-consistent
# If so, expect it to match (not beat) standard model
```

## Long-term Fix

The validators themselves should be updated to:
1. Pass theories that match the standard model within error bars
2. Give bonus points for beating SOTA, but not require it
3. Only fail theories that are significantly worse

This would better reflect the scientific reality where matching observations is success, and beating the standard model is exceptional.