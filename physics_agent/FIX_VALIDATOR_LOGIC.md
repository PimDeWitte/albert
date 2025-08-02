# Validator Logic Fixes

## Issues Found

### 1. Primordial GWs Validator
**Problem**: Requires theories to BEAT standard inflation to pass
- GR theories (Schwarzschild) predict r=0.010 (same as standard inflation)
- Quantum theories also predict r=0.010  
- But only theories that BEAT standard (delta_lnL > 0) pass
- So identical predictions have different outcomes!

**Current Logic**:
```python
result.beats_sota = delta_lnL > 0  # Must improve
result.passed = result.beats_sota and predicted_r < upper_limit
```

**Fix Needed**: Matching standard inflation should also pass
```python
result.beats_sota = delta_lnL > 0
result.passed = predicted_r < upper_limit  # Just need to be within observational limits
```

### 2. CMB Power Spectrum Validator  
**Problem**: Requires improvement over ΛCDM to pass
- GR theories correctly predict ΛCDM (Δχ² = 0)
- But they fail because they don't "improve"

**Current Logic**:
```python
result.passed = delta_chi2 > 0 and theory_chi2 < 100  # Must improve
```

**Fix Needed**: Matching ΛCDM should pass for GR theories
```python
# For GR-consistent theories, matching is success
is_gr_consistent = theory.name in ['Schwarzschild', 'Kerr', 'Kerr-Newman'] or not hasattr(theory, 'enable_quantum')
if is_gr_consistent:
    result.passed = abs(delta_chi2) < 1.0  # Close match to ΛCDM
else:
    result.passed = delta_chi2 > 0  # Quantum theories should improve
```

### 3. Newtonian Limit Issue
**Problem**: Newtonian Limit shouldn't pass CMB/Primordial tests at all
- These are relativistic/quantum phenomena
- Newtonian gravity can't predict them

**Fix Needed**: Add capability check
```python
# Check if theory can handle cosmological scales
if 'Newtonian' in theory.name:
    result.passed = False
    result.notes = "Newtonian gravity cannot predict cosmological phenomena"
```

## Root Cause
The validators were designed with quantum gravity theories in mind, expecting them to IMPROVE on standard models. But GR-consistent theories should MATCH standard models, not beat them.