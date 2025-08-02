# Proper Validator Logic Fixes

## Design Principle
Validators should check theories based on their **actual physics capabilities**, not hardcoded names.

## Issues Fixed

### 1. Pass/Fail Logic
**Problem**: Validators required theories to BEAT standard models to pass
**Fix**: Theories pass if they meet observational constraints

#### Primordial GWs
```python
# OLD: Must beat standard inflation
result.passed = result.beats_sota and predicted_r < upper_limit

# NEW: Just need to be within observational bounds
result.passed = predicted_r < upper_limit
```

#### CMB Power Spectrum  
```python
# OLD: Must improve on ΛCDM
result.passed = delta_chi2 > 0

# NEW: Different criteria based on theory type
if is_quantum_enabled:
    result.passed = delta_chi2 > 0  # Quantum theories should improve
else:
    result.passed = abs(delta_chi2) < 5.0  # Classical theories should match
```

### 2. Capability Checks
Instead of checking theory names, we check actual capabilities:

#### CMB Requirements
```python
has_cosmological_capability = (
    hasattr(theory, 'get_metric') and 
    hasattr(theory, 'get_ricci_tensor')
)
```

#### Primordial GWs Requirements
```python
has_inflationary_capability = (
    hasattr(theory, 'get_metric') and 
    hasattr(theory, 'get_ricci_tensor') and
    hasattr(theory, 'get_ricci_scalar')
)
```

## Result
- Theories without required methods automatically fail with clear explanation
- Classical GR theories pass when they match standard models
- Quantum theories are expected to improve on standard models
- No hardcoded theory names!