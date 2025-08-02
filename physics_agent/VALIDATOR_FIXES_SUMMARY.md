# Validator Fixes Summary

## Overview

We've fixed the fundamental issues in both the GW and PSR J0740 validators that were causing them to return constant WARNING values across all theories.

## GW Validator Fixes

### Problem
- Returned hardcoded `WARNING` with loss 0.1 for ANY theory without specific GW modification methods
- Never actually computed waveforms for theories without modifications

### Fix Applied
1. **Removed hardcoded penalty**: The validator now continues to compute standard GR waveforms even if the theory has no modifications
2. **Added proper thresholds**:
   - PASS: match > 0.97 (excellent match)
   - WARNING: 0.95 < match ≤ 0.97 (good match)
   - FAIL: match ≤ 0.95 (poor match)
3. **Special handling for GR theories**: If no modifications and match > 0.99, marked as PASS with note "(matches GR as expected)"

### Result
- GR-consistent theories now get proper waveform correlation scores
- No more universal 0.1 loss penalty
- Each theory evaluated on its actual waveform predictions

## PSR J0740 Validator Fixes

### Problem
- Used unrealistic 3 microsecond absolute tolerance
- All theories computed ~18.7 μs Shapiro delay (correct GR value)
- This exceeded tolerance by factor of 6.22, giving loss = 5.219 for everyone

### Fix Applied
1. **Changed to relative error comparison**:
   - Compare against expected GR value (18.7 μs)
   - PASS: < 2% deviation
   - WARNING: 2-10% deviation  
   - FAIL: > 10% deviation
2. **Removed absolute tolerance trap**: The Shapiro delay itself is much larger than timing precision - what matters is measuring it accurately, not its absolute value

### Result
- GR-consistent theories now PASS when they correctly predict ~18.7 μs
- Modified gravity theories evaluated on how much they deviate from GR
- No more universal 5.219 loss

## Technical Details

### GW Validator Changes
```python
# Before: Hardcoded penalty
if not theory_has_modifications:
    return {"loss": 0.1, "flags": {"overall": "WARNING"}, ...}

# After: Continue with computation
if not theory_has_modifications:
    print("Note: Computing standard GR waveform")
# ... continues to compute actual waveform ...
```

### PSR J0740 Changes
```python
# Before: Absolute tolerance
if max_delay > tolerance:  # 18.7 μs > 3 μs always fails!
    loss = max_delay / tolerance - 1.0  # Always 5.219

# After: Relative error
gr_shapiro_delay = 18.7e-6
relative_error = abs(max_delay - gr_shapiro_delay) / gr_shapiro_delay
if relative_error > 0.1:  # 10% threshold
    loss = relative_error * 10
```

## Impact

With these fixes:
1. Each theory is evaluated on its actual predictions, not hardcoded values
2. GR-consistent theories properly pass when they match GR
3. Modified theories are evaluated on how much they deviate from GR
4. No more suspicious constant WARNING values across all theories

## Testing

To verify these fixes work correctly:
```bash
python -m physics_agent.theory_engine_core
```

The comprehensive test should now show:
- Varied GW correlation scores based on actual waveforms
- Varied PSR J0740 results based on actual Shapiro delay predictions
- GR theories (Schwarzschild, Kerr) passing both tests