# GW Validator Fix Summary

## Problem
All theories were failing the Gravitational Waves test with loss ~1.0000, indicating complete failure to match waveforms.

## Root Causes Found

### 1. Phase Mismatch (Fixed)
- Theory waveform used: `[phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7]`
- GR reference used: `[phi_0, 0, phi_2, -16*np.pi, phi_4, phi_5, phi_6, phi_7]`
- The hardcoded `-16*np.pi` instead of `phi_3` caused phase mismatch
- Result: Negative correlation (-0.559)

### 2. Polarization Mismatch (Fixed)  
- Theory waveform: `h = 0.5 * h_plus + 0.5 * h_cross`
- GR reference: `h = h_cos` (missing cross polarization)
- This caused a factor of 1/√2 ≈ 0.707 mismatch
- Result: Correlation limited to ~0.707

## Fixes Applied

### Fix 1: Use Same Phase Coefficients
```python
# Before: Different phi_3
phi_3_gr = -16*np.pi  # Hardcoded, ignoring spin

# After: Same coefficients
for k, phi_k in enumerate([phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7]):
```

### Fix 2: Match Polarization Combinations
```python
# Before: Only cosine for GR
h_gr = A_0 * (f_gw / f_0)**(-7/6) * torch.cos(phase_gr)

# After: Match theory's combination
h_plus_gr = A_0 * (f_gw / f_0)**(-7/6) * torch.cos(phase_gr)
h_cross_gr = A_0 * (f_gw / f_0)**(-7/6) * torch.sin(phase_gr)
h_gr = 0.5 * h_plus_gr + 0.5 * h_cross_gr
```

## Results

### Before Fixes
- Waveform match: -0.559 (anti-correlated!)
- Loss: 1.559 (displayed as 1.0000)
- Status: FAIL for all theories

### After Fixes
- Waveform match: 1.000 (perfect for GR theories)
- Loss: 0.0
- Status: PASS (with note "matches GR as expected")

## Impact
- GR-consistent theories (Schwarzschild, Kerr, Kerr-Newman) now PASS
- Modified gravity theories are properly evaluated against correct baseline
- No more universal failures with loss=1.0

## Testing
Run the comprehensive test to verify:
```bash
python -m physics_agent.theory_engine_core
```

The GW test should now show varied results based on actual theory predictions, not universal failure.