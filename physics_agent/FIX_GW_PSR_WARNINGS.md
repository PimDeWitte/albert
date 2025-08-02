# Fix for Constant GW and PSR J0740 WARNING Values

## Issues Identified

### 1. GW Validator (WARNING with loss 0.1000)
- The validator checks if theory implements `gw_speed`, `gw_damping`, or `gw_modifications`
- If none are found, it returns a **hardcoded** WARNING with loss 0.1
- This means ALL theories without these specific methods get the same result!

### 2. PSR J0740 Validator (WARNING with loss 5.2190)
- All theories are computing the same Shapiro delay (~18.66 μs)
- This exceeds the 3 μs tolerance by factor of 6.219
- Loss = max_delay/tolerance - 1.0 = 6.219 - 1.0 = 5.219
- This suggests all theories are using the same GR formula

## Root Cause

Both validators are expecting theories to implement specific methods/attributes that modify their predictions. When these aren't found, they either:
1. Return a default penalty (GW)
2. Fall back to standard GR calculations (PSR J0740)

## Solutions

### Option 1: Accept GR-consistent behavior (Quick Fix)
For theories that are GR-consistent (Schwarzschild, Kerr, etc.), these warnings are actually correct - they should match GR predictions. We can handle this in the test interpretation.

### Option 2: Fix the validators (Better Long-term)
1. **GW Validator**: Should actually compute GR waveforms for theories without modifications, not return a hardcoded penalty
2. **PSR J0740**: Should check if the theory is modifying the metric properly, not just using default GR formulas

### Option 3: Update tolerance (For PSR J0740)
The 3 μs tolerance might be too strict. The actual timing RMS is 0.32 μs, but the Shapiro delay itself could be larger.

## Immediate Workaround

Treat these specific WARNING values as expected behavior for GR-consistent theories.