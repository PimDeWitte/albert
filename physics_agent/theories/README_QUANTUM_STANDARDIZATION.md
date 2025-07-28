# Quantum Theory Parameter Standardization

## Problem

Different quantum gravity theories use different parameters and scales to control quantum effects:

- **Quantum Corrected**: Uses `alpha` (preferred: 1.0)
- **Log-Corrected QG**: Uses `gamma` (preferred: 1e-5)  
- **Einstein Asymmetric**: Uses `alpha` (preferred: 0.0)
- **Post-Quantum Gravity**: Uses `gamma` with different range
- **Fractal**: Uses `D` (fractal dimension)
- **Phase Transition**: Uses `T_c` (critical temperature)
- **String Theory**: Uses `alpha_prime` (string tension)

This makes fair comparison difficult since `alpha=1.0` in one theory might represent much stronger quantum effects than `gamma=1e-5` in another.

## Proposed Standardization

### Option 1: Normalized Quantum Strength Parameter

Define a universal "quantum strength" parameter `q_strength` ∈ [0, 1] where:
- 0 = Classical limit (no quantum effects)
- 0.5 = Moderate quantum corrections
- 1.0 = Maximum quantum effects

Each theory would map this to its internal parameter:
```python
# In Quantum Corrected
alpha = 2.0 * q_strength  # Maps [0,1] to [0,2]

# In Log-Corrected  
gamma = 0.1 * q_strength  # Maps [0,1] to [0,0.1]

# In String Theory
alpha_prime = 1e-66 * (10 ** q_strength)  # Log scale mapping
```

### Option 2: Physics-Based Standardization

Use a physical observable to standardize, such as:
- Deviation from Schwarzschild metric at r=10 Rs
- Modification to Newton's constant at weak field
- Quantum correction to photon sphere radius

Set parameters so all theories produce the same % deviation.

### Option 3: Performance-Based Calibration

Calibrate each theory's parameter to achieve:
- Same trajectory deviation from GR at fixed radius
- Same PPN parameter modifications
- Same quantum uncertainty magnitude

## Implementation Suggestions

1. Add a `standardize_parameters(q_strength)` method to each quantum theory
2. Use a consistent default `q_strength` for leaderboard comparisons
3. Document the mapping for each theory
4. Consider separate leaderboards for different quantum strengths

## Current Status

Currently, theories use their individual `preferred_params` which may not represent comparable quantum effect strengths. This could bias the leaderboard toward theories with conservative default parameters. 