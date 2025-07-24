# Reissner-Nordström Metric

## Overview
The Reissner-Nordström metric is the exact solution to the Einstein-Maxwell equations for a spherically symmetric, non-rotating, electrically charged mass. It serves as the baseline for combined gravitational and electromagnetic effects.

## Mathematical Foundation

The Reissner-Nordström metric in spherical coordinates is:

```
ds² = -f(r)c²dt² + f(r)⁻¹dr² + r²(dθ² + sin²θ dφ²)
```

Where:
```
f(r) = 1 - r_s/r + r_q²/r²
```

And:
- `r_s = 2GM/c²` is the Schwarzschild radius
- `r_q² = GQ²/(4πε₀c⁴)` is the charge radius squared
- `Q` is the electric charge
- `ε₀` is the permittivity of free space

## Physical Interpretation

1. **Gravitational contribution**: The `-r_s/r` term represents pure gravity
2. **Electromagnetic contribution**: The `+r_q²/r²` term represents electrostatic repulsion
3. **Two horizons**: For Q < Q_ext = M√(G/k_e), there are two horizons:
   - Outer horizon: r₊ = (r_s + √(r_s² - 4r_q²))/2
   - Inner horizon: r₋ = (r_s - √(r_s² - 4r_q²))/2

## Role as Dual Baseline

In the compression hypothesis framework:
- Represents combined encoding of gravitational AND electromagnetic information
- Tests whether candidate theories can unify both forces
- A theory with similar losses to both Schwarzschild and RN suggests unification

## Key Properties

- **Exact solution**: Solves both Einstein and Maxwell equations
- **Static**: Time-independent
- **Spherically symmetric**: Preserves angular symmetry
- **Extremal limit**: Q = Q_ext gives a single degenerate horizon
- **Naked singularity**: Q > Q_ext exposes the singularity

## Parameter Choice

For our simulations:
- We use Q ≈ 0.9 × Q_ext (approximately 4.878e21 C)
- This ensures strong electromagnetic effects while avoiding naked singularity
- Creates distinct dynamics from pure Schwarzschild case

## Unification Test

The key insight: A theory that achieves similar losses against both baselines without explicitly including charge suggests it captures the unified nature of gravity and electromagnetism through geometry alone. 