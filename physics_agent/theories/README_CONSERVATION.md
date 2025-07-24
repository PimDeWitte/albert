# Conservation Validator Guidelines

## For Theory Developers

The conservation validator now uses a strict scientific tolerance (1e-5) for all theories. If your theory has physical reasons for conservation violations (e.g., stochastic elements, quantum corrections), you must implement these methods:

### Method 1: Compute Conservation Violation

```python
def computes_conservation_violation(self, hist: torch.Tensor) -> float:
    """
    Compute the expected conservation violation based on physical principles.
    
    Args:
        hist: Trajectory history [t, r, phi, dr/dtau]
        
    Returns:
        Expected relative violation magnitude
    """
    # Example for stochastic theory
    if hasattr(self, 'sigma'):
        # Expected drift scales with noise level and trajectory length
        return self.sigma * np.sqrt(hist.shape[0])
    return 0.0

def conservation_violation_mechanism(self) -> str:
    """Return physical mechanism causing conservation violation."""
    return "Stochastic spacetime fluctuations"
```

### Method 2: Flag Stochastic Elements

```python
def has_stochastic_elements(self) -> bool:
    """Return True if theory has inherent randomness."""
    return hasattr(self, 'sigma') and self.sigma > 0
```

## Examples

### Stochastic Theory
```python
class StochasticTheory(GravitationalTheory):
    def __init__(self, sigma=0.01):
        self.sigma = sigma
        
    def has_stochastic_elements(self):
        return True
        
    def computes_conservation_violation(self, hist):
        # Brownian motion-like drift
        return self.sigma * np.sqrt(hist.shape[0] * self.dtau)
```

### Quantum Corrected Theory
```python
class QuantumCorrected(GravitationalTheory):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        
    def computes_conservation_violation(self, hist):
        # Quantum corrections at small radii
        r_min = torch.min(hist[:, 1])
        if r_min < 10:  # Near horizon
            return self.alpha * (10 / r_min) * 1e-6
        return 0.0
        
    def conservation_violation_mechanism(self):
        return "Quantum backreaction near horizon"
```

## Important Notes

1. **No hardcoded tolerance adjustments** - The validator uses 1e-5 for all theories
2. **Physical justification required** - Conservation violations must have a physical basis
3. **Quantitative predictions** - Return actual expected violation magnitudes
4. **Document mechanisms** - Explain the physics behind violations

Theories without these methods will be held to strict conservation (1e-5 tolerance). 