# Default Black Hole Configuration

## Default: `primordial_mini`

The default black hole preset for all simulations is **primordial_mini**.

### Why This Default?

1. **Numerical Stability**: 
   - Mass: 10^15 kg (asteroid mass)
   - Allows larger timesteps (0.1 in geometric units)
   - Reduces numerical errors in integration

2. **Quantum Gravity Research**:
   - Schwarzschild radius: 1.5×10^-12 m (subatomic scale)
   - Quantum effects become significant
   - Ideal for testing quantum gravity theories

3. **Computational Efficiency**:
   - Orbital periods in femtoseconds
   - 1000-step simulations complete quickly
   - Suitable for rapid testing and development

4. **Physical Relevance**:
   - Hypothetical primordial black holes could exist
   - Mass range where quantum and gravitational effects compete
   - Probes physics at the Planck scale

### Usage

```python
# Default usage (automatically uses primordial_mini)
engine = TheoryEngine()

# Explicit override if needed
engine = TheoryEngine(black_hole_preset='stellar_mass')
```

### Command Line

```bash
# Default usage
python -m physics_agent.theory_engine_core

# Override default
python -m physics_agent.theory_engine_core --black-hole-preset stellar_mass
```

### Alternative Presets

If you need different physics:
- `stellar_mass`: For astrophysical simulations
- `laboratory_micro`: For extreme quantum regime
- `intermediate_mass`: For globular cluster dynamics
- `sagittarius_a_star`: For galactic center physics

The default `primordial_mini` provides the best balance of numerical stability, computational efficiency, and physical relevance for quantum gravity research.