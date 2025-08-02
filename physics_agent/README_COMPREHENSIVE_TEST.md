# Comprehensive Theory Validation Test

## Overview

The comprehensive validation test is now the **default behavior** when running the theory engine. This test serves as the scientific scorecard for all gravitational theories, combining analytical validators and solver-based tests.

## Default Behavior (NEW)

When you run the theory engine without any special flags:

```bash
python -m physics_agent.theory_engine_core
```

It will:
1. Run the comprehensive validation test for all theories
2. Generate HTML and JSON reports
3. Exit after displaying results

## Running Options

### 1. Default: Run comprehensive test only
```bash
python -m physics_agent.theory_engine_core
```

### 2. Run comprehensive test AND continue with simulation
```bash
python -m physics_agent.theory_engine_core --continue-after-test
```

### 3. Skip comprehensive test (not recommended)
```bash
python -m physics_agent.theory_engine_core --skip-comprehensive-test
```

### 4. Standalone script (alternative method)
```bash
python physics_agent/run_comprehensive_validation.py
```

## What the Test Includes

### Analytical Validators (7 tests)
- Mercury Precession
- Light Deflection
- Photon Sphere
- PPN Parameters
- COW Interferometry
- Gravitational Waves
- PSR J0740

### Solver-Based Tests (5 tests)
- Trajectory vs Kerr (1000-step integration)
- Circular Orbit Period
- CMB Power Spectrum
- Primordial Gravitational Waves
- Quantum Geodesic Simulation

## Output

The test generates:
- **HTML Report**: `comprehensive_theory_validation_[timestamp].html`
- **Latest Report**: `physics_agent/reports/latest_comprehensive_validation.html`
- **JSON Data**: `theory_validation_comprehensive_[timestamp].json`

## Why This is Default

The comprehensive test provides:
1. **Standardized evaluation** of all theories
2. **Scientific rigor** with multiple validation methods
3. **Quick assessment** before running lengthy simulations
4. **Comparative rankings** across all theories
5. **Performance metrics** for computational efficiency

## For Legacy Behavior

If you need the old behavior (running full trajectory simulations):
```bash
python -m physics_agent.theory_engine_core --skip-comprehensive-test
```

However, we recommend running the comprehensive test first to ensure theory validity.