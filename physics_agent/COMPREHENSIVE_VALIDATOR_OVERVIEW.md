# Comprehensive Validator Overview: Solver Usage and Test Types

## Summary

This document provides a comprehensive overview of all validators showing:
- Which validators use trajectory/geodesic solvers
- Which validators bypass solvers and use analytical methods
- Additional tests from test_geodesic_validator_comparison.py

## Validator Overview Table

| Validator | Category | Uses Solver? | Solver Type | Method | In test_theories_final.py | In test_geodesic_validator_comparison.py |
|-----------|----------|--------------|-------------|---------|------------------------|----------------------------------------|
| **Conservation Validator** | constraint | ✅ Yes | Geodesic | Uses pre-computed trajectory from engine | ✅ | ❌ |
| **Metric Properties Validator** | constraint | ❌ No | N/A | Analytical metric evaluation | ✅ | ❌ |
| **Mercury Precession Validator** | observational | ❌ No* | N/A | Analytical weak-field approximation | ✅ | ✅ |
| **Light Deflection Validator** | observational | ❌ No | N/A | Analytical PPN calculation | ✅ | ✅ |
| **Photon Sphere Validator** | observational | ❌ No | N/A | Analytical effective potential extremum | ✅ | ✅ |
| **PPN Parameters Validator** | observational | ❌ No | N/A | Analytical weak-field expansion | ✅ | ✅ |
| **COW Interferometry Validator** | observational/quantum | ❌ No | N/A | Analytical metric gradient calculation | ✅ | ✅ |
| **Gravitational Wave Validator** | observational | ❌ No | N/A | Analytical post-Newtonian waveforms | ✅ | ✅ |
| **PSR J0740 Validator** | observational | ❌ No | N/A | Analytical Shapiro delay calculation | ✅ | ✅ |
| **CMB Power Spectrum Validator** | prediction | ✅ Yes* | Quantum Path Integral | Optional quantum path integrator for quantum theories | ❌ | ✅ |
| **Primordial GWs Validator** | prediction | ✅ Yes* | Quantum Path Integral | Optional quantum corrections | ❌ | ✅ |
| **Circular Orbit Period Test** | benchmark | ✅ Yes | Geodesic | Direct geodesic integration | ❌ | ✅ |
| **Trajectory Cache Test** | performance | ✅ Yes | Geodesic | Tests caching system | ❌ | ✅ |
| **Quantum Geodesic Simulator Test** | quantum | ✅ Yes | Quantum Geodesic | Tests quantum geodesic simulator | ❌ | ✅ |

## Detailed Analysis

### Validators That Use Solvers

1. **Conservation Validator**
   - Uses pre-computed geodesic trajectory from engine
   - Tests energy and angular momentum conservation along the trajectory
   - Solver type: Standard geodesic integrator (RK4)

2. **CMB Power Spectrum Validator** (quantum theories only)
   - Optionally uses QuantumPathIntegrator for quantum-enabled theories
   - Computes quantum corrections to power spectrum
   - Falls back to analytical if quantum integration fails

3. **Primordial GWs Validator** (quantum theories only)
   - Similar to CMB, uses quantum path integrals for quantum theories
   - Computes quantum amplitude corrections

4. **Test-only validators from test_geodesic_validator_comparison.py:**
   - **Circular Orbit Period Test**: Direct geodesic integration
   - **Trajectory Cache Test**: Tests the caching system
   - **Quantum Geodesic Simulator Test**: Tests quantum corrections

### Validators That Bypass Solvers (Analytical)

1. **Metric Properties Validator**
   - Directly evaluates metric components at various radii
   - No trajectory integration needed

2. **Mercury Precession Validator**
   - Uses analytical weak-field approximation
   - Note: Comment says "simplified approach - full calculation would integrate geodesics"

3. **Light Deflection Validator**
   - Uses PPN formalism with analytical calculation
   - Extracts gamma parameter from metric

4. **Photon Sphere Validator**
   - Finds extremum of effective potential analytically
   - No trajectory integration

5. **PPN Parameters Validator**
   - Weak-field expansion of metric
   - Extracts parameters analytically

6. **COW Interferometry Validator**
   - Calculates metric gradient numerically
   - Uses analytical quantum phase formula

7. **Gravitational Wave Validator**
   - Analytical post-Newtonian waveforms
   - No numerical integration

8. **PSR J0740 Validator**
   - Analytical Shapiro delay calculation
   - Based on metric components

## Key Insights

1. **Most validators use analytical methods** - Only Conservation validator always uses trajectories
2. **Quantum validators optionally use solvers** - CMB and Primordial GWs use quantum path integrals for quantum theories
3. **Benchmark tests use solvers** - Tests in test_geodesic_validator_comparison.py directly test the solvers
4. **No duplication** - test_theories_final.py includes active validators, while test_geodesic_validator_comparison.py tests the solvers themselves

## Performance Implications

- **Fast validators** (analytical): Most observational tests complete in milliseconds
- **Slow validators** (solver-based): Conservation validator depends on pre-computed trajectory
- **Variable validators**: CMB/Primordial GWs are fast for classical theories, slower for quantum
- **Caching critical**: Trajectory cache provides 1000x+ speedup for solver-based tests

## Notes

- The asterisk (*) indicates validators that conditionally use solvers based on theory type
- test_geodesic_validator_comparison.py serves as the solver validation suite
- All validators must pass solver tests before being included in the main validation suite