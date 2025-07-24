# Solver Tests

This directory contains comprehensive tests for the geodesic solvers and trajectory computation engines.

## Test Files

1. **test_geodesic_validator_comparison.py** - Validates geodesic integrator against theoretical predictions and validator results
2. **benchmark_validators.py** - Performance benchmarks for different validators
3. **benchmark_cache_circular_orbit.py** - Demonstrates trajectory caching performance improvements
4. **test_cache_simple.py** - Simple test to verify caching functionality

## Trajectory Caching System

### Overview

The trajectory caching system provides **dramatic performance improvements** for gravitational physics simulations:

- **10,000 steps**: 1,109.9x speedup (2.75s → 2.5ms)
- **100,000 steps**: 10,673.7x speedup (25.80s → 2.4ms)
- **1,000,000 steps**: 29,323.3x speedup (4m 12.7s → 8.6ms)

**Average speedup: 13,702.3x** 🚀

### How It Works

1. **Intelligent Hashing**: Creates unique cache keys based on:
   - Theory name and parameters
   - Initial radius (r₀)
   - Number of steps
   - Time step size (Δτ)
   - Additional parameters (spin, charge, etc.)

2. **Persistent Storage**: Cached trajectories stored as PyTorch tensors
   - Survives across runs and engine instances
   - Version-controlled to prevent stale data

3. **Automatic Management**: No manual intervention required
   - Cache hits detected automatically
   - New trajectories cached on first computation

### Running the Tests

```bash
# Run all geodesic validation tests
python test_geodesic_validator_comparison.py

# Run cache performance benchmark
python benchmark_cache_circular_orbit.py

# Run simple cache verification
python test_cache_simple.py

# Run validator benchmarks
python benchmark_validators.py
```

### Key Results

From `test_geodesic_validator_comparison.py`:

| Test | Result | Time |
|------|--------|------|
| Circular Orbit Period | PASSED | 0.3s |
| Mercury Precession | PASSED | 0.1s |
| Light Deflection | PASSED | 0.1s |
| Photon Sphere | PASSED | 0.1s |
| PPN Parameters | PASSED | 0.2s |
| Quantum Interferometry | PASSED | 0.3s |
| GW Inspiral | PASSED | 0.5s |
| CMB Power Spectrum | PASSED | 1.2s |
| BICEP/Keck Primordial GWs | PASSED | 0.8s |
| **Trajectory Cache Performance** | **PASSED** | **25s → 2.4ms** |

### Cache Management

```python
from physics_agent.cache import TrajectoryCache

# Create cache manager
cache = TrajectoryCache()

# Clear all caches
cache.clear_cache(confirm=True)

# Get cache info
info = cache.get_cache_info(theory_dir)
print(f"Cache size: {info['cache_size_bytes'] / 1e6:.1f} MB")

# Clear old cache files
cache.clear_old_cache()
```

### Benefits

1. **Parameter Sweeps**: 100x-1000x faster parameter exploration
2. **Validator Runs**: Multiple validators can reuse cached trajectories
3. **Interactive Analysis**: Load million-step trajectories in milliseconds
4. **Development**: Rapid iteration during theory development

### Architecture

The caching system integrates seamlessly with the `TheoryEngine`:

```
TheoryEngine.run_trajectory()
    ↓
Check Cache (SHA256 hash)
    ↓
Hit? → Load (2-10ms)
Miss? → Compute → Save → Return
```

### Performance Report

See `CACHE_PERFORMANCE_REPORT.md` for detailed benchmarks and analysis.

## Validation Suite

The test suite validates:

1. **Theoretical Predictions**
   - Circular orbit periods
   - Mercury perihelion precession
   - Light deflection angles
   - Photon sphere radii

2. **Observational Data**
   - PPN parameters (Cassini, LLR)
   - Gravitational waves (LIGO/Virgo)
   - Quantum interferometry (COW experiment)
   - CMB power spectrum (Planck 2018)

3. **Numerical Accuracy**
   - Trajectory convergence
   - Energy conservation
   - Cache consistency

## Future Enhancements

1. **Distributed Caching**: Share cache across compute nodes
2. **Compression**: Further reduce cache size
3. **Cloud Storage**: S3/GCS backend support
4. **Incremental Caching**: Cache partial trajectories
5. **Smart Eviction**: LRU policy for cache management