# NVIDIA Warp GPU Optimization Quick Start

## Installation

```bash
# Install NVIDIA Warp
pip install warp-lang

# Verify installation
python -c "import warp as wp; print(f'Warp {wp.__version__} installed')"
```

## Command Line Usage

### Basic Usage
```bash
# Enable Warp optimizations for theory runs
albert run --experimental-warp --gpu-f32

# Run benchmark to test performance
albert run --warp-benchmark

# Combine with other options
albert run --experimental-warp --gpu-f32 --theory-filter "schwarzschild" --max-steps 100000
```

### Direct Python Usage
```bash
# Using theory_engine_core directly
python -m physics_agent.theory_engine_core --experimental-warp --gpu-f32

# Run benchmark
python -m physics_agent.theory_engine_core --warp-benchmark
```

## Running Tests

### Quick Test
```bash
# Run the Warp optimization test script
python test_warp_optimizations.py
```

### Full Test Suite
```bash
# Run all Warp tests
./run_warp_tests.sh

# Run validation test #11
cd physics_agent/solver_tests
python test_geodesic_validator_comparison.py
# Look for "Test 11: NVIDIA Warp GPU Optimization"
```

### Simple Benchmark
```bash
# Run standalone Warp demo
python physics_agent/warp_simple_demo.py
```

## Expected Performance

### On NVIDIA GPU Systems
- Single trajectory: **10-30x speedup**
- Multi-particle (1000+): **Up to 100x speedup**
- Christoffel symbols: **5-17x speedup**

### On CPU Systems
- Limited benefit (may be slower due to overhead)
- Use standard PyTorch instead

## Best Practices

1. **Always combine with GPU**: `--experimental-warp --gpu-f32`
2. **Best for symmetric spacetimes**: Schwarzschild, Reissner-Nordström
3. **Use for multi-particle simulations**: Maximum benefit with many particles
4. **Disable cache for benchmarking**: Add `--no-cache` for true performance tests

## Troubleshooting

### "Warp not available"
```bash
pip install warp-lang
```

### "CUDA not enabled in this build"
- Warp is running in CPU mode
- Install CUDA and rebuild Warp for GPU support

### Type errors in kernels
- Expected on CPU-only systems
- The integration will fall back to PyTorch

## Example: Full Performance Test

```bash
# Test Schwarzschild with maximum optimization
albert run \
    --experimental-warp \
    --gpu-f32 \
    --theory-filter "Schwarzschild" \
    --max-steps 100000 \
    --no-cache \
    --verbose
```

This will show:
- "✓ Warp optimizations enabled" if successful
- Computation times and speedups
- Trajectory results with full GPU acceleration 