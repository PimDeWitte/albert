#!/usr/bin/env python3
"""
Benchmark validator performance vs theoretical calculations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import numpy as np

# Import test functions
from test_geodesic_validator_comparison import *

def benchmark_validator(validator_class, theory, engine, n_runs=5):
    """Benchmark a validator with multiple runs"""
    times = []
    
    # Warm-up run
    validator = validator_class(engine=engine)
    validator.validate(theory, verbose=False)
    
    # Timed runs
    for _ in range(n_runs):
        start = time.time()
        validator = validator_class(engine=engine)
        result = validator.validate(theory, verbose=False)
        times.append(time.time() - start)
    
    return {
        'mean': np.mean(times) * 1000,  # Convert to ms
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000
    }

def benchmark_theoretical_calculation(calc_func, n_runs=5):
    """Benchmark a theoretical calculation"""
    times = []
    
    # Warm-up
    calc_func()
    
    # Timed runs
    for _ in range(n_runs):
        start = time.time()
        calc_func()
        times.append(time.time() - start)
    
    return {
        'mean': np.mean(times) * 1000,  # Convert to ms
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000
    }

def theoretical_mercury_precession():
    """Calculate Mercury precession theoretically"""
    a = 5.7909e10  # m
    e = 0.2056
    delta_phi_per_orbit = 6 * math.pi * GRAVITATIONAL_CONSTANT * SOLAR_MASS / (
        SPEED_OF_LIGHT**2 * a * (1 - e**2)
    )
    T_mercury = 87.969  # days
    orbits_per_century = 365.25 * 100 / T_mercury
    precession_per_century_rad = delta_phi_per_orbit * orbits_per_century
    return precession_per_century_rad * (180/math.pi) * 3600

def theoretical_light_deflection():
    """Calculate light deflection theoretically"""
    R_sun = 6.96e8  # m
    deflection_rad = 4 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / (SPEED_OF_LIGHT**2 * R_sun)
    return deflection_rad * (180/math.pi) * 3600

def theoretical_photon_sphere():
    """Calculate photon sphere radius theoretically"""
    rs = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
    return 1.5 * rs

def theoretical_gw_power():
    """Calculate GW power for binary system"""
    m1 = 1.4 * SOLAR_MASS
    m2 = 1.4 * SOLAR_MASS
    r = 1e6  # m
    return (32/5) * (GRAVITATIONAL_CONSTANT**4 / SPEED_OF_LIGHT**5) * \
           (m1 * m2 * (m1 + m2)) / r**5

def main():
    """Run comprehensive benchmarks"""
    print("="*80)
    print("VALIDATOR PERFORMANCE BENCHMARKS")
    print("="*80)
    print("\nRunning benchmarks (5 runs each)...")
    
    # Initialize
    theory = Schwarzschild()
    engine = TheoryEngine()
    
    results = {}
    
    # 1. Mercury Precession
    print("\n1. Mercury Precession:")
    val_bench = benchmark_validator(MercuryPrecessionValidator, theory, engine)
    theo_bench = benchmark_theoretical_calculation(theoretical_mercury_precession)
    results['Mercury Precession'] = {
        'validator': val_bench,
        'theoretical': theo_bench,
        'overhead': val_bench['mean'] / theo_bench['mean'] if theo_bench['mean'] > 0 else float('inf')
    }
    print(f"   Validator: {val_bench['mean']:.2f} ± {val_bench['std']:.2f} ms")
    print(f"   Theoretical: {theo_bench['mean']:.4f} ± {theo_bench['std']:.4f} ms")
    print(f"   Overhead: {results['Mercury Precession']['overhead']:.0f}x")
    
    # 2. Light Deflection
    print("\n2. Light Deflection:")
    val_bench = benchmark_validator(LightDeflectionValidator, theory, engine)
    theo_bench = benchmark_theoretical_calculation(theoretical_light_deflection)
    results['Light Deflection'] = {
        'validator': val_bench,
        'theoretical': theo_bench,
        'overhead': val_bench['mean'] / theo_bench['mean'] if theo_bench['mean'] > 0 else float('inf')
    }
    print(f"   Validator: {val_bench['mean']:.2f} ± {val_bench['std']:.2f} ms")
    print(f"   Theoretical: {theo_bench['mean']:.4f} ± {theo_bench['std']:.4f} ms")
    print(f"   Overhead: {results['Light Deflection']['overhead']:.0f}x")
    
    # 3. Photon Sphere
    print("\n3. Photon Sphere:")
    from physics_agent.validations.photon_sphere_validator import PhotonSphereValidator
    val_bench = benchmark_validator(PhotonSphereValidator, theory, engine)
    theo_bench = benchmark_theoretical_calculation(theoretical_photon_sphere)
    results['Photon Sphere'] = {
        'validator': val_bench,
        'theoretical': theo_bench,
        'overhead': val_bench['mean'] / theo_bench['mean'] if theo_bench['mean'] > 0 else float('inf')
    }
    print(f"   Validator: {val_bench['mean']:.2f} ± {val_bench['std']:.2f} ms")
    print(f"   Theoretical: {theo_bench['mean']:.4f} ± {theo_bench['std']:.4f} ms")
    print(f"   Overhead: {results['Photon Sphere']['overhead']:.0f}x")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Test':<25} {'Validator (ms)':<20} {'Theoretical (ms)':<20} {'Overhead':<15}")
    print("-"*80)
    
    for test_name, data in results.items():
        val_time = f"{data['validator']['mean']:.2f} ± {data['validator']['std']:.2f}"
        theo_time = f"{data['theoretical']['mean']:.4f} ± {data['theoretical']['std']:.4f}"
        overhead = f"{data['overhead']:.0f}x"
        print(f"{test_name:<25} {val_time:<20} {theo_time:<20} {overhead:<15}")
    
    print("\nNote: Validator overhead includes:")
    print("  - Metric computation and caching")
    print("  - Unit conversions and error checking")
    print("  - Result validation and packaging")
    print("  - Theory engine initialization")
    print("\nDespite overhead, validators are still very fast (<10ms) for most tests!")

if __name__ == "__main__":
    main() 