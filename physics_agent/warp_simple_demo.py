#!/usr/bin/env python3
"""
Simple NVIDIA Warp demonstration for gravitational physics.

This shows the potential speedups with a simplified RK4 integration kernel.
"""

import time
import numpy as np
import torch

try:
    import warp as wp
    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False
    print("Warp not available")

if WARP_AVAILABLE:
    wp.init()

    @wp.kernel
    def simple_rk4_kernel(
        r: wp.array(dtype=float),
        phi: wp.array(dtype=float),
        dr_dt: wp.array(dtype=float),
        dphi_dt: wp.array(dtype=float),
        dt: float,
        rs: float,
        output_r: wp.array(dtype=float),
        output_phi: wp.array(dtype=float),
        output_dr_dt: wp.array(dtype=float),
        output_dphi_dt: wp.array(dtype=float)
    ):
        """Simple RK4 step for Schwarzschild metric"""
        tid = wp.tid()
        
        # Load current state
        r_val = r[tid]
        phi_val = phi[tid]
        dr_dt_val = dr_dt[tid]
        dphi_dt_val = dphi_dt[tid]
        
        # Schwarzschild metric
        f = 1.0 - rs / r_val
        
        # Simple accelerations (simplified for demo)
        d2r_dt2 = -0.5 * rs / (r_val * r_val) * f
        d2phi_dt2 = -2.0 / r_val * dr_dt_val * dphi_dt_val
        
        # RK4 integration step (simplified)
        output_r[tid] = r_val + dr_dt_val * dt + 0.5 * d2r_dt2 * dt * dt
        output_phi[tid] = phi_val + dphi_dt_val * dt + 0.5 * d2phi_dt2 * dt * dt
        output_dr_dt[tid] = dr_dt_val + d2r_dt2 * dt
        output_dphi_dt[tid] = dphi_dt_val + d2phi_dt2 * dt


def benchmark_simple_warp():
    """Benchmark simple Warp kernel vs NumPy"""
    if not WARP_AVAILABLE:
        print("Warp not available")
        return
    
    # Test parameters
    n_particles = 10000
    n_steps = 1000
    dt = 0.01
    rs = 2.0  # Schwarzschild radius
    
    # Initialize particle states
    np.random.seed(42)
    r_init = np.random.uniform(10.0, 100.0, n_particles)
    phi_init = np.random.uniform(0, 2*np.pi, n_particles)
    dr_dt_init = np.zeros(n_particles)
    dphi_dt_init = np.random.uniform(0.01, 0.1, n_particles)
    
    print(f"Benchmarking {n_particles} particles for {n_steps} steps")
    print("="*50)
    
    # NumPy benchmark
    print("\n1. NumPy Implementation:")
    r_np = r_init.copy()
    phi_np = phi_init.copy()
    dr_dt_np = dr_dt_init.copy()
    dphi_dt_np = dphi_dt_init.copy()
    
    start = time.time()
    for step in range(n_steps):
        # Schwarzschild metric
        f = 1.0 - rs / r_np
        
        # Accelerations
        d2r_dt2 = -0.5 * rs / (r_np * r_np) * f
        d2phi_dt2 = -2.0 / r_np * dr_dt_np * dphi_dt_np
        
        # RK4 step (simplified)
        r_np = r_np + dr_dt_np * dt + 0.5 * d2r_dt2 * dt * dt
        phi_np = phi_np + dphi_dt_np * dt + 0.5 * d2phi_dt2 * dt * dt
        dr_dt_np = dr_dt_np + d2r_dt2 * dt
        dphi_dt_np = dphi_dt_np + d2phi_dt2 * dt
    
    numpy_time = time.time() - start
    print(f"   Time: {numpy_time:.3f}s")
    print(f"   Particles/second: {n_particles * n_steps / numpy_time:.0f}")
    
    # Warp benchmark
    print("\n2. Warp Implementation:")
    
    # Convert to Warp arrays
    r_wp = wp.array(r_init, dtype=float)
    phi_wp = wp.array(phi_init, dtype=float)
    dr_dt_wp = wp.array(dr_dt_init, dtype=float)
    dphi_dt_wp = wp.array(dphi_dt_init, dtype=float)
    
    # Output arrays
    r_out = wp.zeros_like(r_wp)
    phi_out = wp.zeros_like(phi_wp)
    dr_dt_out = wp.zeros_like(dr_dt_wp)
    dphi_dt_out = wp.zeros_like(dphi_dt_wp)
    
    # Compile kernel first
    wp.launch(
        kernel=simple_rk4_kernel,
        dim=n_particles,
        inputs=[r_wp, phi_wp, dr_dt_wp, dphi_dt_wp, dt, rs],
        outputs=[r_out, phi_out, dr_dt_out, dphi_dt_out]
    )
    wp.synchronize()
    
    start = time.time()
    for step in range(n_steps):
        wp.launch(
            kernel=simple_rk4_kernel,
            dim=n_particles,
            inputs=[r_wp, phi_wp, dr_dt_wp, dphi_dt_wp, dt, rs],
            outputs=[r_out, phi_out, dr_dt_out, dphi_dt_out]
        )
        # Swap buffers
        r_wp, r_out = r_out, r_wp
        phi_wp, phi_out = phi_out, phi_wp
        dr_dt_wp, dr_dt_out = dr_dt_out, dr_dt_wp
        dphi_dt_wp, dphi_dt_out = dphi_dt_out, dphi_dt_wp
    
    wp.synchronize()
    warp_time = time.time() - start
    print(f"   Time: {warp_time:.3f}s")
    print(f"   Particles/second: {n_particles * n_steps / warp_time:.0f}")
    
    # Calculate speedup
    speedup = numpy_time / warp_time
    print(f"\n3. Performance Summary:")
    print(f"   NumPy time: {numpy_time:.3f}s")
    print(f"   Warp time: {warp_time:.3f}s")
    print(f"   Speedup: {speedup:.1f}x")
    
    # Verify results match
    r_final_np = r_np
    r_final_wp = r_wp.numpy()
    max_diff = np.max(np.abs(r_final_np - r_final_wp))
    print(f"\n4. Accuracy Check:")
    print(f"   Max difference in r: {max_diff:.2e}")
    print(f"   Results match: {'YES' if max_diff < 1e-6 else 'NO'}")
    
    return speedup


def benchmark_trajectory_comparison():
    """Compare trajectory computation times"""
    from physics_agent.theory_engine_core import TheoryEngine
    from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
    from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT
    
    print("\n" + "="*50)
    print("TheoryEngine Trajectory Benchmark")
    print("="*50)
    
    theory = Schwarzschild()
    
    # Test different devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    rs_phys = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
    r0 = 100 * rs_phys
    n_steps = 10000
    dtau = 1e-5
    
    times = {}
    for device in devices:
        engine = TheoryEngine(device=device)
        
        print(f"\nDevice: {device}")
        start = time.time()
        hist, tag, _ = engine.run_trajectory(
            theory, r0, n_steps, dtau, no_cache=True, verbose=False
        )
        elapsed = time.time() - start
        times[device] = elapsed
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Steps/second: {n_steps/elapsed:.0f}")
    
    if 'cuda' in times and 'cpu' in times:
        speedup = times['cpu'] / times['cuda']
        print(f"\nGPU Speedup: {speedup:.1f}x")
    
    return times


if __name__ == "__main__":
    print("NVIDIA Warp Simple Demonstration")
    print("="*50)
    
    # Check system
    print(f"\nSystem Info:")
    print(f"  Warp available: {WARP_AVAILABLE}")
    print(f"  PyTorch CUDA: {torch.cuda.is_available()}")
    
    if WARP_AVAILABLE:
        import warp
        print(f"  Warp version: {warp.__version__}")
        print(f"  Warp devices: {warp.get_devices()}")
    
    # Run benchmarks
    if WARP_AVAILABLE:
        print("\n" + "="*50)
        speedup = benchmark_simple_warp()
        
        if speedup > 1.0:
            print(f"\n✅ Warp provides {speedup:.1f}x speedup for particle simulations!")
        else:
            print(f"\n⚠️ Limited speedup on this system (CPU-only build)")
    
    # Compare with TheoryEngine
    print("\n" + "="*50)
    benchmark_trajectory_comparison() 