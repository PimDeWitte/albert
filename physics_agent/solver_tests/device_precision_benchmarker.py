#!/usr/bin/env python3
"""
Device Precision Benchmarker

Compares precision and performance between CPU and GPU for physics simulations.
Provides recommendations on optimal device usage for different scenarios.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physics_agent.geodesic_integrator import ConservedQuantityGeodesicSolver
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""
    device: str
    dtype: str
    test_name: str
    execution_time: float
    precision_error: Optional[float]
    memory_usage: Optional[float]
    success: bool
    error_message: Optional[str] = None
    trajectory_length: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            'device': self.device,
            'dtype': self.dtype,
            'test_name': self.test_name,
            'execution_time': self.execution_time,
            'precision_error': self.precision_error,
            'memory_usage': self.memory_usage,
            'success': self.success,
            'error_message': self.error_message,
            'trajectory_length': self.trajectory_length
        }


class DevicePrecisionBenchmarker:
    """Benchmark precision and performance across different devices."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
        self.reference_results: Dict[str, torch.Tensor] = {}
        
    def log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def get_available_devices(self) -> List[Tuple[str, str]]:
        """Get list of available (device, dtype) combinations."""
        devices = []
        
        # CPU is always available
        devices.append(('cpu', 'float64'))  # High precision reference
        devices.append(('cpu', 'float32'))  # Lower precision
        
        # Check for CUDA GPUs
        if torch.cuda.is_available():
            devices.append(('cuda', 'float32'))
            devices.append(('cuda', 'float64'))
        
        # Check for Apple Silicon
        if torch.backends.mps.is_available():
            devices.append(('mps', 'float32'))
            # Note: MPS doesn't support float64
        
        return devices
    
    def benchmark_trajectory_computation(
        self, 
        device: str, 
        dtype_str: str,
        steps: int = 1000
    ) -> BenchmarkResult:
        """Benchmark a single trajectory computation."""
        test_name = "trajectory_computation"
        dtype = torch.float64 if dtype_str == 'float64' else torch.float32
        
        try:
            # Create engine
            engine = TheoryEngine(device=device, dtype=dtype, verbose=False)
            theory = Schwarzschild()
            
            # Set up parameters
            rs_phys = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
            r0 = torch.tensor(10 * rs_phys, dtype=dtype, device=device)
            dtau = torch.tensor(0.1, dtype=dtype, device=device)
            
            # Warm up (important for GPU)
            if device != 'cpu':
                _ = engine.run_trajectory(
                    theory, r0, 10, dtau, no_cache=True, verbose=False
                )
            
            # Time the computation
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            
            hist, _, _ = engine.run_trajectory(
                theory, r0, steps, dtau,
                no_cache=True,
                verbose=False
            )
            
            torch.cuda.synchronize() if device == 'cuda' else None
            execution_time = time.time() - start_time
            
            # Store reference result for precision comparison
            ref_key = f"{test_name}_reference"
            if ref_key not in self.reference_results:
                # First run becomes the reference (should be cpu/float64)
                self.reference_results[ref_key] = hist.cpu().double()
                precision_error = 0.0
            else:
                # Compare with reference
                ref = self.reference_results[ref_key]
                hist_cpu = hist.cpu().double()
                
                # Ensure same shape
                min_len = min(len(hist_cpu), len(ref))
                hist_cpu = hist_cpu[:min_len]
                ref = ref[:min_len]
                
                # Calculate relative error
                precision_error = torch.abs(hist_cpu - ref).max().item() / torch.abs(ref).max().item()
            
            return BenchmarkResult(
                device=device,
                dtype=dtype_str,
                test_name=test_name,
                execution_time=execution_time,
                precision_error=precision_error,
                memory_usage=None,
                success=True,
                trajectory_length=len(hist) if hist is not None else 0
            )
            
        except Exception as e:
            return BenchmarkResult(
                device=device,
                dtype=dtype_str,
                test_name=test_name,
                execution_time=float('inf'),
                precision_error=None,
                memory_usage=None,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_conservation_test(
        self,
        device: str,
        dtype_str: str,
        steps: int = 500
    ) -> BenchmarkResult:
        """Benchmark conservation law calculations."""
        test_name = "conservation_laws"
        dtype = torch.float64 if dtype_str == 'float64' else torch.float32
        
        try:
            engine = TheoryEngine(device=device, dtype=dtype, verbose=False)
            theory = Schwarzschild()
            
            rs_phys = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
            r0 = torch.tensor(15 * rs_phys, dtype=dtype, device=device)
            dtau = torch.tensor(0.1, dtype=dtype, device=device)
            
            # Warm up for GPU
            if device != 'cpu':
                _ = engine.run_trajectory(
                    theory, r0, 10, dtau, no_cache=True, verbose=False
                )
            
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            
            # Run trajectory
            hist, _, _ = engine.run_trajectory(
                theory, r0, steps, dtau,
                no_cache=True,
                verbose=False
            )
            
            # Calculate conservation metrics
            if hist is not None and len(hist) > 1:
                r = hist[:, 1]
                phi = hist[:, 2]
                
                # Angular momentum
                dphi_dtau = torch.diff(phi) / 0.1
                dphi_dtau = torch.cat([dphi_dtau[:1], dphi_dtau])
                L = r**2 * dphi_dtau
                
                # Conservation error
                L_mean = L.mean().abs()
                if L_mean > 1e-10:
                    L_variation = (L.max() - L.min()) / L_mean
                else:
                    L_variation = torch.tensor(0.0)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            execution_time = time.time() - start_time
            
            # Store conservation error as precision metric
            precision_error = L_variation.item() if hist is not None else float('inf')
            
            return BenchmarkResult(
                device=device,
                dtype=dtype_str,
                test_name=test_name,
                execution_time=execution_time,
                precision_error=precision_error,
                memory_usage=None,
                success=True,
                trajectory_length=len(hist) if hist is not None else 0
            )
            
        except Exception as e:
            return BenchmarkResult(
                device=device,
                dtype=dtype_str,
                test_name=test_name,
                execution_time=float('inf'),
                precision_error=None,
                memory_usage=None,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_extreme_conditions(
        self,
        device: str,
        dtype_str: str
    ) -> BenchmarkResult:
        """Benchmark numerical stability at extreme conditions."""
        test_name = "extreme_conditions"
        dtype = torch.float64 if dtype_str == 'float64' else torch.float32
        
        try:
            engine = TheoryEngine(device=device, dtype=dtype, verbose=False)
            theory = Schwarzschild()
            
            rs_phys = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
            
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            
            # Test very close to event horizon
            r0_close = torch.tensor(2.1 * rs_phys, dtype=dtype, device=device)
            dtau = torch.tensor(0.01, dtype=dtype, device=device)
            
            hist_close, _, _ = engine.run_trajectory(
                theory, r0_close, 50, dtau,
                no_cache=True,
                verbose=False
            )
            
            # Test far from black hole
            r0_far = torch.tensor(1000 * rs_phys, dtype=dtype, device=device)
            
            hist_far, _, _ = engine.run_trajectory(
                theory, r0_far, 50, dtau,
                no_cache=True,
                verbose=False
            )
            
            torch.cuda.synchronize() if device == 'cuda' else None
            execution_time = time.time() - start_time
            
            # Check stability (both should succeed)
            success = hist_close is not None and hist_far is not None
            
            # Measure precision by checking trajectory smoothness
            if success and len(hist_close) > 2:
                # Second derivative as stability metric
                r_close = hist_close[:, 1]
                d2r = torch.diff(torch.diff(r_close))
                precision_error = torch.abs(d2r).max().item() if len(d2r) > 0 else 0.0
            else:
                precision_error = float('inf')
            
            return BenchmarkResult(
                device=device,
                dtype=dtype_str,
                test_name=test_name,
                execution_time=execution_time,
                precision_error=precision_error,
                memory_usage=None,
                success=success
            )
            
        except Exception as e:
            return BenchmarkResult(
                device=device,
                dtype=dtype_str,
                test_name=test_name,
                execution_time=float('inf'),
                precision_error=None,
                memory_usage=None,
                success=False,
                error_message=str(e)
            )
    
    def run_benchmarks(self, test_steps: Dict[str, int] = None) -> Dict[str, Any]:
        """Run all benchmarks across available devices."""
        if test_steps is None:
            test_steps = {
                'trajectory': 1000,
                'conservation': 500,
                'extreme': 50
            }
        
        self.log("="*60)
        self.log("Device Precision Benchmarking")
        self.log("="*60)
        
        devices = self.get_available_devices()
        self.log(f"Found {len(devices)} device/dtype combinations:")
        for device, dtype in devices:
            self.log(f"  - {device}/{dtype}")
        self.log("")
        
        # Run benchmarks
        for device, dtype in devices:
            self.log(f"\nBenchmarking {device}/{dtype}...")
            
            # Trajectory computation
            self.log("  - Trajectory computation...")
            result = self.benchmark_trajectory_computation(device, dtype, test_steps['trajectory'])
            self.results.append(result)
            if result.success:
                self.log(f"    Time: {result.execution_time:.3f}s, Error: {result.precision_error:.2e}")
            else:
                self.log(f"    Failed: {result.error_message}")
            
            # Conservation laws
            self.log("  - Conservation laws...")
            result = self.benchmark_conservation_test(device, dtype, test_steps['conservation'])
            self.results.append(result)
            if result.success:
                self.log(f"    Time: {result.execution_time:.3f}s, Conservation error: {result.precision_error:.2e}")
            else:
                self.log(f"    Failed: {result.error_message}")
            
            # Extreme conditions
            self.log("  - Extreme conditions...")
            result = self.benchmark_extreme_conditions(device, dtype)
            self.results.append(result)
            if result.success:
                self.log(f"    Time: {result.execution_time:.3f}s, Stability: {result.precision_error:.2e}")
            else:
                self.log(f"    Failed: {result.error_message}")
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        return {
            'results': [r.to_dict() for r in self.results],
            'recommendations': recommendations,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate device recommendations based on benchmark results."""
        recommendations = {
            'summary': {},
            'detailed': {},
            'warnings': []
        }
        
        # Group results by test
        test_groups = {}
        for result in self.results:
            if result.test_name not in test_groups:
                test_groups[result.test_name] = []
            test_groups[result.test_name].append(result)
        
        # Analyze each test
        for test_name, results in test_groups.items():
            # Find reference (cpu/float64)
            reference = next((r for r in results if r.device == 'cpu' and r.dtype == 'float64'), None)
            if not reference or not reference.success:
                recommendations['warnings'].append(f"No valid reference for {test_name}")
                continue
            
            # Compare all results
            test_recs = []
            for result in results:
                if not result.success:
                    continue
                
                # Calculate speedup
                speedup = reference.execution_time / result.execution_time if result.execution_time > 0 else 0
                
                # Precision relative to reference
                precision_ratio = result.precision_error / reference.precision_error if reference.precision_error > 0 else 1.0
                
                rec = {
                    'device': result.device,
                    'dtype': result.dtype,
                    'speedup': speedup,
                    'precision_ratio': precision_ratio,
                    'execution_time': result.execution_time,
                    'precision_error': result.precision_error
                }
                
                # Generate recommendation
                if speedup > 1.5 and precision_ratio < 10:
                    rec['recommendation'] = 'RECOMMENDED'
                    rec['reason'] = f"{speedup:.1f}x faster with acceptable precision"
                elif speedup > 1.0 and precision_ratio < 2:
                    rec['recommendation'] = 'GOOD'
                    rec['reason'] = "Faster with minimal precision loss"
                elif precision_ratio > 100:
                    rec['recommendation'] = 'NOT_RECOMMENDED'
                    rec['reason'] = "Significant precision loss"
                else:
                    rec['recommendation'] = 'ACCEPTABLE'
                    rec['reason'] = "Trade-off between speed and precision"
                
                test_recs.append(rec)
            
            # Find best option
            best = max(test_recs, key=lambda x: x['speedup'] / max(x['precision_ratio'], 1.0))
            recommendations['detailed'][test_name] = {
                'best_option': f"{best['device']}/{best['dtype']}",
                'speedup': best['speedup'],
                'reason': best['reason'],
                'all_options': test_recs
            }
        
        # Overall summary
        if any('cuda' in r['best_option'] for r in recommendations['detailed'].values()):
            recommendations['summary']['gpu_recommended'] = True
            recommendations['summary']['reason'] = "GPU provides significant speedup for most tests"
        else:
            recommendations['summary']['gpu_recommended'] = False
            recommendations['summary']['reason'] = "CPU provides better precision/performance balance"
        
        return recommendations
    
    def save_results(self, filepath: str):
        """Save benchmark results to JSON file."""
        results = self.run_benchmarks()
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        self.log(f"\nResults saved to: {filepath}")
    
    def print_summary(self):
        """Print a summary of recommendations."""
        if not self.results:
            self.log("No benchmark results available. Run benchmarks first.")
            return
        
        recommendations = self.generate_recommendations()
        
        self.log("\n" + "="*60)
        self.log("RECOMMENDATIONS")
        self.log("="*60)
        
        # Overall recommendation
        summary = recommendations['summary']
        self.log(f"\nOverall: {'USE GPU' if summary.get('gpu_recommended', False) else 'USE CPU'}")
        self.log(f"Reason: {summary.get('reason', 'No recommendation available')}")
        
        # Per-test recommendations
        self.log("\nPer-test recommendations:")
        for test_name, details in recommendations['detailed'].items():
            self.log(f"\n{test_name}:")
            self.log(f"  Best: {details['best_option']} ({details['speedup']:.1f}x speedup)")
            self.log(f"  {details['reason']}")
        
        # Warnings
        if recommendations.get('warnings'):
            self.log("\nWarnings:")
            for warning in recommendations['warnings']:
                self.log(f"  - {warning}")


def run_device_precision_benchmark(verbose: bool = True, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run device precision benchmarks and return results.
    
    Args:
        verbose: Whether to print detailed output
        save_path: Optional path to save results JSON
        
    Returns:
        Dictionary with benchmark results and recommendations
    """
    benchmarker = DevicePrecisionBenchmarker(verbose=verbose)
    results = benchmarker.run_benchmarks()
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    if verbose:
        benchmarker.print_summary()
    
    return results


if __name__ == "__main__":
    # Run benchmarks when called directly
    results = run_device_precision_benchmark(verbose=True)
    
    # Save to cache directory
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'cache', 'benchmarks'
    )
    os.makedirs(cache_dir, exist_ok=True)
    
    save_path = os.path.join(cache_dir, f'device_benchmark_{time.strftime("%Y%m%d_%H%M%S")}.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {save_path}")
    
    # Update device recommendations
    from physics_agent.solver_tests.device_recommendation_manager import update_device_recommendations
    update_device_recommendations(results) 