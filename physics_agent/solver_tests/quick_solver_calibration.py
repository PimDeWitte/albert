#!/usr/bin/env python3
"""
Quick Solver Calibration

Lightweight validation suite that runs before each physics agent run to ensure:
1. Geodesic integrators are working correctly
2. Validators are properly configured
3. Environment is set up correctly

This skips computationally expensive tests for fast execution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import time
from typing import Dict, Tuple, Optional
import traceback
from datetime import datetime

# Import essential components
from physics_agent.geodesic_integrator import GeodesicRK4Solver
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT
from physics_agent.validations.validator_registry import validator_registry
from physics_agent.solver_tests.device_precision_benchmarker import DevicePrecisionBenchmarker


class SolverCalibration:
    """Quick calibration tests for the solver system."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        if self.verbose:
            elapsed = time.time() - self.start_time
            print(f"[{elapsed:6.2f}s] [{level:^7}] {message}")
    
    def run_calibration(self, include_device_benchmark: bool = False) -> Tuple[bool, Dict[str, any]]:
        """
        Run all calibration tests.
        
        Args:
            include_device_benchmark: If True, run device precision benchmarks (slower)
        
        Returns:
            Tuple of (success, results_dict)
        """
        self.log("Starting solver calibration...")
        
        tests = [
            ("Environment Check", self.test_environment),
            ("Geodesic Solver", self.test_geodesic_solver),
            ("Theory Engine", self.test_theory_engine),
            ("Validator Registry", self.test_validator_registry),
            ("Basic Validation", self.test_basic_validation)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            self.log(f"Running {test_name}...")
            try:
                result = test_func()
                self.results[test_name] = {
                    'passed': result,
                    'error': None
                }
                if result:
                    self.log(f"✓ {test_name} passed", "SUCCESS")
                else:
                    self.log(f"✗ {test_name} failed", "ERROR")
                    all_passed = False
            except Exception as e:
                self.log(f"✗ {test_name} crashed: {str(e)}", "ERROR")
                self.results[test_name] = {
                    'passed': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                all_passed = False
        
        # Run device precision benchmarks if requested
        if include_device_benchmark:
            self.log("Running device precision benchmarks...")
            try:
                benchmarker = DevicePrecisionBenchmarker(verbose=False)
                # Use very small test steps for quick calibration
                benchmark_results = benchmarker.run_benchmarks(
                    test_steps={'trajectory': 50, 'conservation': 25, 'extreme': 10}
                )
                
                self.results['device_benchmarks'] = {
                    'passed': True,
                    'error': None,
                    'results': benchmark_results
                }
                
                # Display recommendations
                recommendations = benchmark_results.get('recommendations', {})
                summary = recommendations.get('summary', {})
                
                if summary.get('gpu_recommended'):
                    self.log("💡 GPU RECOMMENDED: " + summary.get('reason', ''), "INFO")
                else:
                    self.log("💡 CPU RECOMMENDED: " + summary.get('reason', ''), "INFO")
                
                # Show per-test recommendations
                detailed = recommendations.get('detailed', {})
                if detailed:
                    self.log("\nDevice recommendations by test:", "INFO")
                    for test_name, details in detailed.items():
                        self.log(f"  {test_name}: {details['best_option']} "
                               f"({details['speedup']:.1f}x speedup)", "INFO")
                
                # Save recommendations (silently)
                from physics_agent.solver_tests.device_recommendation_manager import device_recommendation_manager
                device_recommendation_manager.update_from_benchmark(benchmark_results)
                
            except Exception as e:
                self.log(f"Device benchmark failed: {str(e)}", "WARNING")
                self.results['device_benchmarks'] = {
                    'passed': False,
                    'error': str(e)
                }
        
        # Summary
        total_time = time.time() - self.start_time
        self.log(f"Calibration completed in {total_time:.2f}s", "INFO")
        
        if all_passed:
            self.log("✓ All calibration tests passed!", "SUCCESS")
        else:
            failed_tests = [name for name, result in self.results.items() 
                          if not result.get('passed', False)]
            self.log(f"✗ Failed tests: {', '.join(failed_tests)}", "ERROR")
        
        # Create calibration certificate
        results_dict = {
            'results': self.results,
            'duration': total_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate certificate
        from physics_agent.solver_tests.calibration_certificate import create_calibration_certificate
        certificate = create_calibration_certificate(results_dict)
        
        self.log(f"\nCalibration Certificate ID: {certificate['certificate_id']}", "INFO")
        self.log(f"Certificate Status: {certificate['status']}", "INFO")
        
        return all_passed, self.results
    
    def test_environment(self) -> bool:
        """Test that the environment is properly configured."""
        # Check PyTorch
        if not torch.cuda.is_available():
            self.log("CUDA not available - using CPU", "WARNING")
        
        # Check tensor operations
        test_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = torch.sum(test_tensor).item()
        if abs(result - 6.0) > 1e-10:
            return False
        
        # Check imports
        required_modules = [
            'numpy', 'scipy', 'matplotlib', 'sympy'
        ]
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                self.log(f"Missing required module: {module}", "ERROR")
                return False
        
        return True
    
    def test_geodesic_solver(self) -> bool:
        """Test basic geodesic integration."""
        theory = Schwarzschild()
        M_sun = torch.tensor(SOLAR_MASS, dtype=torch.float64)
        
        try:
            solver = GeodesicRK4Solver(
                theory, 
                M_phys=M_sun, 
                c=SPEED_OF_LIGHT, 
                G=GRAVITATIONAL_CONSTANT
            )
        except Exception as e:
            self.log(f"Failed to create solver: {e}", "ERROR")
            return False
        
        # Test circular orbit parameters at a safe radius
        rs_phys = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
        r_orbit = 10 * rs_phys  # 10 Schwarzschild radii
        r_geom = solver.to_geometric_length(torch.tensor(r_orbit))
        
        try:
            E_geom, L_geom = solver.compute_circular_orbit_params(r_geom)
            if torch.isnan(E_geom) or torch.isnan(L_geom):
                self.log("NaN in circular orbit parameters", "ERROR")
                return False
        except Exception as e:
            self.log(f"Failed to compute orbit parameters: {e}", "ERROR")
            return False
        
        # Test a single integration step
        solver.E = E_geom.item()
        solver.Lz = L_geom.item()
        
        y = torch.tensor([0.0, r_geom.item(), 0.0, 0.0], dtype=torch.float64)
        h = torch.tensor(0.01, dtype=torch.float64)
        
        try:
            y_new = solver.rk4_step(y, h)
            if y_new is None or torch.any(torch.isnan(y_new)):
                self.log("Integration step failed", "ERROR")
                return False
        except Exception as e:
            self.log(f"RK4 step failed: {e}", "ERROR")
            return False
        
        return True
    
    def test_theory_engine(self) -> bool:
        """Test theory engine initialization and basic operations."""
        try:
            engine = TheoryEngine(verbose=False)
        except Exception as e:
            self.log(f"Failed to create TheoryEngine: {e}", "ERROR")
            return False
        
        # Test trajectory computation with minimal steps
        theory = Schwarzschild()
        r0 = torch.tensor(6e7, dtype=torch.float64)  # Safe starting radius
        n_steps = 10  # Very few steps for speed
        dtau = torch.tensor(0.01, dtype=torch.float64)
        
        try:
            hist, tag, kicks = engine.run_trajectory(
                theory, r0, n_steps, dtau,
                no_cache=True,  # Skip cache for calibration
                verbose=False
            )
            if hist is None:
                self.log("Trajectory computation returned None", "ERROR")
                return False
        except Exception as e:
            self.log(f"Trajectory computation failed: {e}", "ERROR")
            return False
        
        return True
    
    def test_validator_registry(self) -> bool:
        """Test that validator registry is properly configured."""
        # Check that we have the expected validators
        tested = validator_registry.get_tested_validators()
        if len(tested) < 9:  # We expect at least 9 tested validators
            self.log(f"Only {len(tested)} tested validators found", "WARNING")
        
        # Check registry report generation
        try:
            report = validator_registry.generate_registry_report()
            if not isinstance(report, dict):
                return False
        except Exception as e:
            self.log(f"Registry report generation failed: {e}", "ERROR")
            return False
        
        return True
    
    def test_basic_validation(self) -> bool:
        """Run a single quick validation test."""
        # Import a fast validator
        from physics_agent.validations.metric_properties_validator import MetricPropertiesValidator
        
        theory = Schwarzschild()
        engine = TheoryEngine(verbose=False)
        
        try:
            validator = MetricPropertiesValidator(engine)
            # Create minimal trajectory for validation
            hist = torch.zeros((10, 4), dtype=torch.float64)
            result = validator.validate(theory, hist)
            
            if not isinstance(result, dict):
                self.log("Validator returned invalid result", "ERROR")
                return False
            
            if 'flags' not in result or 'overall' not in result['flags']:
                self.log("Validator result missing required fields", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Validation test failed: {e}", "ERROR")
            return False
        
        return True


def run_quick_calibration(verbose: bool = True, include_device_benchmark: bool = True) -> bool:
    """
    Run quick calibration and return success status.
    
    Args:
        verbose: Whether to print detailed output
        include_device_benchmark: If True, include device precision benchmarks
        
    Returns:
        bool: True if all tests passed
    """
    calibration = SolverCalibration(verbose=verbose)
    success, results = calibration.run_calibration(include_device_benchmark=include_device_benchmark)
    
    # Save calibration results
    if verbose:
        import json
        from datetime import datetime
        
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'cache', 'calibration'
        )
        os.makedirs(cache_dir, exist_ok=True)
        
        result_file = os.path.join(
            cache_dir,
            f'calibration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(result_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'results': results,
                'duration': time.time() - calibration.start_time
            }, f, indent=2)
        
        # Also save as latest
        latest_file = os.path.join(cache_dir, 'calibration_latest.json')
        with open(latest_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'success': success,
                'results': results,
                'duration': time.time() - calibration.start_time
            }, f, indent=2)
    
    return success


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run quick solver calibration")
    parser.add_argument('--no-device-benchmark', action='store_true',
                       help='Skip device precision benchmarking')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    args = parser.parse_args()
    
    # Run calibration
    success = run_quick_calibration(
        verbose=not args.quiet,
        include_device_benchmark=not args.no_device_benchmark
    )
    sys.exit(0 if success else 1) 