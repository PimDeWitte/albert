#!/usr/bin/env python3
"""
Device Recommendation Manager

Manages device recommendations based on benchmark results.
Saves preferred configurations and applies them automatically.
"""

import json
import os
from typing import Dict, Optional, Tuple
from pathlib import Path
import torch


class DeviceRecommendationManager:
    """Manages device recommendations for physics simulations."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the recommendation manager."""
        if config_dir is None:
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'cache', 'config'
            )
        self.config_dir = config_dir
        os.makedirs(self.config_dir, exist_ok=True)
        
        self.config_file = os.path.join(self.config_dir, 'device_recommendations.json')
        self.recommendations = self.load_recommendations()
    
    def load_recommendations(self) -> Dict[str, any]:
        """Load saved recommendations from disk."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_recommendations(self):
        """Save recommendations to disk."""
        with open(self.config_file, 'w') as f:
            json.dump(self.recommendations, f, indent=2)
    
    def update_from_benchmark(self, benchmark_results: Dict[str, any]):
        """Update recommendations based on new benchmark results."""
        recommendations = benchmark_results.get('recommendations', {})
        
        # Update overall recommendation
        summary = recommendations.get('summary', {})
        self.recommendations['overall'] = {
            'gpu_recommended': summary.get('gpu_recommended', False),
            'reason': summary.get('reason', 'No recommendation available'),
            'timestamp': benchmark_results.get('timestamp', 'Unknown')
        }
        
        # Update per-test recommendations
        detailed = recommendations.get('detailed', {})
        self.recommendations['per_test'] = {}
        
        for test_name, details in detailed.items():
            # Extract device and dtype from best_option
            best_option = details.get('best_option', 'cpu/float64')
            device, dtype = best_option.split('/')
            
            self.recommendations['per_test'][test_name] = {
                'device': device,
                'dtype': dtype,
                'speedup': details.get('speedup', 1.0),
                'reason': details.get('reason', '')
            }
        
        # Save updated recommendations
        self.save_recommendations()
    
    def get_optimal_device(self, test_type: Optional[str] = None) -> Tuple[str, torch.dtype]:
        """
        Get optimal device configuration for a given test type.
        
        Args:
            test_type: Type of test (e.g., 'trajectory_computation')
                      If None, returns overall recommendation
        
        Returns:
            Tuple of (device, dtype)
        """
        # Default configuration
        default_device = 'cpu'
        default_dtype = torch.float64
        
        # Check if we have recommendations
        if not self.recommendations:
            return default_device, default_dtype
        
        # Get specific test recommendation if available
        if test_type and 'per_test' in self.recommendations:
            test_rec = self.recommendations['per_test'].get(test_type, {})
            if test_rec:
                device = test_rec.get('device', default_device)
                dtype_str = test_rec.get('dtype', 'float64')
                dtype = torch.float64 if dtype_str == 'float64' else torch.float32
                return device, dtype
        
        # Fall back to overall recommendation
        overall = self.recommendations.get('overall', {})
        if overall.get('gpu_recommended'):
            # Check available GPUs
            if torch.cuda.is_available():
                return 'cuda', torch.float32
            elif torch.backends.mps.is_available():
                return 'mps', torch.float32
        
        return default_device, default_dtype
    
    def get_recommendation_summary(self) -> str:
        """Get a human-readable summary of current recommendations."""
        if not self.recommendations:
            return "No device recommendations available. Run benchmarks to generate recommendations."
        
        lines = ["Device Recommendations Summary"]
        lines.append("=" * 40)
        
        # Overall recommendation
        overall = self.recommendations.get('overall', {})
        if overall:
            lines.append(f"\nOverall: {'USE GPU' if overall.get('gpu_recommended') else 'USE CPU'}")
            lines.append(f"Reason: {overall.get('reason', 'Unknown')}")
            lines.append(f"Last updated: {overall.get('timestamp', 'Unknown')}")
        
        # Per-test recommendations
        per_test = self.recommendations.get('per_test', {})
        if per_test:
            lines.append("\nPer-test optimal configurations:")
            for test_name, config in per_test.items():
                lines.append(f"\n{test_name}:")
                lines.append(f"  Device: {config.get('device', 'cpu')}")
                lines.append(f"  Precision: {config.get('dtype', 'float64')}")
                lines.append(f"  Speedup: {config.get('speedup', 1.0):.1f}x")
                if config.get('reason'):
                    lines.append(f"  {config.get('reason', '')}")
        
        return "\n".join(lines)
    
    def apply_optimal_settings(self, engine):
        """
        Apply optimal device settings to a TheoryEngine instance.
        
        Args:
            engine: TheoryEngine instance to configure
        """
        # Get overall optimal device
        device, dtype = self.get_optimal_device()
        
        # Check if device is available
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        elif device == 'mps' and not torch.backends.mps.is_available():
            device = 'cpu'
        
        # Apply settings if different
        if engine.device != device or engine.dtype != dtype:
            print(f"Applying optimal device configuration: {device}/{dtype}")
            engine.device = device
            engine.dtype = dtype
            
            # Move existing tensors if needed
            if hasattr(engine, 'solver') and engine.solver:
                engine.solver.device = device
                engine.solver.dtype = dtype


# Global instance
device_recommendation_manager = DeviceRecommendationManager()


def get_optimal_device_config(test_type: Optional[str] = None) -> Tuple[str, torch.dtype]:
    """
    Get optimal device configuration for a given test type.
    
    Args:
        test_type: Type of test (e.g., 'trajectory_computation')
        
    Returns:
        Tuple of (device, dtype)
    """
    return device_recommendation_manager.get_optimal_device(test_type)


def update_device_recommendations(benchmark_results: Dict[str, any]):
    """Update device recommendations based on benchmark results."""
    device_recommendation_manager.update_from_benchmark(benchmark_results)
    print("\nDevice recommendations updated!")
    print(device_recommendation_manager.get_recommendation_summary())


if __name__ == '__main__':
    # Test the recommendation manager
    manager = DeviceRecommendationManager()
    print(manager.get_recommendation_summary())
    
    # Show optimal device for different test types
    print("\nOptimal configurations:")
    for test_type in ['trajectory_computation', 'conservation_laws', 'extreme_conditions', None]:
        device, dtype = manager.get_optimal_device(test_type)
        print(f"  {test_type or 'Overall'}: {device}/{dtype}") 