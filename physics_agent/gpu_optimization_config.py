"""
GPU Optimization Configuration for Physics Agent

This module provides configuration settings and utilities for GPU acceleration
of geodesic simulations, particularly for computationally intensive metrics
like Kerr and Kerr-Newman.

IMPORTANT: Scientific computations require float64 precision. This module
ensures precision is maintained or clearly warns when it cannot be.
"""

import torch
import os
import warnings
from typing import Optional, Tuple, Dict


def get_optimal_device(require_float64: bool = True) -> str:
    """
    Determine the optimal device for computations.
    
    <reason>chain: Select device that supports required precision (float64 by default)</reason>
    
    Args:
        require_float64: If True, prefer devices that support float64
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        # CUDA supports float64
        return 'cuda'
    elif torch.backends.mps.is_available():
        if not require_float64:
            # MPS only supports float32, warn user
            warnings.warn(
                "MPS (Apple Silicon) does not support float64. "
                "Falling back to CPU to maintain precision.",
                UserWarning
            )
        # For scientific computing requiring float64, use CPU
        return 'cpu'
    else:
        return 'cpu'


def check_device_precision_support(device: str, dtype: torch.dtype) -> bool:
    """
    Check if a device supports the requested precision.
    
    <reason>chain: Verify device can handle required precision before use</reason>
    
    Args:
        device: Device string
        dtype: Requested data type
        
    Returns:
        True if device supports the dtype, False otherwise
    """
    if device == 'cpu':
        # CPU supports all dtypes
        return True
    elif device == 'cuda':
        # CUDA supports float64
        return True
    elif device == 'mps':
        # MPS only supports float32
        return dtype in [torch.float32, torch.float16]
    else:
        # Unknown device, assume it doesn't support float64
        return dtype != torch.float64


def configure_gpu_optimizations(device: str = None, dtype: torch.dtype = torch.float64) -> dict:
    """
    Configure GPU-specific optimizations.
    
    <reason>chain: Enable performance optimizations while maintaining required precision</reason>
    
    Args:
        device: Target device (if None, auto-detect)
        dtype: Required data type (default float64 for precision)
    
    Returns:
        Dictionary of optimization settings
    """
    if device is None:
        device = get_optimal_device(require_float64=(dtype == torch.float64))
    
    # Check if device supports requested precision
    if not check_device_precision_support(device, dtype):
        warnings.warn(
            f"Device '{device}' does not support {dtype}. "
            f"Falling back to CPU to maintain precision.",
            UserWarning
        )
        device = 'cpu'
    
    config = {
        'enable_torch_compile': torch.__version__ >= '2.0.0',
        'batch_size': 1,  # Default to sequential processing
        'cache_christoffel': True,
        'cache_size_limit': 10000,
        'enable_mixed_precision': False,  # Always False for scientific computing
        'enable_cudnn_benchmark': False,
        'disable_gradient_computation': False,  # Don't disable gradients - needed for Christoffel symbols
        'device': device,
        'dtype': dtype,
    }
    
    if device == 'cuda' and dtype == torch.float64:
        # CUDA-specific optimizations for float64
        config['batch_size'] = 16  # Reduced from 32 for float64 memory usage
        config['enable_cudnn_benchmark'] = True
        
        # Check available memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            if total_memory > 16 * 1024**3:  # > 16GB for float64
                config['batch_size'] = 32
                config['cache_size_limit'] = 50000
            elif total_memory < 8 * 1024**3:  # < 8GB
                config['batch_size'] = 8
                config['cache_size_limit'] = 5000
                
    elif device == 'mps':
        # Should not reach here if dtype is float64
        warnings.warn(
            "MPS device selected but float64 not supported. "
            "Precision loss may occur!",
            UserWarning
        )
        config['batch_size'] = 16
        
    return config


def optimize_tensor_operations(device: str = None, dtype: torch.dtype = torch.float64) -> None:
    """
    Apply global PyTorch optimizations for tensor operations.
    
    <reason>chain: Set PyTorch flags for optimal performance while maintaining precision</reason>
    
    Args:
        device: Target device ('cuda', 'mps', or 'cpu')
        dtype: Required precision
    """
    if device is None:
        device = get_optimal_device(require_float64=(dtype == torch.float64))
    
    # Verify device supports required precision
    if not check_device_precision_support(device, dtype):
        device = 'cpu'
    
    # Set number of threads for CPU operations
    if device == 'cpu':
        # Use all available cores
        torch.set_num_threads(os.cpu_count())
    
    # CUDA-specific settings
    if device == 'cuda' and torch.cuda.is_available():
        if dtype == torch.float64:
            # Disable TF32 for float64 to maintain precision
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            # Enable TF32 for float32 (maintains sufficient precision)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        
        # Reduce memory fragmentation
        torch.cuda.empty_cache()
        
    # Optimize memory allocator
    if hasattr(torch.cuda, 'set_allocator_settings'):
        torch.cuda.set_allocator_settings('max_split_size_mb:512')


def create_optimized_solver_kwargs(device: str = None, dtype: torch.dtype = torch.float64) -> dict:
    """
    Create keyword arguments for optimized geodesic solvers.
    
    <reason>chain: Consolidate optimization settings ensuring precision is maintained</reason>
    
    Args:
        device: Target device
        dtype: Data type for computations (default float64 for precision)
        
    Returns:
        Dictionary of solver keyword arguments
    """
    if device is None:
        device = get_optimal_device(require_float64=(dtype == torch.float64))
    
    # Get configuration ensuring device supports dtype
    config = configure_gpu_optimizations(device, dtype)
    
    # Use the verified device from config
    device = config['device']
    
    return {
        'device': device,
        'dtype': dtype,
        'cache_christoffel': config['cache_christoffel'],
        'cache_size_limit': config['cache_size_limit'],
        'batch_size': config['batch_size'],
        'disable_gradient_computation': config['disable_gradient_computation'],
    }


def estimate_memory_usage(n_steps: int, batch_size: int, state_dim: int = 6,
                         dtype: torch.dtype = torch.float64) -> Tuple[float, float]:
    """
    Estimate memory usage for geodesic integration.
    
    <reason>chain: Help users avoid out-of-memory errors by predicting usage</reason>
    
    Args:
        n_steps: Number of integration steps
        batch_size: Batch size for parallel processing
        state_dim: Dimension of state space (4 or 6)
        dtype: Data type
        
    Returns:
        Tuple of (trajectory_memory_gb, peak_memory_gb)
    """
    bytes_per_element = 8 if dtype == torch.float64 else 4
    
    # Trajectory storage
    trajectory_bytes = n_steps * state_dim * bytes_per_element
    
    # Working memory (batch processing)
    working_bytes = batch_size * state_dim * bytes_per_element * 10  # Factor of 10 for intermediates
    
    # Christoffel cache (rough estimate)
    cache_bytes = 10000 * 64 * bytes_per_element  # 64 symbols per cache entry
    
    trajectory_gb = trajectory_bytes / (1024**3)
    peak_gb = (trajectory_bytes + working_bytes + cache_bytes) / (1024**3)
    
    return trajectory_gb, peak_gb


def get_device_info() -> Dict[str, any]:
    """
    Get information about available compute devices.
    
    <reason>chain: Provide diagnostic information for debugging precision issues</reason>
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cpu': {
            'available': True,
            'supports_float64': True,
            'num_threads': torch.get_num_threads(),
        }
    }
    
    if torch.cuda.is_available():
        info['cuda'] = {
            'available': True,
            'supports_float64': True,
            'device_count': torch.cuda.device_count(),
            'devices': []
        }
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['cuda']['devices'].append({
                'name': props.name,
                'memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}"
            })
    
    if torch.backends.mps.is_available():
        info['mps'] = {
            'available': True,
            'supports_float64': False,  # MPS doesn't support float64
            'warning': 'MPS does not support float64 precision'
        }
    
    return info 