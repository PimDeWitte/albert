#!/usr/bin/env python3
"""
Error handling utilities for validators to ensure robust execution.
<reason>chain: Provides centralized error handling to prevent crashes during validation</reason>
"""

import torch
import numpy as np
from typing import Any


def safe_tensor_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, handling tensors, numpy arrays, and edge cases.
    
    <reason>chain: Prevents common .item() errors when value is already a float</reason>
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        
    Returns:
        float: The converted value or default
    """
    if value is None:
        return default
    
    # Already a basic Python type
    if isinstance(value, (int, float)):
        return float(value)
    
    # Handle torch tensors
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        else:
            # Multi-element tensor - return mean or first element
            return value.mean().item() if value.numel() > 0 else default
    
    # Handle numpy arrays/scalars
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.flat[0])
        else:
            return float(np.mean(value)) if value.size > 0 else default
    
    # Handle numpy scalar types
    if hasattr(value, 'item'):
        try:
            return float(value.item())
        except:
            pass
    
    # Last resort - try direct conversion
    try:
        return float(value)
    except:
        return default


def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert an object to be JSON serializable.
    
    <reason>chain: Comprehensive conversion handling all common scientific computing types</reason>
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    
    # Handle basic types
    if isinstance(obj, (str, int, bool)):
        return obj
    
    if isinstance(obj, float):
        # Handle special float values
        if np.isnan(obj):
            return "NaN"
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    
    # Handle tensors
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return make_json_serializable(obj.item())
        else:
            return make_json_serializable(obj.tolist())
    
    # Handle numpy types
    if isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    
    if isinstance(obj, (np.integer, np.floating)):
        return make_json_serializable(obj.item())
    
    # Handle complex numbers
    if isinstance(obj, complex):
        return {
            'real': make_json_serializable(obj.real),
            'imag': make_json_serializable(obj.imag),
            '_type': 'complex'
        }
    
    # Handle sequences
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    
    # Handle objects with __dict__
    if hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    
    # Default: convert to string
    return str(obj)


def safe_metric_call(theory, r, M, c, G, device=None, dtype=None):
    """
    Safely call a theory's get_metric method with proper type handling.
    
    <reason>chain: Ensures metric calls don't fail due to type mismatches</reason>
    
    Returns:
        tuple: (g_tt, g_rr, g_pp, g_tp) or None if error
    """
    try:
        # Ensure inputs are tensors
        if not torch.is_tensor(r):
            r = torch.tensor(r, device=device, dtype=dtype)
        if not torch.is_tensor(M):
            M = torch.tensor(M, device=device, dtype=dtype)
        
        # Ensure r has batch dimension
        if r.dim() == 0:
            r = r.unsqueeze(0)
        
        # Call metric with proper types
        result = theory.get_metric(r, M, c, G)
        
        # Validate result
        if result is None or len(result) != 4:
            return None
        
        return result
        
    except Exception as e:
        print(f"Warning: Error in get_metric call: {e}")
        return None


def wrap_validator_method(validator_method):
    """
    Decorator to wrap validator methods with comprehensive error handling.
    
    <reason>chain: Prevents individual validator failures from crashing the entire run</reason>
    """
    def wrapped(self, theory, *args, **kwargs):
        try:
            result = validator_method(self, theory, *args, **kwargs)
            # Ensure result is JSON-serializable
            if isinstance(result, dict):
                return make_json_serializable(result)
            return result
        except Exception as e:
            # Return a proper failure result
            return {
                "loss": 1.0,
                "flags": {"overall": "FAIL"},
                "details": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "validator": getattr(self, 'name', 'Unknown'),
                    "theory": getattr(theory, 'name', 'Unknown')
                }
            }
    
    return wrapped 