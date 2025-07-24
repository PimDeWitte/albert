import torch
from functools import wraps
import inspect
import json
import numpy as np
import pickle
import os

def get_metric_wrapper(model_get_metric):
    """Wrapper to handle different get_metric parameter names."""
    sig = inspect.signature(model_get_metric)
    param_map = {
        'M': ['M', 'M_param'],
        'c': ['c', 'C_param'],
        'G': ['G', 'G_param'],
    }

    @wraps(model_get_metric)
    def wrapper(r, M, c, G, **kwargs):
        call_kwargs = {'r': r}
        for target_param, source_params in param_map.items():
            for source_param in source_params:
                if source_param in sig.parameters:
                    # Map from the standard names (M, c, G) to whatever the function expects
                    if target_param == 'M': call_kwargs[source_param] = M
                    elif target_param == 'c': call_kwargs[source_param] = c
                    elif target_param == 'G': call_kwargs[source_param] = G
                    break
        
        # Pass through any other kwargs that the signature accepts
        for param_name in sig.parameters:
            if param_name not in call_kwargs and param_name in kwargs:
                call_kwargs[param_name] = kwargs[param_name]

        return model_get_metric(**call_kwargs)
    
    return wrapper


def save_trajectory(filename: str, trajectory_data: dict, format: str = 'json'):
    """
    Save trajectory data to file.
    
    Args:
        filename: Output filename
        trajectory_data: Dictionary containing trajectory data
        format: 'json' or 'pickle'
    """
    # Convert numpy arrays and tensors to lists for JSON
    def convert_for_json(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    # Ensure directory exists
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    if format == 'json':
        with open(filename, 'w') as f:
            json.dump(convert_for_json(trajectory_data), f, indent=2)
    elif format == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(trajectory_data, f)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_trajectory(filename: str, format: str = None) -> dict:
    """
    Load trajectory data from file.
    
    Args:
        filename: Input filename
        format: 'json' or 'pickle' (auto-detected if None)
        
    Returns:
        Dictionary containing trajectory data
    """
    if format is None:
        # Auto-detect format from extension
        if filename.endswith('.json'):
            format = 'json'
        elif filename.endswith('.pkl') or filename.endswith('.pickle'):
            format = 'pickle'
        else:
            # Try JSON first
            format = 'json'
    
    if format == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Convert complex numbers back
        def convert_complex(obj):
            if isinstance(obj, dict) and 'real' in obj and 'imag' in obj and len(obj) == 2:
                return complex(obj['real'], obj['imag'])
            elif isinstance(obj, dict):
                return {k: convert_complex(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_complex(v) for v in obj]
            return obj
            
        return convert_complex(data)
        
    elif format == 'pickle':
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {format}") 