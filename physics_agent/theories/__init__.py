"""
Multi-Physics Theory Framework

This framework supports theories across multiple fields of physics:
- Gravitational physics (general relativity and modifications)
- Thermodynamics (black hole thermodynamics, emergent gravity)
- Fluid dynamics (analog gravity, relativistic fluids)
- Quantum mechanics (quantum gravity, decoherence)
- Electromagnetism (charged black holes, plasma physics)
- Particle physics (quantum field theory in curved spacetime)
- Cosmology (dark energy, inflation)

<reason>chain: Restructured to support multiple physics fields beyond just gravity</reason>
"""

import os
import importlib
from typing import List, Dict, Any, Optional

def get_available_fields() -> List[str]:
    """Get list of all available physics fields."""
    return [
        'gravitational',
        'thermodynamic',
        'fluid_dynamics', 
        'electromagnetism',
        'particle_physics',
        'cosmology'
    ]

def discover_theories_in_field(field: str) -> Dict[str, Any]:
    """
    Discover all theories in a specific physics field.
    
    Args:
        field: Physics field name (e.g., 'gravitational', 'thermodynamic')
        
    Returns:
        Dictionary mapping theory names to their module paths
    """
    theories = {}
    
    # Get the base directory for theories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    field_dir = os.path.join(base_dir, field)
    
    if not os.path.exists(field_dir):
        return theories
    
    # Walk through the field directory
    for item in os.listdir(field_dir):
        item_path = os.path.join(field_dir, item)
        
        # Skip non-directories and special directories
        if not os.path.isdir(item_path) or item.startswith('_') or item.startswith('.'):
            continue
            
        # Check if it has a theory.py file
        theory_file = os.path.join(item_path, 'theory.py')
        if os.path.exists(theory_file):
            # Use directory name as theory key
            theories[item] = {
                'module_path': f'physics_agent.theories.{field}.{item}.theory',
                'field': field,
                'directory': item
            }
    
    return theories

def load_theory_class(module_path: str):
    """
    Dynamically load a theory class from its module path.
    
    Args:
        module_path: Full module path (e.g., 'physics_agent.theories.gravitational.kerr.theory')
        
    Returns:
        Theory class or None if loading fails
    """
    try:
        module = importlib.import_module(module_path)
        
        # Find the theory class (should inherit from GravitationalTheory)
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                hasattr(obj, 'get_metric') and 
                name != 'GravitationalTheory'):
                return obj
                
    except Exception as e:
        print(f"Failed to load theory from {module_path}: {e}")
        
    return None

def get_all_theories(fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get all available theories, optionally filtered by fields.
    
    Args:
        fields: List of fields to include (None = all fields)
        
    Returns:
        Dictionary mapping theory identifiers to their info
    """
    all_theories = {}
    
    if fields is None:
        fields = get_available_fields()
    
    for field in fields:
        field_theories = discover_theories_in_field(field)
        
        # Add field prefix to avoid naming conflicts
        for theory_name, theory_info in field_theories.items():
            full_name = f"{field}/{theory_name}"
            all_theories[full_name] = theory_info
    
    return all_theories

def instantiate_theory(theory_identifier: str, **kwargs):
    """
    Instantiate a theory by its identifier.
    
    Args:
        theory_identifier: Either 'field/theory_name' or module path
        **kwargs: Arguments to pass to theory constructor
        
    Returns:
        Theory instance or None if instantiation fails
    """
    # Check if it's a field/name identifier
    if '/' in theory_identifier and not theory_identifier.startswith('physics_agent'):
        field, name = theory_identifier.split('/', 1)
        module_path = f'physics_agent.theories.{field}.{name}.theory'
    else:
        module_path = theory_identifier
    
    theory_class = load_theory_class(module_path)
    
    if theory_class is None:
        return None
        
    try:
        return theory_class(**kwargs)
    except Exception as e:
        print(f"Failed to instantiate theory {theory_identifier}: {e}")
        return None

# For backward compatibility, import some common gravitational theories
try:
    from .gravitational.defaults.baselines.schwarzschild import Schwarzschild
    from .gravitational.defaults.baselines.kerr import Kerr
    from .gravitational.defaults.baselines.kerr_newman import KerrNewman
except ImportError:
    Schwarzschild = None
    Kerr = None
    KerrNewman = None

__all__ = [
    'get_available_fields',
    'discover_theories_in_field',
    'load_theory_class',
    'get_all_theories',
    'instantiate_theory',
    'Schwarzschild',
    'Kerr', 
    'KerrNewman'
]