"""
Theory Loader - Dynamically discover and load gravitational theories from the theories/ directory
"""

import os
import sys
import importlib.util
import inspect
from typing import Dict, List, Any, Optional
sys.path.append('..')
from .base_theory import GravitationalTheory


class TheoryLoader:
    """Dynamically loads gravitational theories from the theories/ directory structure."""
    
    def __init__(self, theories_base_dir: str = "theories"):
        self.theories_base_dir = theories_base_dir
        self.loaded_theories: Dict[str, type] = {}
        self.theory_instances: Dict[str, GravitationalTheory] = {}
        
    def discover_theories(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover all available theories in the theories directory.
        
        Returns:
            Dict mapping theory names to their metadata (path, category, parameters, etc.)
        """
        theories = {}
        
        # Walk through the theories directory
        for root, dirs, files in os.walk(self.theories_base_dir):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            # Look for theory.py files directly or in source/ subdirectories
            theory_path = None
            if 'theory.py' in files:
                theory_path = os.path.join(root, 'theory.py')
                rel_path = os.path.relpath(root, self.theories_base_dir)
            elif root.endswith('source') and 'theory.py' in files:
                theory_path = os.path.join(root, 'theory.py')
                theory_parent = os.path.dirname(root)
                rel_path = os.path.relpath(theory_parent, self.theories_base_dir)
            
            if theory_path:
                theory_name = rel_path.replace(os.sep, '/')
                
                # Load and inspect the theory
                try:
                    theory_classes = self._load_theory_classes(theory_path)
                    
                    for class_name, theory_class in theory_classes.items():
                        # Get theory metadata
                        category = getattr(theory_class, 'category', 'unknown')
                        sweep = getattr(theory_class, 'sweep', None)
                        
                        # Get constructor parameters
                        sig = inspect.signature(theory_class.__init__)
                        params = {}
                        for param_name, param in sig.parameters.items():
                            if param_name != 'self':
                                params[param_name] = {
                                    'default': param.default if param.default != inspect.Parameter.empty else None,
                                    'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
                                }
                        
                        # Create a unique key for this theory
                        theory_key = f"{theory_name}/{class_name}"
                        
                        theories[theory_key] = {
                            'path': theory_path,
                            'class_name': class_name,
                            'theory_name': theory_name,
                            'category': category,
                            'parameters': params,
                            'sweep': sweep,
                            'description': theory_class.__doc__ or "No description available"
                        }
                        
                        # Store the class for later instantiation
                        self.loaded_theories[theory_key] = theory_class
                        
                except Exception as e:
                    print(f"Failed to load theory from {theory_path}: {e}")
            
        # Also look for .py files in baselines/ subdirectories
        for root, dirs, files in os.walk(self.theories_base_dir):
            if root.endswith('baselines'):
                for file in files:
                    if file.endswith('.py') and file != '__init__.py':
                        theory_path = os.path.join(root, file)
                        theory_parent = os.path.dirname(root)
                        rel_path = os.path.relpath(theory_parent, self.theories_base_dir)
                        theory_name = rel_path.replace(os.sep, '/')
                        
                        # Load and inspect the theory
                        try:
                            theory_classes = self._load_theory_classes(theory_path)
                            
                            for class_name, theory_class in theory_classes.items():
                                # Get theory metadata
                                category = getattr(theory_class, 'category', 'unknown')
                                sweep = getattr(theory_class, 'sweep', None)
                                
                                # Get constructor parameters
                                sig = inspect.signature(theory_class.__init__)
                                params = {}
                                for param_name, param in sig.parameters.items():
                                    if param_name != 'self':
                                        params[param_name] = {
                                            'default': param.default if param.default != inspect.Parameter.empty else None,
                                            'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
                                        }
                                
                                # Create a unique key for this theory
                                theory_key = f"{theory_name}/{class_name}"
                                
                                theories[theory_key] = {
                                    'path': theory_path,
                                    'class_name': class_name,
                                    'theory_name': theory_name,
                                    'category': category,
                                    'parameters': params,
                                    'sweep': sweep,
                                    'description': theory_class.__doc__ or "No description available"
                                }
                                
                                # Store the class for later instantiation
                                self.loaded_theories[theory_key] = theory_class
                                
                        except Exception as e:
                            print(f"Failed to load theory from {theory_path}: {e}")
                    
        return theories
    
    def _load_theory_classes(self, theory_path: str) -> Dict[str, type]:
        """Load all GravitationalTheory subclasses from a Python file."""
        # Add the directory to sys.path temporarily
        theory_dir = os.path.dirname(theory_path)
        sys.path.insert(0, theory_dir)
        
        try:
            # Load the module with a unique name based on the theory path
            module_name = os.path.splitext(os.path.basename(theory_path))[0]
            theory_dir_name = os.path.basename(os.path.dirname(theory_path))
            
            # Create a fully qualified module name that matches the import path
            # This makes the theory objects pickleable for multiprocessing
            # Convert file path to module path: theories/newtonian_limit/theory.py -> physics_agent.theories.newtonian_limit.theory
            rel_path = os.path.relpath(theory_path, os.path.dirname(self.theories_base_dir))
            module_parts = rel_path.replace(os.sep, '.').replace('.py', '')
            unique_module_name = f"physics_agent.{module_parts}"
            
            spec = importlib.util.spec_from_file_location(unique_module_name, theory_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[unique_module_name] = module  # Register the module
            spec.loader.exec_module(module)
            
            # Find all GravitationalTheory subclasses
            theory_classes = {}
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, GravitationalTheory) and 
                    obj != GravitationalTheory):
                    theory_classes[name] = obj
                    
            return theory_classes
            
        finally:
            # Remove from sys.path
            sys.path.remove(theory_dir)
    
    def instantiate_theory(self, theory_key: str, **kwargs) -> Optional[GravitationalTheory]:
        """
        Instantiate a theory with given parameters.
        
        Args:
            theory_key: The theory identifier (e.g., "linear_signal_loss/LinearSignalLoss")
            **kwargs: Parameters to pass to the theory constructor
            
        Returns:
            An instance of the theory, or None if failed
        """
        if theory_key not in self.loaded_theories:
            print(f"Theory {theory_key} not found in loaded theories")
            return None
            
        try:
            theory_class = self.loaded_theories[theory_key]
            instance = theory_class(**kwargs)
            return instance
        except Exception as e:
            print(f"Failed to instantiate theory {theory_key}: {e}")
            return None
    
    def get_default_instances(self) -> Dict[str, GravitationalTheory]:
        """Get default instances of all discovered theories."""
        instances = {}
        
        for theory_key, theory_class in self.loaded_theories.items():
            try:
                # Try to instantiate with default parameters
                instance = theory_class()
                instances[theory_key] = instance
            except Exception:
                # If default instantiation fails, try with parameter sweep defaults
                theory_info = self.discover_theories().get(theory_key, {})
                params = theory_info.get('parameters', {})
                
                # Build kwargs from parameter defaults
                kwargs = {}
                for param_name, param_info in params.items():
                    if param_info['default'] is not None:
                        kwargs[param_name] = param_info['default']
                
                try:
                    instance = theory_class(**kwargs)
                    instances[theory_key] = instance
                except Exception as e:
                    print(f"Could not instantiate {theory_key} even with defaults: {e}")
                    
        return instances
    
    def get_theory_options(self) -> List[Dict[str, Any]]:
        """
        Get a list of theory options suitable for a dropdown menu.
        
        Returns:
            List of dicts with 'value' (theory_key) and 'label' (display name)
        """
        options = []
        theories = self.discover_theories()
        
        # Group by category
        categories = {}
        for theory_key, info in theories.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((theory_key, info))
        
        # Build options with category grouping
        for category in ['classical', 'quantum', 'quantum', 'unknown']:
            if category in categories:
                for theory_key, info in sorted(categories[category]):
                    # Create a nice display name
                    theory_name = info['theory_name'].replace('_', ' ').title()
                    class_name = info['class_name']
                    
                    # Get parameter info for label
                    params = info['parameters']
                    param_str = ""
                    if params:
                        param_parts = []
                        for pname, pinfo in params.items():
                            if pinfo['default'] is not None:
                                param_parts.append(f"{pname}={pinfo['default']}")
                        if param_parts:
                            param_str = f" ({', '.join(param_parts)})"
                    
                    label = f"[{category.upper()}] {theory_name} - {class_name}{param_str}"
                    
                    options.append({
                        'value': theory_key,
                        'label': label,
                        'category': category,
                        'info': info
                    })
                    
        return options


# Utility function for quick access
def load_all_theories(theories_dir: str = "theories") -> Dict[str, GravitationalTheory]:
    """Load all theories and return their default instances."""
    loader = TheoryLoader(theories_dir)
    loader.discover_theories()
    return loader.get_default_instances() 