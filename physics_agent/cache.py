#!/usr/bin/env python3
"""
Cache Module - Handles all trajectory caching operations
<reason>chain: Separating cache logic improves maintainability and enables better cache strategies</reason>
"""
import os
import re
import torch
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Union, Dict, Any
from torch import Tensor

# Import constants that affect trajectory computation
from physics_agent.constants import NUMERICAL_THRESHOLDS, INTEGRATION_STEP_FACTORS

# <reason>chain: Version control for cache invalidation when core logic changes</reason>
SOFTWARE_VERSION = "1.0.0"


class TrajectoryCache:
    """
    Manages caching of computed trajectories for gravitational theories.
    <reason>chain: Encapsulating cache logic in a class enables better state management and testing</reason>
    """
    
    def __init__(self, cache_base_dir: Optional[str] = None):
        """
        Initialize the trajectory cache manager.
        
        Args:
            cache_base_dir: Base directory for cache storage. If None, uses default location.
        """
        if cache_base_dir is None:
            # <reason>chain: Default to physics_agent/cache directory</reason>
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cache_base_dir = os.path.join(script_dir, 'cache')
            
        self.cache_base_dir = cache_base_dir
        self.trajectories_dir = os.path.join(cache_base_dir, 'trajectories', SOFTWARE_VERSION)
        
        # <reason>chain: Ensure cache directories exist</reason>
        os.makedirs(self.trajectories_dir, exist_ok=True)
        
    def get_cache_path(self, 
                      theory_name: str, 
                      r0: Union[Tensor, float], 
                      n_steps: int, 
                      dtau: Union[Tensor, float], 
                      dtype_str: str,
                      **kwargs) -> str:
        """
        Generate a unique cache path for a trajectory.
        <reason>chain: Include all parameters that affect trajectory computation in cache key</reason>
        
        Args:
            theory_name: Name of the gravitational theory
            r0: Initial radius (tensor or float)
            n_steps: Number of simulation steps
            dtau: Time step size
            dtype_str: String representation of data type (e.g., 'float32')
            **kwargs: Additional parameters that affect trajectory
            
        Returns:
            Full path to cache file
        """
        # <reason>chain: Handle both tensor and scalar inputs flexibly</reason>
        r0_val = r0.item() if isinstance(r0, torch.Tensor) else float(r0)
        dtau_val = dtau.item() if isinstance(dtau, torch.Tensor) else float(dtau)
        
        # <reason>chain: Sanitize theory name for filesystem compatibility</reason>
        sanitized_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', theory_name)
        
        # <reason>chain: Include all parameters except n_steps in hash to avoid cache collisions</reason>
        extra_params = {
            'r0': r0_val,
            'dtau': dtau_val,
            'dtype': dtype_str
        }
        for key in ['run_to_horizon', 'horizon_threshold', 'particle_name', 
            'particle_mass', 'particle_charge', 'particle_spin', 'particle_type',
            'quantum_interval', 'quantum_beta', 'y0_general', 'singularity_threshold']:
            if key in kwargs and kwargs[key] is not None:
                if key == 'y0_general' and isinstance(kwargs[key], torch.Tensor):
                    # <reason>chain: Handle tensor parameters by converting to hashable format</reason>
                    extra_params[key] = kwargs[key].cpu().numpy().tobytes()
                else:
                    extra_params[key] = kwargs[key]
        
        # <reason>chain: Include theory class name and module to distinguish between different theory implementations</reason>
        # This prevents cache collisions between different theories with similar names
        if 'theory_module' in kwargs:
            extra_params['theory_module'] = kwargs['theory_module']
        if 'theory_class' in kwargs:
            extra_params['theory_class'] = kwargs['theory_class']
        
        # <reason>chain: Include metric-specific parameters that affect the trajectory</reason>
        # Add any theory parameters that would change the metric (a, q_e, alpha, etc.)
        if 'metric_params' in kwargs:
            extra_params['metric_params'] = str(sorted(kwargs['metric_params'].items()))
        
        # <reason>chain: Include critical numerical constants that affect trajectory computation</reason>
        # This ensures cache invalidation when constants change
        extra_params['numerical_thresholds'] = {
            'radius_min': NUMERICAL_THRESHOLDS['radius_min'],
            'radius_max': NUMERICAL_THRESHOLDS['radius_max'],
            'singularity_radius': NUMERICAL_THRESHOLDS['singularity_radius'],
            'orbit_stability': NUMERICAL_THRESHOLDS['orbit_stability'],
            'epsilon': NUMERICAL_THRESHOLDS['epsilon'],
            'norm_check': NUMERICAL_THRESHOLDS['norm_check']
        }
        
        # <reason>chain: Include integration step factors that affect numerical accuracy</reason>
        extra_params['integration_factors'] = {
            'standard_reduction': INTEGRATION_STEP_FACTORS['standard_reduction'],
            'aggressive_reduction': INTEGRATION_STEP_FACTORS['aggressive_reduction'],
            'ergo_sphere_limit': INTEGRATION_STEP_FACTORS['ergo_sphere_limit']
        }
        
        # <reason>chain: Include software version for cache invalidation on updates</reason>
        extra_params['software_version'] = SOFTWARE_VERSION
        
        # <reason>chain: Include physical constants if provided (affects unit conversions)</reason>
        if 'M_si' in kwargs:
            extra_params['M_si'] = kwargs['M_si']
        if 'c_si' in kwargs:
            extra_params['c_si'] = kwargs['c_si']
        if 'G_si' in kwargs:
            extra_params['G_si'] = kwargs['G_si']
        
        # <reason>chain: Include black hole preset to ensure cache separation by mass</reason>
        # This is critical because trajectories depend strongly on the central mass
        if 'black_hole_preset' in kwargs:
            extra_params['black_hole_preset'] = kwargs['black_hole_preset']
                    
        # Always use hash-based filename for consistency and to include all parameters
        # <reason>chain: Generate hash of all parameters except n_steps for unique identification</reason>
        # <reason>chain: This allows longer trajectories to be broken down into shorter cached segments</reason>
        param_hash = hashlib.sha256(
            str(sorted(extra_params.items())).encode()
        ).hexdigest()[:16]  # Use 16 characters for good uniqueness
        
        # <reason>chain: Format: (theory)_(params_hash)_steps_(n_steps).pt</reason>
        # <reason>chain: The _steps suffix allows fetching partial trajectories from longer runs</reason>
        filename = f"{sanitized_name}_{param_hash}_steps_{n_steps}.pt"
        
        # <reason>chain: Organize cache by black hole preset for clarity</reason>
        # Create subdirectory for black hole preset if provided
        if 'black_hole_preset' in kwargs:
            bh_preset = kwargs['black_hole_preset']
            # Sanitize preset name for filesystem
            sanitized_preset = re.sub(r'[^a-zA-Z0-9_\-]', '_', bh_preset)
            preset_dir = os.path.join(self.trajectories_dir, sanitized_preset)
            os.makedirs(preset_dir, exist_ok=True)
            return os.path.join(preset_dir, filename)
        else:
            # Fallback to root trajectories directory for legacy compatibility
            return os.path.join(self.trajectories_dir, filename)
        
    def load_trajectory(self, cache_path: str, device: torch.device, 
                       max_steps: Optional[int] = None) -> Optional[Tensor]:
        """
        Load a cached trajectory from disk.
        <reason>chain: Separate loading logic for better error handling</reason>
        <reason>chain: Supports loading partial trajectories from longer cache files</reason>
        
        Args:
            cache_path: Path to cache file
            device: PyTorch device to load tensor to
            max_steps: Maximum number of steps to load (None = load all)
            
        Returns:
            Loaded trajectory tensor or None if loading fails
        """
        if not os.path.exists(cache_path):
            # <reason>chain: Try to find a longer trajectory we can use</reason>
            if max_steps is not None:
                longer_cache = self._find_longer_trajectory(cache_path, max_steps)
                if longer_cache:
                    cache_path = longer_cache
                else:
                    return None
            else:
                return None
            
        try:
            print(f"Loading cached trajectory from: {cache_path}")
            trajectory = torch.load(cache_path, map_location=device)
            
            # <reason>chain: Return only the requested number of steps if specified</reason>
            if max_steps is not None and trajectory.shape[0] > max_steps:
                print(f"  Truncating trajectory from {trajectory.shape[0]} to {max_steps} steps")
                trajectory = trajectory[:max_steps]
                
            return trajectory
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None
            
    def save_trajectory(self, 
                       trajectory: Tensor, 
                       cache_path: str,
                       dtype: torch.dtype) -> bool:
        """
        Save a computed trajectory to cache.
        <reason>chain: Separate saving logic with proper error handling</reason>
        
        Args:
            trajectory: Trajectory tensor to save
            cache_path: Path where to save the cache file
            dtype: Data type to save tensor as
            
        Returns:
            True if save successful, False otherwise
        """
        if trajectory is None or trajectory.shape[0] <= 1:
            return False
            
        try:
            # <reason>chain: Ensure directory exists before saving</reason>
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # <reason>chain: Convert to specified dtype before saving</reason>
            torch.save(trajectory.to(dtype=dtype), cache_path)
            print(f"Saved trajectory to cache: {cache_path}")
            return True
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            return False
    
    def _find_longer_trajectory(self, requested_path: str, min_steps: int) -> Optional[str]:
        """
        Find a cached trajectory with more steps than requested.
        <reason>chain: Allows reusing longer trajectory runs by loading partial data</reason>
        
        Args:
            requested_path: The originally requested cache path
            min_steps: Minimum number of steps needed
            
        Returns:
            Path to a longer trajectory file, or None if not found
        """
        # Extract base pattern from requested filename
        base_dir = os.path.dirname(requested_path)
        filename = os.path.basename(requested_path)
        
        # Parse the filename pattern: (theory)_(hash)_steps_(N).pt
        import re
        match = re.match(r'(.+)_([a-f0-9]+)_steps_(\d+)\.pt$', filename)
        if not match:
            return None
            
        theory_name, param_hash, requested_steps = match.groups()
        requested_steps = int(requested_steps)
        
        # Look for files with same theory and hash but more steps
        pattern = f"{theory_name}_{param_hash}_steps_*.pt"
        
        try:
            import glob
            matching_files = glob.glob(os.path.join(base_dir, pattern))
            
            # Find files with enough steps
            candidates = []
            for file_path in matching_files:
                file_match = re.match(r'.+_steps_(\d+)\.pt$', os.path.basename(file_path))
                if file_match:
                    file_steps = int(file_match.group(1))
                    if file_steps >= min_steps:
                        candidates.append((file_path, file_steps))
            
            # Return the file with fewest steps that still meets requirement
            if candidates:
                candidates.sort(key=lambda x: x[1])
                chosen_path, chosen_steps = candidates[0]
                print(f"  Found longer trajectory: {chosen_steps} steps (requested {requested_steps})")
                return chosen_path
                
        except Exception as e:
            print(f"Warning: Error searching for longer trajectories: {e}")
            
        return None
            
    def get_cache_info(self, theory_dir: str) -> Dict[str, Any]:
        """
        Extract cache information for a theory run.
        <reason>chain: Provide cache statistics for analysis and debugging</reason>
        
        Args:
            theory_dir: Directory containing theory results
            
        Returns:
            Dictionary with cache information
        """
        cache_info = {"was_cached": False}
        
        # <reason>chain: Check if cached trajectory exists</reason>
        cache_path = os.path.join(theory_dir, "trajectory_cached.pt")
        if os.path.exists(cache_path):
            cache_info["was_cached"] = True
            
            # <reason>chain: Get cache file statistics</reason>
            cache_stat = os.stat(cache_path)
            cache_info["cache_created"] = cache_stat.st_mtime
            cache_info["cache_size_bytes"] = cache_stat.st_size
            
            # <reason>chain: Load trajectory to get properties</reason>
            try:
                trajectory = torch.load(cache_path, map_location='cpu')
                cache_info["cache_properties"] = {
                    "shape": list(trajectory.shape),
                    "dtype": str(trajectory.dtype),
                    "steps": trajectory.shape[0] if len(trajectory.shape) > 0 else 0
                }
            except Exception as e:
                cache_info["cache_properties"] = {"error": str(e)}
                
        return cache_info
        
    def clear_cache(self, confirm: bool = True) -> None:
        """
        Clear all cached trajectories.
        <reason>chain: Provide safe cache clearing with optional confirmation</reason>
        
        Args:
            confirm: Whether to ask for user confirmation before clearing
        """
        cache_dirs = [
            self.trajectories_dir,
            os.path.dirname(self.trajectories_dir),  # Parent directory
            self.cache_base_dir
        ]
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                if confirm:
                    response = input(f"Clear cache directory {cache_dir}? [y/N]: ")
                    if response.lower() != 'y':
                        continue
                        
                try:
                    print(f"Clearing cache directory: {cache_dir}")
                    shutil.rmtree(cache_dir)
                    print("Cache cleared.")
                except Exception as e:
                    print(f"Error clearing cache: {e}")
                    
    def clear_old_cache(self) -> None:
        """
        Clear old trajectory cache files that use outdated naming schemes.
        <reason>chain: Maintain cache hygiene by removing obsolete files</reason>
        """
        old_pattern = re.compile(r'^.*_r0-\d+\.\d+_steps-\d+_dt-\d+\.\d+_dtype-.*\.pt$')
        new_pattern = re.compile(r'^.*_r\d+\.\d+_[a-f0-9]{12}\.pt$')
        
        total_removed = 0
        total_size = 0
        
        # <reason>chain: Walk through cache directory to find old files</reason>
        for root, dirs, files in os.walk(self.cache_base_dir):
            for file in files:
                if file.endswith('.pt'):
                    file_path = Path(root) / file
                    
                    # <reason>chain: Check if file matches old pattern but not new pattern</reason>
                    if old_pattern.match(file) and not new_pattern.match(file):
                        size = file_path.stat().st_size
                        print(f"  Removing old cache file: {file} ({size/1024:.1f} KB)")
                        file_path.unlink()
                        total_removed += 1
                        total_size += size
                        
        print(f"\nSummary:")
        print(f"  Removed {total_removed} old cache files")
        print(f"  Freed {total_size/1024/1024:.1f} MB of disk space")
        
        if total_removed == 0:
            print("  No old cache files found.")
        else:
            print("\nOld cache files cleared. New trajectories will be computed with proper parameter hashing.")


# <reason>chain: Module-level convenience functions for backward compatibility</reason>
def clear_cache(confirm: bool = False) -> None:
    """Clear the trajectory cache directory."""
    cache = TrajectoryCache()
    cache.clear_cache(confirm=confirm)
    

def get_trajectory_cache_path(theory_name: str, 
                            r0: Union[Tensor, float], 
                            n_steps: int, 
                            dtau: Union[Tensor, float], 
                            dtype_str: str,
                            **kwargs) -> str:
    """Get cache path for a trajectory (backward compatibility wrapper)."""
    cache = TrajectoryCache()
    return cache.get_cache_path(theory_name, r0, n_steps, dtau, dtype_str, **kwargs) 