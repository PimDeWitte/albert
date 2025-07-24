import os
import json
import hashlib
import platform
import datetime
import torch
import numpy as np
from typing import Dict, Any
import subprocess
import sys

class ReproducibilityFramework:
    """
    Ensures complete reproducibility of scientific results.
    Logs all parameters, environment details, and provides verification tools.
    """
    
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'environment': self._capture_environment(),
            'parameters': {},
            'checksums': {},
            'random_seeds': {}
        }
        
    def _capture_environment(self) -> Dict[str, Any]:
        """Capture complete computational environment."""
        env = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': sys.version,
                'python_implementation': platform.python_implementation()
            },
            'hardware': {
                'cpu_count': os.cpu_count(),
                'cuda_available': torch.cuda.is_available(),
                'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            },
            'libraries': {
                'torch': torch.__version__,
                'numpy': np.__version__,
                'scipy': self._get_package_version('scipy'),
                'matplotlib': self._get_package_version('matplotlib')
            },
            'git': self._capture_git_state()
        }
        
        # Capture CUDA details if available
        if torch.cuda.is_available():
            env['hardware']['cuda_version'] = torch.version.cuda
            env['hardware']['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            
        return env
        
    def _get_package_version(self, package: str) -> str:
        """Get version of installed package."""
        try:
            module = __import__(package)
            return getattr(module, '__version__', 'unknown')
        except ImportError:
            return 'not installed'
            
    def _capture_git_state(self) -> Dict[str, str]:
        """Capture git commit hash and diff."""
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
            
            # Check for uncommitted changes
            status = subprocess.check_output(['git', 'status', '--porcelain']).decode()
            has_changes = len(status.strip()) > 0
            
            # Get diff if there are changes
            diff = ''
            if has_changes:
                diff = subprocess.check_output(['git', 'diff']).decode()
                
            return {
                'commit': commit,
                'branch': branch,
                'has_uncommitted_changes': has_changes,
                'diff': diff if has_changes else None
            }
        except subprocess.CalledProcessError:
            return {'error': 'Not a git repository'}
            
    def log_parameters(self, params: Dict[str, Any]):
        """Log all simulation parameters."""
        self.metadata['parameters'].update(params)
        
    def log_random_state(self, name: str = 'default'):
        """Log current random number generator states."""
        self.metadata['random_seeds'][name] = {
            'torch': torch.get_rng_state().tolist(),
            'numpy': np.random.get_state()[1].tolist(),
            'cuda': torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None
        }
        
    def compute_data_checksum(self, data: Any, name: str):
        """Compute and store checksum of data for verification."""
        if isinstance(data, torch.Tensor):
            data_bytes = data.cpu().numpy().tobytes()
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        else:
            data_bytes = json.dumps(data, sort_keys=True).encode()
            
        checksum = hashlib.sha256(data_bytes).hexdigest()
        self.metadata['checksums'][name] = checksum
        return checksum
        
    def save_metadata(self):
        """Save all reproducibility metadata."""
        metadata_path = os.path.join(self.run_dir, 'reproducibility_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def verify_reproduction(self, other_metadata_path: str) -> Dict[str, Any]:
        """Verify if results can be reproduced by comparing metadata."""
        with open(other_metadata_path, 'r') as f:
            other_metadata = json.load(f)
            
        verification = {
            'compatible': True,
            'warnings': [],
            'errors': []
        }
        
        # Check environment compatibility
        if self.metadata['environment']['platform']['system'] != other_metadata['environment']['platform']['system']:
            verification['warnings'].append(f"Different OS: {self.metadata['environment']['platform']['system']} vs {other_metadata['environment']['platform']['system']}")
            
        # Check library versions
        for lib, version in self.metadata['environment']['libraries'].items():
            other_version = other_metadata['environment']['libraries'].get(lib)
            if version != other_version:
                verification['warnings'].append(f"Different {lib} version: {version} vs {other_version}")
                
        # Check parameters
        for param, value in self.metadata['parameters'].items():
            other_value = other_metadata['parameters'].get(param)
            if value != other_value:
                verification['errors'].append(f"Different parameter {param}: {value} vs {other_value}")
                verification['compatible'] = False
                
        # Check git state
        if self.metadata['environment']['git'].get('commit') != other_metadata['environment']['git'].get('commit'):
            verification['warnings'].append("Different git commits")
            
        return verification
        
    def generate_reproduction_script(self) -> str:
        """Generate script to reproduce the results."""
        script = f"""#!/usr/bin/env python3
# Auto-generated reproduction script
# Generated: {self.metadata['timestamp']}

import torch
import numpy as np
import os
import sys

# Set environment
os.environ['PYTHONHASHSEED'] = '0'
torch.set_num_threads({os.cpu_count()})

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Import and run
from physics_agent.theory_engine_core import TheoryEngine

# Parameters
params = {json.dumps(self.metadata['parameters'], indent=4)}

# Initialize engine
engine = TheoryEngine(**params.get('engine_params', {{}}))

# Run simulation
# ... (complete based on actual run)
"""
        return script 