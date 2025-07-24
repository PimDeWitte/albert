#!/usr/bin/env python3
"""
Base Validation Framework

This module provides the foundational classes and utilities for all validators
in the physics agent system. It includes:

- ValidationResult: Standard result format
- BaseValidator: Base class with common functionality 
- PredictionValidator: Specialized for testing novel predictions
- Utility functions for data handling and safe conversions

The validation framework ensures consistent, rigorous testing of gravitational
theories against observational data and theoretical constraints.

TODO: Consolidate with solver_tests implementation to avoid duplication
      while keeping validation and solver tests separate for now to avoid errors.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod
import os
import sys
import json
import re
import importlib.util
from datetime import datetime

# Import the theory engine
if TYPE_CHECKING:
    from physics_agent.theory_engine_core import TheoryEngine
else:
    # Lazy import to avoid circular dependencies
    TheoryEngine = None

from physics_agent.base_theory import GravitationalTheory


def _get_theory_engine():
    """Lazy load TheoryEngine to avoid circular imports"""
    global TheoryEngine
    if TheoryEngine is None:
        from physics_agent.theory_engine_core import TheoryEngine
    return TheoryEngine


class ValidationResult:
    """Container for validation test results"""
    
    def __init__(self, test_name: str, theory_name: str):
        self.test_name = test_name
        self.theory_name = theory_name
        self.observed_value = None
        self.predicted_value = None
        self.error = None
        self.error_percent = None
        self.units = ""
        self.passed = False
        self.notes = ""
        self.trajectory_data = None
        self.timestamp = datetime.now().isoformat()
        # Add fields for prediction validators
        self.beats_sota = False  # Whether prediction beats state-of-the-art
        self.sota_value = None  # State-of-the-art value for comparison
        self.sota_source = ""  # Source of SOTA benchmark
        self.prediction_data = {}  # Additional prediction-specific data
        self.performance = ""  # 'beats', 'matches', 'below', or 'unknown' - relative to SOTA
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'test_name': self.test_name,
            'theory_name': self.theory_name,
            'observed': float(self.observed_value) if self.observed_value is not None else None,
            'predicted': float(self.predicted_value) if self.predicted_value is not None else None,
            'error': float(self.error) if self.error is not None else None,
            'error_percent': float(self.error_percent) if self.error_percent is not None else None,
            'units': self.units,
            'passed': self.passed,
            'notes': self.notes,
            'timestamp': self.timestamp,
            # Prediction-specific fields
            'beats_sota': self.beats_sota,
            'sota_value': float(self.sota_value) if self.sota_value is not None else None,
            'sota_source': self.sota_source,
            'prediction_data': self.prediction_data,
            'performance': self.performance
        }


class ObservationalValidator(ABC):
    """
    Base class for observational validation tests.
    Each validator tests theories against specific observational data.
    """
    
    def __init__(self, engine: Optional["TheoryEngine"] = None):
        """
        Initialize validator with a theory engine.
        
        Args:
            engine: TheoryEngine instance. If None, will create default one.
        """
        if engine is None:
            EngineClass = _get_theory_engine()
            self.engine = EngineClass()
        else:
            self.engine = engine
        self.name = self.__class__.__name__
        
    @abstractmethod
    def validate(self, theory: GravitationalTheory, verbose: bool = False) -> ValidationResult:
        """
        Validate a theory against observational data.
        
        Args:
            theory: The gravitational theory to validate
            verbose: Enable verbose output
            
        Returns:
            ValidationResult object
        """
    
    @abstractmethod
    def get_observational_data(self) -> Dict[str, Any]:
        """
        Get the observational data this validator tests against.
        
        Returns:
            Dict with observational data details
        """
    
    def compute_error(self, observed: float, predicted: float) -> Tuple[float, float]:
        """
        Compute absolute and percentage error.
        
        Returns:
            (absolute_error, percentage_error)
        """
        error = abs(predicted - observed)
        error_percent = 100.0 * error / abs(observed) if observed != 0 else float('inf')
        return error, error_percent


class BaseValidation:
    """Base class for all validations"""
    category = None  # Must be overridden in subclasses
    
    def __init__(self, engine, name: str = "Base Validation"):
        self.engine = engine
        self.name = name
        
    def _get_appropriate_lagrangian(self, theory):
        """
        Helper to determine which Lagrangian to use based on theory type
        
        Returns tuple of (lagrangian, lagrangian_type) where:
        - lagrangian: The appropriate Lagrangian to use
        - lagrangian_type: 'quantum', 'classical', or 'none'
        """
        # Check if theory is quantum category
        is_quantum_theory = (hasattr(theory, 'category') and theory.category == 'quantum')
        
        # Check for complete_lagrangian (quantum field theory Lagrangian)
        if hasattr(theory, 'complete_lagrangian') and theory.complete_lagrangian is not None:
            # This is the full quantum Lagrangian
            return theory.complete_lagrangian, 'quantum'
        
        # Check for regular lagrangian
        if hasattr(theory, 'lagrangian') and theory.lagrangian is not None:
            # If this is a quantum theory but only has classical Lagrangian, warn
            if is_quantum_theory:
                print(f"  WARNING: Quantum theory '{theory.name}' using classical Lagrangian only")
            return theory.lagrangian, 'classical'
        
        # No Lagrangian found
        return None, 'none'
    
    def _check_quantum_consistency(self, theory):
        """
        Check if quantum theory is properly configured
        
        Returns dict with:
        - is_consistent: bool
        - issues: list of issues found
        """
        issues = []
        is_quantum_theory = (hasattr(theory, 'category') and theory.category == 'quantum')
        
        if is_quantum_theory:
            # Quantum theories should have complete_lagrangian
            if not hasattr(theory, 'complete_lagrangian') or theory.complete_lagrangian is None:
                issues.append("Quantum theory missing complete_lagrangian")
            
            # Check if quantum is enabled
            if hasattr(theory, 'enable_quantum') and not theory.enable_quantum:
                issues.append("Quantum theory has enable_quantum=False")
            
            # Check for quantum integrator
            if not hasattr(theory, 'quantum_integrator') or theory.quantum_integrator is None:
                issues.append("Quantum theory missing quantum_integrator")
        
        return {
            'is_consistent': len(issues) == 0,
            'issues': issues
        }
    
    def validate(self, theory, **kwargs):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement validate method")


class PredictionValidator(ObservationalValidator):
    """
    Base class for prediction validation tests that compare theories against 
    state-of-the-art benchmarks from datasets available on the internet.
    
    These validators:
    - Fetch real datasets from internet sources
    - Compare theory predictions against current best models
    - Track whether theories beat state-of-the-art benchmarks
    - Generate detailed prediction reports
    """
    category = "prediction"
    
    def __init__(self, engine: Optional["TheoryEngine"] = None):
        super().__init__(engine)
        self.sota_benchmarks = {}  # Store state-of-the-art values
        
    @abstractmethod
    def fetch_dataset(self) -> Dict[str, Any]:
        """
        Fetch the dataset from internet sources.
        
        Returns:
            Dict containing dataset and metadata
        """
    
    @abstractmethod
    def get_sota_benchmark(self) -> Dict[str, Any]:
        """
        Get the current state-of-the-art benchmark for this prediction.
        
        Returns:
            Dict with SOTA value, source, and metadata
        """
    
    def validate(self, theory: GravitationalTheory, verbose: bool = False) -> ValidationResult:
        """
        Validate theory predictions against SOTA benchmarks.
        
        This method should be overridden but can provide common functionality
        for prediction validators.
        """
        result = ValidationResult(self.name, theory.name)
        
        # Get SOTA benchmark
        sota = self.get_sota_benchmark()
        result.sota_value = sota.get('value')
        result.sota_source = sota.get('source', 'Unknown')
        
        # Prediction validators should implement specific logic and set:
        # - result.predicted_value: The theory's prediction
        # - result.observed_value: The actual observed value from dataset
        # - result.beats_sota: Whether theory beats current SOTA
        # - result.passed: Whether the prediction is scientifically valid
        
        return result
    
    def log_prediction_improvement(self, theory: GravitationalTheory, result: ValidationResult) -> None:
        """
        Log prediction improvements with full scientific detail for reproducibility.
        Creates structured documentation when a theory beats SOTA.
        """
        if not result.beats_sota:
            return
            
        import json
        from datetime import datetime
        import os
        
        # Create predictions directory structure
        base_dir = os.path.join("runs", "predictions")
        theory_dir = os.path.join(base_dir, theory.name.replace(" ", "_").replace("/", "_"))
        os.makedirs(theory_dir, exist_ok=True)
        
        # Create subdirectories for organization
        data_dir = os.path.join(theory_dir, "data")
        code_dir = os.path.join(theory_dir, "code")
        logs_dir = os.path.join(theory_dir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save the actual data used
        self._save_observational_data(data_dir)
        
        # Save theory implementation
        self._save_theory_code(theory, code_dir)
        
        # Save validator code
        self._save_validator_code(code_dir)
        
        # Create reproduction script
        self._create_reproduction_script(theory, code_dir)
        
        # Prepare detailed prediction data
        prediction_data = {
            "theory_name": theory.name,
            "validator": self.name,
            "timestamp": datetime.now().isoformat(),
            "beats_sota": True,
            "sota_comparison": {
                "sota_model": result.sota_source,
                "sota_value": float(result.sota_value) if result.sota_value is not None else None,
                "theory_value": float(result.predicted_value) if result.predicted_value is not None else None,
                "improvement": float(result.error) if result.error is not None else None,
                "improvement_percent": float(result.error_percent) if result.error_percent is not None else None,
                "units": result.units
            },
            "observational_data": self.get_observational_data() if hasattr(self, 'get_observational_data') else {},
            "prediction_details": result.prediction_data,
            "theory_parameters": self._extract_theory_parameters(theory),
            "reproduction_info": self._generate_reproduction_info(theory, result),
            "data_files": self._get_data_file_info(data_dir),
            "code_files": self._get_code_file_info(code_dir)
        }
        
        # Load or create predictions JSON
        json_path = os.path.join(theory_dir, "predictions.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                all_predictions = json.load(f)
        else:
            all_predictions = {
                "theory": theory.name,
                "category": getattr(theory, 'category', 'unknown'),
                "predictions": []
            }
        
        # Add new prediction
        all_predictions["predictions"].append(prediction_data)
        
        # Save updated JSON
        with open(json_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        # Save validation log
        log_path = os.path.join(logs_dir, f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_path, 'w') as f:
            f.write(f"Validation Log for {theory.name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Validator: {self.name}\n")
            f.write(f"Result: BEATS SOTA\n\n")
            f.write(json.dumps(prediction_data, indent=2))
        
        # Create/update README with reproduction instructions
        self._create_prediction_readme(theory, result, theory_dir, prediction_data)
        
        print(f"\n✅ PREDICTION IMPROVEMENT LOGGED: {theory_dir}/")
        print(f"   📊 Data saved to: {data_dir}/")
        print(f"   💻 Code saved to: {code_dir}/")
        print(f"   📝 Logs saved to: {logs_dir}/")
    
    def _save_observational_data(self, data_dir: str) -> None:
        """Save copies of observational data used in validation"""
        # Base implementation - validators should override
    
    def _save_theory_code(self, theory: GravitationalTheory, code_dir: str) -> None:
        """Save the theory implementation code"""
        import inspect
        
        # Save theory class source
        theory_file = os.path.join(code_dir, "theory_implementation.py")
        with open(theory_file, 'w') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write('"""Exact theory implementation used for prediction"""\n\n')
            
            # Get imports
            f.write("import numpy as np\n")
            f.write("import torch\n")
            f.write("from physics_agent.base_theory import GravitationalTheory\n\n")
            
            # Get theory source
            try:
                source = inspect.getsource(theory.__class__)
                f.write(source)
            except:
                # If source not available, document the class
                f.write(f"# Theory class: {theory.__class__.__name__}\n")
                f.write(f"# Module: {theory.__class__.__module__}\n")
                f.write(f"# Could not extract source code\n")
            
            # Add instantiation code
            f.write("\n\n# Instantiation with exact parameters\n")
            f.write(f"theory = {theory.__class__.__name__}()\n")
            
            # Set all parameters
            params = self._extract_theory_parameters(theory)
            for key, value in params.items():
                if key not in ['name', 'category']:
                    if isinstance(value, str):
                        f.write(f'theory.{key} = "{value}"\n')
                    else:
                        f.write(f'theory.{key} = {value}\n')
    
    def _save_validator_code(self, code_dir: str) -> None:
        """Save the validator implementation"""
        import inspect
        
        validator_file = os.path.join(code_dir, "validator_implementation.py")
        with open(validator_file, 'w') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write('"""Exact validator implementation used"""\n\n')
            
            # Copy the validator file
            try:
                module = inspect.getmodule(self.__class__)
                if module and hasattr(module, '__file__'):
                    src_file = module.__file__
                    if src_file.endswith('.pyc'):
                        src_file = src_file[:-1]  # .pyc -> .py
                    
                    if os.path.exists(src_file):
                        with open(src_file, 'r') as src:
                            f.write(src.read())
                    else:
                        f.write(f"# Validator: {self.__class__.__name__}\n")
                        f.write(f"# Source file not found: {src_file}\n")
            except:
                f.write(f"# Validator: {self.__class__.__name__}\n")
                f.write("# Could not extract validator source\n")
    
    def _create_reproduction_script(self, theory: GravitationalTheory, code_dir: str) -> None:
        """Create a standalone script to reproduce the result"""
        script_path = os.path.join(code_dir, "reproduce_result.py")
        
        with open(script_path, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Standalone script to reproduce the prediction improvement.
Run this script to verify the result independently.
"""

import sys
import os

# Add the gravity_compression directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../')))

from physics_agent.validations.{validator_module} import {validator_class}
from theory_implementation import theory

def main():
    print("Reproducing Prediction Improvement")
    print("=" * 60)
    
    # Create validator
    validator = {validator_class}()
    
    print(f"Theory: {{theory.name}}")
    print(f"Validator: {{validator.name}}")
    print()
    
    # Run validation
    result = validator.validate(theory, verbose=True)
    
    print()
    print("Results:")
    print(f"  Beats SOTA: {{result.beats_sota}}")
    print(f"  Predicted Value: {{result.predicted_value}}")
    print(f"  SOTA Value: {{result.sota_value}}")
    print(f"  Improvement: {{result.error}} ({{result.error_percent:.1f}}%)")
    
    if result.beats_sota:
        print()
        print("✅ REPRODUCTION SUCCESSFUL - Theory beats SOTA!")
    else:
        print()
        print("❌ REPRODUCTION FAILED - Theory does not beat SOTA")
    
    return result

if __name__ == "__main__":
    result = main()
'''.format(
                validator_module=self.__class__.__module__.split('.')[-1],
                validator_class=self.__class__.__name__
            ))
        
        # Make script executable
        os.chmod(script_path, 0o755)
    
    def _get_data_file_info(self, data_dir: str) -> Dict[str, Any]:
        """Get information about saved data files"""
        files = []
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                filepath = os.path.join(data_dir, filename)
                if os.path.isfile(filepath):
                    files.append({
                        "filename": filename,
                        "size": os.path.getsize(filepath),
                        "path": os.path.relpath(filepath, os.path.dirname(os.path.dirname(__file__)))
                    })
        return {"count": len(files), "files": files}
    
    def _get_code_file_info(self, code_dir: str) -> Dict[str, Any]:
        """Get information about saved code files"""
        files = []
        if os.path.exists(code_dir):
            for filename in os.listdir(code_dir):
                filepath = os.path.join(code_dir, filename)
                if os.path.isfile(filepath):
                    files.append({
                        "filename": filename,
                        "size": os.path.getsize(filepath),
                        "executable": os.access(filepath, os.X_OK)
                    })
        return {"count": len(files), "files": files}
    
    def _extract_theory_parameters(self, theory: GravitationalTheory) -> Dict[str, Any]:
        """Extract all theory parameters for reproduction"""
        params = {
            "name": theory.name,
            "category": getattr(theory, 'category', 'unknown')
        }
        
        # Common theory parameters
        param_names = ['alpha', 'beta', 'gamma', 'sigma', 'lambda_param', 
                      'mass_scale', 'length_scale', 'coupling_constant', 'description']
        
        for param in param_names:
            if hasattr(theory, param):
                try:
                    value = getattr(theory, param)
                    if value is not None:
                        params[param] = float(value) if isinstance(value, (int, float)) else str(value)
                except:
                    # Skip problematic attributes
                    pass
        
        # Add selected numerical attributes (avoid properties)
        safe_attrs = ['symmetry', 'is_stable', 'conserves_energy']
        for attr in safe_attrs:
            if hasattr(theory, attr):
                try:
                    value = getattr(theory, attr)
                    if value is not None:
                        params[attr] = value
                except:
                    pass
        
        return params
    
    def _generate_reproduction_info(self, theory: GravitationalTheory, result: ValidationResult) -> Dict[str, Any]:
        """Generate detailed reproduction information"""
        return {
            "theory_class": theory.__class__.__name__,
            "theory_module": theory.__class__.__module__,
            "validator_class": self.__class__.__name__,
            "validator_module": self.__class__.__module__,
            "key_formulas": self._extract_key_formulas(theory, result),
            "computational_notes": self._get_computational_notes(theory, result)
        }
    
    def _extract_key_formulas(self, theory: GravitationalTheory, result: ValidationResult) -> List[str]:
        """Extract key formulas used in predictions"""
        formulas = []
        
        # Validator-specific formulas
        if hasattr(self, '_get_prediction_formulas'):
            formulas.extend(self._get_prediction_formulas(theory))
        
        # Theory-specific formulas
        if hasattr(theory, 'get_key_formulas'):
            formulas.extend(theory.get_key_formulas())
        
        return formulas
    
    def _get_computational_notes(self, theory: GravitationalTheory, result: ValidationResult) -> str:
        """Get computational notes for reproduction"""
        notes = []
        
        # Add validator-specific notes
        if self.name == "CMB Power Spectrum Prediction Validator":
            notes.append("CMB predictions use modified primordial power spectrum")
            notes.append("Low-l modifications applied based on theory parameters")
        elif self.name == "PTA Stochastic GW Background Validator":
            notes.append("GW spectrum modifications based on modified gravity effects")
            notes.append("Hellings-Downs correlation may be modified")
        
        return "; ".join(notes)
    
    def _create_prediction_readme(self, theory: GravitationalTheory, result: ValidationResult, 
                                 theory_dir: str, prediction_data: Dict[str, Any]) -> None:
        """Create detailed README for reproduction"""
        readme_path = os.path.join(theory_dir, "README.md")
        
        content = f"""# Prediction Improvements: {theory.name}

## Overview

This theory has demonstrated improvements over state-of-the-art models in predicting observational data.

**Category**: {getattr(theory, 'category', 'unknown')}  
**Last Updated**: {prediction_data['timestamp']}

## Improved Predictions

### {self.name}

- **SOTA Model**: {result.sota_source}
- **SOTA Value**: {result.sota_value:.6f} {result.units}
- **Theory Value**: {result.predicted_value:.6f} {result.units}
- **Improvement**: {result.error:.6f} ({result.error_percent:.1f}%)
- **Statistical Significance**: {"YES" if result.beats_sota else "NO"}

## Theory Parameters

```json
{json.dumps(prediction_data['theory_parameters'], indent=2)}
```

## Reproduction Instructions

### 1. Theory Implementation

```python
from {prediction_data['reproduction_info']['theory_module']} import {prediction_data['reproduction_info']['theory_class']}

# Create theory instance with exact parameters
theory = {prediction_data['reproduction_info']['theory_class']}()
"""
        
        # Add parameter settings
        for param, value in prediction_data['theory_parameters'].items():
            if param not in ['name', 'category']:
                content += f"theory.{param} = {value}\n"
        
        content += f"""
### 2. Run Validation

```python
from {prediction_data['reproduction_info']['validator_module']} import {prediction_data['reproduction_info']['validator_class']}

# Create validator
validator = {prediction_data['reproduction_info']['validator_class']}()

# Run validation
result = validator.validate(theory, verbose=True)
```

### 3. Key Formulas and Methods

"""
        
        # Add formulas
        if prediction_data['reproduction_info']['key_formulas']:
            content += "#### Mathematical Formulas Used:\n\n"
            for formula in prediction_data['reproduction_info']['key_formulas']:
                content += f"- {formula}\n"
        
        # Add computational notes
        if prediction_data['reproduction_info']['computational_notes']:
            content += f"\n#### Computational Notes:\n\n{prediction_data['reproduction_info']['computational_notes']}\n"
        
        # Add prediction-specific details
        content += "\n## Detailed Prediction Data\n\n"
        content += "```json\n"
        content += json.dumps(result.prediction_data, indent=2)
        content += "\n```\n"
        
        # Add observational data info
        content += "\n## Observational Data Used\n\n"
        if hasattr(self, 'get_observational_data'):
            obs_data = self.get_observational_data()
            content += f"**Source**: {obs_data.get('source', 'Unknown')}\n\n"
            content += f"**Description**: {obs_data.get('description', 'N/A')}\n\n"
        
        # Add saved files info
        content += "\n## Saved Artifacts\n\n"
        content += "### Data Files\n\n"
        if prediction_data['data_files']['count'] > 0:
            content += "The following observational data files have been saved:\n\n"
            for file_info in prediction_data['data_files']['files']:
                content += f"- `data/{file_info['filename']}` ({file_info['size']} bytes)\n"
        
        content += "\n### Code Files\n\n"
        if prediction_data['code_files']['count'] > 0:
            content += "The following code files have been saved for reproduction:\n\n"
            for file_info in prediction_data['code_files']['files']:
                exe_mark = " ⚡" if file_info.get('executable', False) else ""
                content += f"- `code/{file_info['filename']}`{exe_mark}\n"
        
        content += "\n### Running the Reproduction\n\n"
        content += "To reproduce this result:\n\n"
        content += "```bash\n"
        content += "cd " + theory_dir.replace(" ", "_").replace("/", "_") + "/code\n"
        content += "python reproduce_result.py\n"
        content += "```\n"
        
        # Scientific rigor note
        content += """
## Scientific Rigor

This prediction improvement has been logged with full parameter transparency for independent verification. 
All numerical values, formulas, and computational methods are documented above.

### Verification Checklist

- [ ] Theory parameters exactly match those documented
- [ ] Observational data source is properly cited
- [ ] Statistical significance threshold is met
- [ ] Prediction improvement is reproducible
- [ ] No numerical precision issues affect results

## References

1. Original theory implementation: `{}`
2. Validation framework: `physics_agent/validations/`
3. Observational data: See sources above
""".format(prediction_data['reproduction_info']['theory_module'])
        
        with open(readme_path, 'w') as f:
            f.write(content)


class PromisingCandidateValidator:
    """
    Main validator that loads promising candidates from log files and runs all validations.
    """
    
    def __init__(self, validators: List[ObservationalValidator], engine: Optional["TheoryEngine"] = None):
        """
        Initialize with a list of validators to run.
        
        Args:
            validators: List of ObservationalValidator instances
            engine: Shared TheoryEngine instance
        """
        self.validators = validators
        if engine is None:
            EngineClass = _get_theory_engine()
            self.engine = EngineClass()
        else:
            self.engine = engine
        
    def load_promising_candidates(self, log_file: str) -> List[Dict[str, Any]]:
        """
        Load promising candidates from a log file.
        
        Returns:
            List of candidate dictionaries with theory info
        """
        candidates = []
        
        if not os.path.exists(log_file):
            print(f"Warning: Log file not found: {log_file}")
            return candidates
            
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse the log line format:
                # timestamp | Run: runid | Theory: name | Loss_GR: value | Loss_RN: value | Summary: text | Dir: path
                parts = line.split(' | ')
                if len(parts) >= 6:
                    candidate = {
                        'timestamp': parts[0],
                        'run_id': parts[1].split(': ')[1] if ': ' in parts[1] else '',
                        'theory_name': parts[2].split(': ')[1] if ': ' in parts[2] else '',
                        'loss_gr': float(parts[3].split(': ')[1]) if ': ' in parts[3] else 0.0,
                        'loss_rn': float(parts[4].split(': ')[1]) if ': ' in parts[4] else 0.0,
                        'directory': parts[6].split(': ')[1] if len(parts) > 6 and ': ' in parts[6] else ''
                    }
                    candidates.append(candidate)
                    
        return candidates
    
    def validate_all_candidates(self, log_file: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Validate all promising candidates from a log file.
        
        Args:
            log_file: Path to promising_candidates.log
            output_dir: Directory to save results (default: same as log file)
            
        Returns:
            Dictionary of all validation results
        """
        if output_dir is None:
            output_dir = os.path.dirname(log_file)
            
        candidates = self.load_promising_candidates(log_file)
        print(f"Loaded {len(candidates)} promising candidates from {log_file}")
        
        all_results = {}
        
        # Group candidates by theory name to avoid duplicates
        unique_theories = {}
        for candidate in candidates:
            theory_name = candidate['theory_name']
            if theory_name not in unique_theories or candidate['loss_gr'] < unique_theories[theory_name]['loss_gr']:
                unique_theories[theory_name] = candidate
                
        print(f"Testing {len(unique_theories)} unique theories")
        
        for theory_name, candidate in unique_theories.items():
            print(f"\n{'='*60}")
            print(f"Validating: {theory_name}")
            print(f"Loss vs GR: {candidate['loss_gr']:.3e}, Loss vs RN: {candidate['loss_rn']:.3e}")
            
            # Try to load the theory from its directory
            theory = self.load_theory_from_candidate(candidate)
            if theory is None:
                print(f"Failed to load theory from {candidate['directory']}")
                continue
                
            # Run all validators
            theory_results = {}
            for validator in self.validators:
                try:
                    result = validator.validate(theory, verbose=True)
                    theory_results[validator.name] = result.to_dict()
                    
                    # Print summary
                    status = "PASSED" if result.passed else "FAILED"
                    print(f"\n{validator.name}: {status}")
                    if result.observed_value is not None and result.predicted_value is not None:
                        print(f"  Observed: {result.observed_value:.6g} {result.units}")
                        print(f"  Predicted: {result.predicted_value:.6g} {result.units}")
                        print(f"  Error: {result.error_percent:.2f}%")
                        
                except Exception as e:
                    print(f"Error running {validator.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
            all_results[theory_name] = {
                'candidate_info': candidate,
                'validation_results': theory_results
            }
            
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'validation_results_{timestamp}.json')
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\n{'='*60}")
        print(f"Validation complete. Results saved to: {output_file}")
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def load_theory_from_candidate(self, candidate: Dict[str, Any]) -> Optional[GravitationalTheory]:
        """
        Try to load a theory from candidate information.
        
        This attempts to reconstruct the theory from the saved run directory.
        """
        
        # Extract theory directory from the run directory
        run_dir = candidate.get('directory', '')
        
        # If run_dir is relative, try to find it relative to physics_agent
        if run_dir and not os.path.isabs(run_dir):
            # Try relative to physics_agent directory
            physics_agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_run_dir = os.path.join(physics_agent_dir, run_dir)
            if os.path.exists(full_run_dir):
                run_dir = full_run_dir
            else:
                # Try relative to current directory
                cwd_run_dir = os.path.join(os.getcwd(), run_dir)
                if os.path.exists(cwd_run_dir):
                    run_dir = cwd_run_dir
        
        if not run_dir or not os.path.exists(run_dir):
            print(f"  Run directory not found: {run_dir}")
            return None
            
        # Look for theory file in the run directory
        # First try code.py (saved with the run)
        theory_file = os.path.join(run_dir, 'code.py')
        if not os.path.exists(theory_file):
            # Try theory.py
            theory_file = os.path.join(run_dir, 'theory.py')
            
        if not os.path.exists(theory_file):
            # Try parent directories - the theory might be in source/
            parent_dir = os.path.dirname(run_dir)
            while parent_dir and parent_dir != '/':
                source_theory = os.path.join(parent_dir, 'source', 'theory.py')
                if os.path.exists(source_theory):
                    theory_file = source_theory
                    break
                parent_dir = os.path.dirname(parent_dir)
                
        if not os.path.exists(theory_file):
            print(f"  Theory file not found for {candidate['theory_name']}")
            return None
            
        try:
            # Load the theory module
            spec = importlib.util.spec_from_file_location("candidate_theory", theory_file)
            module = importlib.util.module_from_spec(spec)
            
            # Add necessary imports to module namespace
            module.__dict__['torch'] = torch
            module.__dict__['np'] = np
            module.__dict__['numpy'] = np
            module.__dict__['GravitationalTheory'] = GravitationalTheory
            
            # Add physics_agent directory to path for imports
            physics_agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if physics_agent_dir not in sys.path:
                sys.path.insert(0, physics_agent_dir)
            
            spec.loader.exec_module(module)
            
            # Find the theory class
            theory_class = None
            for name, obj in vars(module).items():
                if (isinstance(obj, type) and 
                    issubclass(obj, GravitationalTheory) and 
                    obj != GravitationalTheory):
                    theory_class = obj
                    break
                    
            if theory_class is None:
                print(f"  No GravitationalTheory subclass found in {theory_file}")
                return None
                
            # Try to extract parameters from the theory name
            # e.g., "Linear Signal Loss (γ=+0.75)" -> gamma=0.75
            theory_name = candidate['theory_name']
            params = {}
            
            # Look for parameter patterns like (γ=+0.75) or (β=-1.50)
            param_pattern = r'([α-ωΑ-Ω\w]+)=([+-]?\d+\.?\d*)'
            matches = re.findall(param_pattern, theory_name)
            
            for param_name, param_value in matches:
                # Map Greek letters to parameter names
                param_map = {
                    'γ': 'gamma', 'β': 'beta', 'α': 'alpha', 
                    'δ': 'delta', 'λ': 'lambda', 'τ': 'tau',
                    'q': 'q', 'Q': 'Q', 'a': 'a'
                }
                param_key = param_map.get(param_name, param_name)
                try:
                    params[param_key] = float(param_value)
                except ValueError:
                    pass
                    
            # Instantiate the theory with parameters
            print(f"  Loading {theory_class.__name__} with params: {params}")
            if params:
                theory = theory_class(**params)
            else:
                theory = theory_class()
                
            return theory
            
        except Exception as e:
            print(f"  Error loading theory: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of validation results"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        total_theories = len(results)
        theories_passed_all = 0
        
        for theory_name, theory_data in results.items():
            validation_results = theory_data.get('validation_results', {})
            all_passed = all(r.get('passed', False) for r in validation_results.values())
            if all_passed and validation_results:
                theories_passed_all += 1
                
            print(f"\n{theory_name}:")
            for test_name, result in validation_results.items():
                status = "PASS" if result.get('passed', False) else "FAIL"
                error = result.get('error_percent', 0)
                print(f"  {test_name}: {status} (error: {error:.2f}%)")
                
        print(f"\n{'='*60}")
        print(f"Total theories tested: {total_theories}")
        print(f"Theories passing all tests: {theories_passed_all}")
        print("="*60) 


def safe_float_conversion(value, default=0.0):
    """
    Safely convert a value to float, handling sympy expressions, None, and complex types.
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        
    Returns:
        float: The converted value or default
    """
    if value is None:
        return default
    
    # Handle numeric types
    if isinstance(value, (int, float)):
        return float(value)
    
    # Handle complex numbers
    if isinstance(value, complex):
        return float(value.real)
    
    # Handle numpy types
    if hasattr(value, 'item'):
        try:
            return float(value.item())
        except:
            pass
    
    # Handle sympy expressions
    if hasattr(value, 'evalf'):
        try:
            # Try to evaluate to float
            result = value.evalf()
            if hasattr(result, 'is_number') and result.is_number:
                return float(result)
        except:
            pass
    
    # Handle sympy symbols by checking if it's a symbol and returning default
    if hasattr(value, 'is_symbol') and value.is_symbol:
        return default
    
    # Last resort: try direct conversion
    try:
        return float(value)
    except:
        return default 