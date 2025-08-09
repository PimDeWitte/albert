# Additional Validations for Default Theories

This directory contains additional validation tests specific to the default theory collection.
These validators extend the core validation suite with specialized observational tests.

## Available Validators

- **cassini_ppn_validator.py**: Tests the PPN γ parameter against Cassini spacecraft measurements
- **pulsar_timing_validation.py**: PSR B1913+16 periastron advance (to be updated)
- **mercury_perihelion_validation.py**: Extended Mercury tests (to be updated)
- **pulsar_anomaly_validation.py**: Pulsar anomaly checks (to be updated)
- **unification_symmetry_validation.py**: Theory unification tests (to be updated)

## How It Works

When running `run_validator.py` on theories from the defaults collection, these additional
validators will be automatically discovered and run alongside the core validators.

## Creating Additional Validators

1. Create a new file inheriting from `physics_agent.validations.ObservationalValidator`
2. Implement `get_observational_data()` and `validate()` methods
3. Add to `__init__.py` for automatic discovery

Example:
```python
from physics_agent.validations import ObservationalValidator, ValidationResult

class MyValidator(ObservationalValidator):
    def get_observational_data(self):
        return {...}
    
    def validate(self, theory, verbose=False):
        result = ValidationResult(self.name, theory.name)
        # ... validation logic ...
        return result
```
