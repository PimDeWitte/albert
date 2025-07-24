# Validation Framework

This directory contains all validators for testing gravitational theories against observational data and theoretical constraints.

## Recent Updates

### Dataloader Integration (2025)

Updated key validators to use the centralized dataset loader for consistent data management:

- **CMB Power Spectrum Validator**: Uses `dataloader://planck_cmb_2018`
- **PTA Stochastic GW Validator**: Uses `dataloader://nanograv_15yr` 
- **Primordial GWs Validator**: Uses `dataloader://bicep_keck_2021`

Each validator now:
1. Attempts to load data via the centralized dataloader
2. Falls back to local files or hardcoded values if dataloader fails
3. Gracefully handles missing data (e.g., 404 errors) without failing theories

### TODO: Consolidation with solver_tests

The validation framework shares similar functionality with `solver_tests/` but they are kept separate for now to avoid introducing errors during the transition. Future work should:

1. Identify overlapping test implementations
2. Create shared utilities for common test patterns
3. Ensure validation tests and solver tests remain distinct but avoid code duplication

## Architecture

## 📚 Documentation

- **[ALL_VALIDATORS_DOCUMENTATION.md](./ALL_VALIDATORS_DOCUMENTATION.md)** - Complete documentation of all 17+ validators
- **[SCIENTIFIC_VALIDATION_STANDARDS.md](./SCIENTIFIC_VALIDATION_STANDARDS.md)** - Rigorous standards for scientific acceptance

## Quick Start

```python
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected

# Initialize with high precision
engine = TheoryEngine(dtype=torch.float64)
theory = QuantumCorrected(alpha=0.5)

# Run trajectory
hist, _, _ = engine.run_trajectory(theory, r0, N_STEPS, DTau, ...)

# Run all validations
results = engine.run_all_validations(theory, hist, y0_general)
```

## Validator Categories

### 🔬 Constraint Validators
Test fundamental mathematical and physical constraints:
- **Conservation Validator** - Energy & angular momentum conservation
- **Lagrangian Validator** - Metric-Lagrangian consistency  
- **Metric Properties Validator** - Mathematical structure of spacetime

### 🔭 Observational Validators
Compare predictions with experimental data:
- **Mercury Precession** - Classic weak-field test
- **Light Deflection** - Solar gravitational lensing
- **PPN Parameters** - Comprehensive Solar System tests
- **Photon Sphere** - Black hole shadow predictions
- **Gravitational Waves** - Waveform matching
- **Cosmology** - Large-scale structure tests

### ⚛️ Quantum Validators
Test quantum gravity effects:
- **Atom Interferometry** - Matter wave phase shifts
- **Quantum Clocks** - Ultra-precise time dilation
- **Gravitational Decoherence** - Quantum coherence limits
- **Hawking Radiation** - Black hole thermodynamics

### 📊 Quality Assurance Tools
Ensure numerical accuracy and reproducibility:
- **Precision Tracker** - Real-time error monitoring
- **Uncertainty Quantifier** - Error propagation analysis
- **Reproducibility Framework** - Complete environment logging
- **Scientific Report Generator** - Publication-ready LaTeX reports

## Available Validators

### Additional Validators

#### PpnValidator
Computes PPN parameters and compares to GR values.

#### PhotonSphereValidator
Validates black hole shadow and photon sphere properties.

#### GwValidator
Tests gravitational wave waveform matching.

#### HawkingValidator
Computes Hawking temperature and entropy.

#### CosmologyValidator
Validates cosmological redshift and distance measures.

## Validation Standards

All validators adhere to strict scientific standards:

1. **Precision**: float64 minimum, error bounds < 1e-10
2. **Tolerances**: Theory-specific but typically < 1e-6 for observables
3. **Statistical Rigor**: 95% confidence intervals, proper error propagation
4. **Reproducibility**: Complete parameter and environment tracking

## Running Validations

### By Category
```python
# Only constraint tests
results = engine.run_all_validations(theory, hist, y0_general, 
                                    categories=["constraint"])

# Only observational tests  
results = engine.run_all_validations(theory, hist, y0_general,
                                    categories=["observational"])

# Only quantum tests
results = engine.run_all_validations(theory, hist, y0_general,
                                    categories=["quantum"])
```

### With Quality Tracking
```python
from physics_agent.validations import (
    PrecisionTracker, 
    UncertaintyQuantifier,
    ReproducibilityFramework,
    ScientificReportGenerator
)

# Initialize trackers
precision = PrecisionTracker(dtype=torch.float64)
uncertainty = UncertaintyQuantifier(confidence_level=0.95)
reproducibility = ReproducibilityFramework(run_dir)

# Run with tracking
# ... (see full examples in ALL_VALIDATORS_DOCUMENTATION.md)

# Generate publication-ready report
reporter = ScientificReportGenerator("./reports")
report_path = reporter.generate_full_report(
    theory_name, validation_results, 
    precision_report, uncertainty_report,
    reproducibility_metadata
)
```

## Output Format

All validators return a standardized format:
```python
{
    "loss": float,  # Numerical loss/error metric
    "flags": {
        "overall": "PASS" | "FAIL",
        "specific_test": "PASS" | "FAIL",
        # ... more specific flags
    },
    "details": {
        # Validator-specific details
        "observed": float,
        "predicted": float, 
        "error_percent": float,
        # ...
    }
}
```

## Adding New Validators

See [ALL_VALIDATORS_DOCUMENTATION.md](./ALL_VALIDATORS_DOCUMENTATION.md#adding-new-validators) for detailed instructions.

## Scientific Publications

Results from this validation framework are suitable for publication in:
- Physical Review D/Letters
- Classical and Quantum Gravity
- Living Reviews in Relativity
- MNRAS/ApJ (for cosmological tests)

The framework meets or exceeds peer review standards through:
- Comprehensive error analysis
- Complete reproducibility
- Statistical significance testing
- Publication-ready report generation

## Support

For questions or contributions, please refer to the main project documentation or open an issue. 