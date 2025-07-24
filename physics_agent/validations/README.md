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

### 🔬 Constraint Validators (Always Run)
Test fundamental mathematical and physical constraints:
- **Conservation Validator** - Energy & angular momentum conservation
- **Metric Properties Validator** - Mathematical structure of spacetime

### 🔭 Observational Validators (Always Run)
Compare predictions with experimental data:
- **Mercury Precession** - Classic weak-field test
- **Light Deflection** - Solar gravitational lensing
- **PPN Parameters** - Comprehensive Solar System tests
- **Photon Sphere** - Black hole shadow predictions
- **Gravitational Waves** - Waveform matching

### ⚛️ Quantum Validators (Run for Quantum Theories)
Test quantum gravity effects:
- **COW Neutron Interferometry** - Matter wave phase shifts in gravity

### 📊 Prediction Validators (Run in Phase 3)
Test novel predictions against state-of-the-art:
- **CMB Power Spectrum** - Tests against Planck 2018 data
- **Primordial GWs** - Tests inflationary predictions

### Quality Assurance Tools
Ensure numerical accuracy and reproducibility:
- **Precision Tracker** - Real-time error monitoring
- **Uncertainty Quantifier** - Error propagation analysis
- **Reproducibility Framework** - Complete environment logging
- **Scientific Report Generator** - Publication-ready LaTeX reports

## Currently Active Validators

The following validators are tested and run in the current version:

### Phase 1 & 2 (Constraint & Observational)
1. **Conservation Validator** - Energy and angular momentum conservation
2. **Metric Properties Validator** - Metric mathematical properties
3. **Mercury Precession Validator** - Perihelion advance test
4. **Light Deflection Validator** - Solar gravitational lensing
5. **PPN Parameters Validator** - Post-Newtonian parameters
6. **Photon Sphere Validator** - Black hole shadow size
7. **GW Waveform Validator** - Gravitational wave templates
8. **COW Interferometry Validator** - Quantum matter wave interference (quantum theories only)

### Phase 3 (Prediction - Run After All Theories Complete)
9. **CMB Power Spectrum Validator** - Tests against Planck 2018 anomalies
10. **Primordial GWs Validator** - Tensor-to-scalar ratio predictions

## Validators Not Currently Active

The following validators are implemented but not included in standard runs:
- Lagrangian Validator (not tested in solver_tests)
- Atom Interferometry Validator
- Gravitational Decoherence Validator  
- Quantum Clock Validator
- Quantum Lagrangian Grounding Validator
- Hawking Radiation Validator
- Cosmology Validator
- PsrJ0740 Validator
- Renormalizability Validator
- Unification Scale Validator
- PTA Stochastic GW Validator
- QED Precision Validator

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

# Only prediction tests (run separately in Phase 3)
results = engine.run_all_validations(theory, hist, y0_general,
                                    categories=["prediction"])
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
- Journal of Cosmology and Astroparticle Physics
- Monthly Notices of the Royal Astronomical Society

## References

See individual validator files for specific experimental references and datasets used. 