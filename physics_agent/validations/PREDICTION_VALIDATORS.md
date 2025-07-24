# Prediction Validators Documentation

## Overview

Prediction validators are a special class of validators that test gravitational theories against state-of-the-art benchmarks using real datasets from the internet. Unlike constraint and observational validators that test known physics, prediction validators evaluate whether theories can make novel predictions that beat current best models.

**Key Requirements**:
1. **Real Data Only**: Prediction validators require actual observational data. They will skip validation if data cannot be downloaded or is not available locally.
2. **Finalists Only**: Prediction validators only run on theories that have successfully completed their trajectories and passed initial validations.

**Important**: Mock or synthetic data is never used for actual validation to ensure scientific integrity.

## Currently Active Prediction Validators

Only **2 prediction validators** are currently active in Phase 3:

1. **CMB Power Spectrum Validator** ✅
2. **Primordial GWs Validator** ✅

The PTA Stochastic GW Validator is implemented but not included in standard runs.

## Workflow

1. **Theory Evaluation**: All theories run through standard trajectory simulations and validations
2. **Finalist Selection**: Only theories that complete successfully (not moved to `fail/` directory) are considered finalists
3. **Batch Prediction**: After all theories complete, prediction validators run on all finalists
4. **Consolidated Report**: Results are saved in a single `predictions/` folder in the main run directory

## Purpose

These validators address a key challenge in theoretical physics: **Can modified theories of gravity make testable predictions that outperform standard models?**

By comparing against:
- Current state-of-the-art (SOTA) models
- Real observational datasets
- Statistical benchmarks from the literature

We can identify theories that:
1. Explain known anomalies better than standard physics
2. Make novel, testable predictions for future experiments
3. Provide quantitative improvements over existing models

## Data Requirements

Before running prediction validators, ensure you have:

1. **For CMB Validator**: Internet connection to download Planck 2018 data from ESA
   - Data will be cached after first download
   - If download fails, validation will be skipped

2. **For Primordial GWs Validator**: BICEP/Keck constraints are built-in
   - No external data download required

3. **For PTA Validator (inactive)**: NANOGrav data file (`v1p1_all_dict.json`)
   - Only needed if manually running this validator
   - Place in one of these locations:
     - `physics_agent/validations/.cache/`
     - `physics_agent/validations/`
     - `physics_agent/data/`
   - Template will be created showing expected format

## Active Prediction Validators

### 1. CMB Power Spectrum Validator (`CMBPowerSpectrumValidator`) ✅

**Purpose**: Tests theories against Planck 2018 CMB angular power spectrum data, focusing on large-scale (low-l) anomalies.

**Dataset**: 
- Source: Planck 2018 TT Power Spectrum
- URL: ESA Planck Legacy Archive
- Focus: Low multipoles (l=2-30) where anomalies exist

**SOTA Benchmark**: Standard ΛCDM model chi-squared fit

**Known Anomalies**:
- Quadrupole deficit (~20% power reduction at l=2)
- Low power at l=3-5
- Hemispherical asymmetry
- Alignment of low multipoles

**Success Criteria**: 
- Theory chi² must improve by Δχ² > 2 over ΛCDM (meaningful improvement)
- Reasonable physical parameters

**Novel Predictions**: Theories with IR modifications (e.g., stochastic loss) may naturally explain low-l deficit without fine-tuning.

### 2. Primordial GWs Validator (`PrimordialGWsValidator`) ✅

**Purpose**: Tests theories' predictions for primordial gravitational waves from inflation.

**Dataset**:
- Source: BICEP/Keck 2023 constraints
- Measurement: Upper limit on tensor-to-scalar ratio
- Built into validator (no download needed)

**SOTA Benchmark**: Single-field slow-roll inflation

**Observations**:
- r < 0.032 (95% upper limit)
- Consistency relation: n_t = -r/8 (for single-field models)

**Success Criteria**:
- r must be positive and below upper limit
- Physical tensor tilt n_t
- Consistency with inflationary models

**Novel Predictions**: Modified gravity theories may predict different r-n_t relations or allow higher r values while satisfying constraints.

### 3. PTA Stochastic GW Validator (`PTAStochasticGWValidator`) ❌ INACTIVE

**Note**: This validator is implemented but not included in standard runs.

**Purpose**: Tests theories against NANOGrav pulsar timing array observations of the stochastic gravitational wave background.

**Dataset**:
- Source: NANOGrav 15-year Data Release
- Measurement: GW strain amplitude at nanohertz frequencies
- Significance: ~4 sigma detection

**SOTA Benchmark**: Supermassive Black Hole Binary (SMBHB) model

**Observations**:
- Amplitude: 2.4e-15 ± 0.6e-15 at f=1/year
- Spectral index: -0.67 ± 0.15 (consistent with -2/3 for binaries)
- Hellings-Downs correlation detected

**Success Criteria**:
- Theory log-likelihood must improve by ΔlnL > 0.1 over SMBHB (any meaningful improvement)
- Physical amplitude and spectral index

**Novel Predictions**: Modified gravity theories may predict different GW spectra or modified Hellings-Downs correlations.

## Results Structure

During a theory run, prediction results are saved in a single consolidated location:

```
runs/run_20240118_123456/
├── Theory_A/              # Individual theory results
├── Theory_B/              
├── fail/                  # Failed theories
└── predictions/           # Consolidated prediction results
    ├── predictions_report.json    # Full detailed report
    └── summary.txt               # Human-readable summary
```

### predictions_report.json Structure

```json
{
    "run_timestamp": "run_20240118_123456",
    "total_finalists": 5,
    "validators_used": ["CMB Power Spectrum Prediction Validator", "PTA Stochastic GW Background Validator"],
    "results": {
        "CMB Power Spectrum Prediction Validator": {
            "sota_benchmark": {
                "value": 53.08,
                "source": "Standard ΛCDM model",
                "units": "chi²/dof"
            },
            "theories": [
                {
                    "theory": "Stochastic Loss (γ=0.50, σ=1.00e-02)",
                    "category": "quantum",
                    "beats_sota": true,
                    "improvement": 8.39,
                    "theory_value": 44.69,
                    "sota_value": 53.08,
                    "units": "chi²/dof"
                }
            ],
            "summary": {
                "total_theories": 5,
                "beating_sota": 2,
                "percentage_beating_sota": 40.0
            }
        }
    }
}
```

## Implementation Notes

### Theory Reconstruction

To enable prediction validation on finalists, theories save their reconstruction information:

```json
{
    "name": "Stochastic Loss (γ=0.50, σ=1.00e-02)",
    "class_name": "StochasticLoss",
    "module_name": "physics_agent.theories.stochastic_loss.theory",
    "parameters": {
        "gamma": 0.5,
        "sigma": 0.01
    }
}
```

This allows the prediction runner to reconstruct theory instances after the main simulation completes.

## Usage

### Running Predictions

```bash
# Predictions run automatically on finalists after main simulation
python -m physics_agent.theory_engine_core

# Disable all validations (including predictions)
python -m physics_agent.theory_engine_core --disable-validation

# Check results
cat runs/latest/predictions/summary.txt
```

### Interpreting Results

The summary.txt file provides a quick overview:

```
Prediction Validation Report
============================================================
Run: run_20240118_123456
Finalists tested: 5
Total predictions: 5

CMB Power Spectrum Prediction Validator
------------------------------------------------------------
SOTA: Standard ΛCDM model (53.08 chi²/dof)
Theories beating SOTA: 2/5 (40.0%)

Top performers:
  1. ✓ Stochastic Loss (γ=0.50, σ=1.00e-02) (quantum)
     Improvement: +8.39 chi²/dof
     Theory: 44.69 vs SOTA: 53.08
  2. ✓ Quantum Corrected (α=0.10) (quantum)
     Improvement: +5.12 chi²/dof
     Theory: 47.96 vs SOTA: 53.08
  3. ✗ Modified Gravity (classical)
     Improvement: -2.34 chi²/dof
     Theory: 55.42 vs SOTA: 53.08
```

## Scientific Impact

Theories that consistently beat SOTA across multiple datasets:
1. Warrant detailed investigation
2. Should be tested with higher precision
3. May guide experimental design
4. Could reveal new physics

The prediction validator framework provides quantitative evidence for theory selection beyond just matching known results.

## Future Extensions

Additional prediction validators can be added for:

1. **LIGO/Virgo GW Catalog**
   - Test: Modified dispersion relations
   - SOTA: GR waveforms
   
2. **JWST Galaxy Formation**
   - Test: Early universe modifications
   - SOTA: ΛCDM structure formation
   
3. **Dark Energy Survey**
   - Test: Modified gravity on cosmological scales
   - SOTA: ΛCDM + dark energy
   
4. **Event Horizon Telescope**
   - Test: Black hole shadow predictions
   - SOTA: Kerr metric

5. **DESI BAO Measurements**
   - Test: Modified expansion history
   - SOTA: ΛCDM cosmology 