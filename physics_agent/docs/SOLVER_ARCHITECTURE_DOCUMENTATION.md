# Gravitational Theory Solver Architecture & Testing Framework

## Executive Summary

The physics engine implements a sophisticated solver hierarchy to simulate particle trajectories in curved spacetime. The core insight is that **geodesic equations for different particle types require specialized solvers** to maintain numerical precision and physical accuracy. Simple geodesic integration is lossy, especially for charged particles, massless particles, and quantum systems.

## Solver Architecture

### Core Solver Hierarchy

```mermaid
graph TB
    subgraph "TheoryEngine Core"
        TE[TheoryEngine<br/>Master Controller]
        TE --> |"Selects Solver"| SM[Solver Selection<br/>Based on Theory & Particle]
    end

    subgraph "Solver Types"
        SM --> |"Symmetric Spacetime"| SYM[4D State Space Solvers]
        SM --> |"General Spacetime"| GEN[6D State Space Solvers]
        SM --> |"Quantum Theory"| QUANT[Quantum Path Integrator]
        
        SYM --> CSQ[ConservedQuantityGeodesicSolver<br/>"Uses E, Lz conservation"]
        SYM --> CSCQ[ConservedQuantityChargedGeodesicSolver<br/>"4D + electromagnetic forces"]
        SYM --> PGS[PhotonGeodesicSolver<br/>"Null geodesics, impact parameter"]
        
        GEN --> GRS[GeneralRelativisticGeodesicSolver<br/>"Full 6D phase space"]
        GEN --> CPS[ChargedParticleGeodesicSolver<br/>"6D + Lorentz force"]
        GEN --> UGM[UnifiedGravityModelGeodesicSolver<br/>"Gauge fields + quantum corrections"]
        
        QUANT --> QPI[QuantumPathIntegrator<br/>"Path integral formulation"]
        QPI --> |"Finds stationary path"| CSQ
    end

    subgraph "Solver Selection Logic"
        SM --> D1{Theory has<br/>conserved quantities?}
        D1 -->|Yes| D2{Particle type?}
        D1 -->|No| D3{Particle type?}
        
        D2 -->|Massive Neutral| CSQ
        D2 -->|Charged| CSCQ
        D2 -->|Massless| PGS
        
        D3 -->|Massive Neutral| GRS
        D3 -->|Charged| CPS
        D3 -->|UGM Theory| UGM
        
        SM --> D4{Quantum theory?}
        D4 -->|Yes| QPI
    end
```

## Solver Descriptions

### 4D State Space Solvers (Symmetric Spacetimes)

#### ConservedQuantityGeodesicSolver
- **Use Case**: Stationary, axisymmetric spacetimes (Schwarzschild, Kerr, etc.)
- **State Vector**: `[t, r, φ, dr/dτ]` (4 dimensions)
- **Key Feature**: Exploits conserved energy E and angular momentum Lz
- **Implementation**: `geodesic_integrator.py:133-384`
- **Efficiency**: ~3-5x faster than 6D solvers due to reduced dimensionality

#### ConservedQuantityChargedGeodesicSolver
- **Use Case**: Charged particles in symmetric spacetimes
- **State Vector**: `[t, r, φ, dr/dτ]` + electromagnetic forces
- **Key Feature**: Adds Lorentz force while maintaining 4D efficiency
- **Implementation**: `geodesic_integrator.py:1003-1037`

#### PhotonGeodesicSolver
- **Use Case**: Massless particles (photons, gravitons)
- **State Vector**: `[t, r, φ, dr/dλ]` (λ = affine parameter)
- **Key Feature**: Uses impact parameter b = L/E for null geodesics
- **Implementation**: `geodesic_integrator.py:788-877`

### 6D State Space Solvers (General Spacetimes)

#### GeneralRelativisticGeodesicSolver
- **Use Case**: Non-symmetric spacetimes, no conserved quantities
- **State Vector**: `[t, r, φ, u^t, u^r, u^φ]` (full 4-velocity)
- **Key Feature**: Computes Christoffel symbols on-the-fly
- **Implementation**: `geodesic_integrator.py:428-727`

#### ChargedParticleGeodesicSolver
- **Use Case**: Charged particles in general spacetimes
- **State Vector**: 6D + electromagnetic 4-force
- **Key Feature**: Full Lorentz force calculation
- **Implementation**: `geodesic_integrator.py:730-787`

#### UnifiedGravityModelGeodesicSolver
- **Use Case**: UGM theories with gauge fields
- **State Vector**: 6D + gauge field contributions
- **Key Feature**: Includes H_a^nu gauge fields and quantum corrections
- **Implementation**: `geodesic_integrator.py:880-999`

### Quantum Solvers

#### QuantumPathIntegrator
- **Use Case**: Quantum theories requiring path integral formulation
- **Methods**: Monte Carlo sampling, WKB approximation, stationary phase
- **Key Feature**: Uses classical solver to find stationary path, adds quantum fluctuations
- **Implementation**: `quantum_path_integrator.py:36-796`

## Testing Framework

### Test Architecture Flow

```mermaid
graph LR
    subgraph "Test Suite (evaluation.py)"
        TM[Test Manager]
        TM --> AT[Analytical Tests]
        TM --> ST[Solver Tests]
        TM --> TT[Trajectory Tests]
    end

    subgraph "Analytical Validators"
        AT --> MPV[Mercury Precession]
        AT --> LDV[Light Deflection]
        AT --> PSV[Photon Sphere]
        AT --> PPV[PPN Parameters]
        AT --> CIV[COW Interferometry]
        AT --> GWV[Gravitational Waves]
        AT --> PJV[PSR J0740]
    end

    subgraph "Solver-Based Tests"
        ST --> COT[Circular Orbit Period]
        ST --> CMB[CMB Power Spectrum]
        ST --> PGW[Primordial GWs]
        ST --> QGS[Quantum Geodesic Sim]
    end

    subgraph "Trajectory Integration Tests"
        TT --> TVK[Trajectory vs Kerr<br/>"1000-step integration"]
        TVK --> LC[Loss Calculator]
        LC --> MSE[MSE Loss]
        LC --> FFT[FFT Loss]
        LC --> EPL[Endpoint Loss]
        LC --> COS[Cosine Similarity]
    end

    subgraph "Baseline Comparison"
        TVK --> KB[Kerr Baseline]
        TVK --> SB[Schwarzschild Baseline]
        TVK --> KNB[Kerr-Newman Baseline]
    end
```

### Test Types

#### 1. Analytical Validators
These tests use analytical approximations to validate theory predictions:
- **Mercury Precession**: Weak-field approximation for perihelion advance
- **Light Deflection**: PPN γ parameter calculation
- **Photon Sphere**: Effective potential extremum
- **PPN Parameters**: Metric expansion for γ and β
- **COW Interferometry**: Metric gradient calculation
- **Gravitational Waves**: Post-Newtonian waveforms
- **PSR J0740**: Shapiro time delay

#### 2. Solver-Based Tests
These tests run actual trajectory integrations:
- **Trajectory vs Kerr**: 1000-step integration with loss calculation
- **Circular Orbit Period**: Tests conservation laws
- **CMB Power Spectrum**: Cosmological perturbation tests
- **Quantum Geodesic Sim**: Quantum path integral validation

### Loss Calculation Against Baselines

The system computes multiple loss metrics to quantify deviation from General Relativity:

```python
# From theory_engine_core.py:2790-2799
loss = engine.loss_calculator.compute_trajectory_loss(
    theory_trajectory, baseline_hist, loss_type
)
```

#### Loss Types:
1. **trajectory_mse**: Mean squared error of full trajectory
2. **fft**: Frequency domain comparison
3. **endpoint_mse**: Final position deviation
4. **cosine**: Angular similarity of trajectory shapes

## Visualization Pipeline

```mermaid
graph TD
    subgraph "TheoryVisualizer (theory_visualizer.py)"
        TV[TheoryVisualizer]
        TV --> GCP[generate_comparison_plot<br/>"Single particle 3D plot"]
        TV --> GAP[generate_all_particles_comparison<br/>"All particles overlay"]
        TV --> GMPG[generate_multi_particle_grid<br/>"2x2 particle grid"]
        TV --> GUMP[generate_unified_multi_particle_plot<br/>"Unified 3D view"]
    end

    subgraph "Plot Elements"
        GCP --> EH[Event Horizon<br/>"Black sphere"]
        GCP --> TR[Trajectories<br/>"Colored paths"]
        GCP --> BL[Baselines<br/>"Dashed lines"]
        GCP --> QU[Quantum Uncertainty<br/>"Cloud visualization"]
        GCP --> IB[Info Box<br/>"Theory parameters"]
    end

    subgraph "Particle Visualization"
        TR --> |electron| BC[Blue Color]
        TR --> |photon| GC[Gold Color]
        TR --> |proton| RC[Red Color]
        TR --> |neutrino| GNC[Green Color]
    end
```

## Key Implementation Files

### Core Engine
- `theory_engine_core.py:670-1390` - Main trajectory runner with solver selection
- `theory_engine_core.py:2385-3108` - Theory evaluation and loss computation

### Solvers
- `geodesic_integrator.py` - All classical geodesic solvers
- `quantum_path_integrator.py` - Quantum trajectory calculation
- `unified_trajectory_calculator.py` - Unified interface for all solvers

### Testing
- `evaluation.py` - Main test suite combining analytical and solver tests
- `test_geodesic_validator_comparison.py` - Schwarzschild baseline comparisons
- `loss_calculator.py` - Loss computation utilities

### Visualization
- `theory_visualizer.py` - All plotting functions
- Generates publication-quality 3D trajectory plots with:
  - Event horizon visualization
  - Multi-particle comparisons
  - Quantum uncertainty clouds
  - Validation status indicators

## Why Specialized Solvers?

1. **Numerical Precision**: Different particle types require different numerical treatments
2. **Physical Accuracy**: Charged particles need electromagnetic forces, massless particles need null geodesics
3. **Computational Efficiency**: 4D solvers are 3-5x faster for symmetric spacetimes
4. **Quantum Effects**: Path integral formulation for quantum theories
5. **Validation**: Each solver type can be validated against known analytical solutions

## Performance Considerations

- **Caching**: Trajectories are cached to avoid recomputation
- **Parallel Processing**: Multiple particles computed simultaneously
- **Early Stopping**: Trajectories stop at horizon crossings
- **Adaptive Integration**: Step sizes adjusted for numerical stability

This architecture ensures that each theory is tested with the most appropriate numerical methods, providing both accuracy and efficiency in the search for novel gravitational theories.