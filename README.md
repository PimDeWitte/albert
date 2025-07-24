# 🌌 Albert: Physics at The Speed of AI

<div align="center">
  <img src="docs/sketch.png" alt="Albert Einstein Sketch" width="200"/>
  
  **A timely agent for gravitational theory research**
  
  [![GitHub](https://img.shields.io/badge/GitHub-View%20Code-blue?logo=github)](https://github.com/pimdewitte/albert)
  [![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord)](https://discord.gg/xdybbSk5)
  [![Status](https://img.shields.io/badge/Status-Research%20Preview-yellow)]()
  
  *One engine to model everything. In code. Rooted in the laws of physics.*
</div>

---

## 🚀 Quick Start

```bash
# One-line installation
curl -fsSL https://raw.githubusercontent.com/PimDeWitte/albert/refs/heads/main/download_cli.sh | bash

# Clone and setup
git clone https://github.com/pimdewitte/albert.git
cd albert
./setup_unified.sh

# Run all theories (standard run)
albert run

# Run with specific options
albert run --steps 1000
albert run --theory-filter "ugm"
albert run --gpu-f32
albert run --enable-sweeps

# Configure Albert (API keys, etc.)
albert setup

# Discover new theories automatically
albert discover --initial "unified field theory"

# Discover variations of an existing theory
albert discover --from theories/einstein_unified/theory.py

# Optional: Make albert available globally
sudo ln -s $(pwd)/albert /usr/local/bin/albert
# Now you can use 'albert' from anywhere
```

---

## 🎯 Command Line Interface

Albert provides a unified CLI with multiple subcommands:

### `albert run` - Run Theory Simulations
```bash
# Run all theories with default settings
albert run

# Run specific theories
albert run --theory-filter "kerr"           # Run Kerr theory
albert run --category ugm                   # Run all UGM theories
albert run --candidates                     # Include candidate theories

# Performance options
albert run --gpu-f32                        # GPU with float32
albert run --cpu-f64                        # CPU with float64
albert run --steps 10000                    # Custom step count
albert run --no-cache                       # Force recomputation

# Parameter sweeps
albert run --enable-sweeps                  # Enable parameter sweeps
albert run --sweep-only gamma               # Sweep only gamma parameter
albert run --sweep-workers 8                # Set parallel workers

# Advanced options
albert run --close-orbit                    # Use 6RS orbit (stronger fields)
albert run --early-stop                     # Enable convergence detection
albert run --experimental                   # Enable quantum kicks
albert run --verbose                        # Detailed logging
```

### `albert discover` - AI Theory Discovery
```bash
# Start discovery with default settings
albert discover

# Discovery with initial prompt
albert discover --initial "unified field theory with torsion"

# Improve existing theory
albert discover --from theories/einstein_unified/theory.py

# Continuous monitoring mode
albert discover --self-monitor
```

### `albert setup` - Configuration
```bash
# Interactive setup wizard
albert setup
```

### Other Commands
```bash
albert validate theories/my_theory/theory.py  # Validate specific theory
albert --help                                 # Show all commands
```

---

## 📊 Complete Validator Reference

Albert uses 10 core validators to test gravitational theories against experimental data:

### Constraint Validators (Must Pass)
- **Conservation**: Energy/angular momentum < 1e-12 drift
- **Metric Properties**: Signature, smoothness, asymptotic limits

### Observational Validators  
- **Mercury Precession**: 42.98 ± 0.04 arcsec/century
- **Light Deflection**: 1.7509 ± 0.0003 arcsec  
- **PPN Parameters**: γ = 1.000 ± 0.002, β = 1.000 ± 0.003
- **Photon Sphere**: Black hole shadow size (EHT)
- **Gravitational Waves**: LIGO/Virgo waveform match > 0.95

### Quantum Validators
- **COW Interferometry**: Neutron phase shift tests (quantum theories only)

### Prediction Validators (Phase 3)
- **CMB Power Spectrum**: Planck 2018 anomalies (χ²/dof)
- **Primordial GWs**: Tensor-to-scalar ratio (r < 0.032)

---

## 🧬 Self-Discovery System

Albert uses AI to generate and test new gravitational theories automatically:

### How It Works
1. **AI Generation**: LLM generates novel theory code based on prompts
2. **Validation**: Theories tested against 10 core validators
3. **Ranking**: Top performers promoted to candidate status
4. **Storage**: Candidates saved with full results and metadata
5. **Review**: Community can review and submit via pull requests

### Discovery Modes
```bash
# Basic discovery
albert discover

# Guided discovery with physics hints
albert discover --initial "incorporate holographic principle"

# Theory improvement
albert discover --from theories/quantum_corrected/theory.py

# Continuous discovery with monitoring
albert discover --self-monitor
```

---

## ⚡ Performance Features

### PyTorch Tensor Caching
- **First run**: Full computation (minutes)
- **Cached runs**: Near-instant (milliseconds)
- **Speedup**: Up to 29,000x for large trajectories
- **Storage**: ~30MB per trajectory

### Parallel Computing
- Parameter sweeps run in parallel
- Auto-detects optimal worker count
- GPU support for float32 operations
- MPS support for Apple Silicon

### Optimization Settings
```bash
# Maximum performance
albert run --gpu-f32 --enable-sweeps --sweep-workers 16

# Maximum precision
albert run --cpu-f64 --steps 1000000

# Quick testing
albert run --steps 100 --theory-filter "test"
```

---

## 🚀 Creating Your Own Theory

1. **Create theory file**: `theories/my_theory/theory.py`
2. **Define your metric**:
```python
from physics_agent.base_theory import GravitationalTheory, Tensor
import torch

class MyTheory(GravitationalTheory):
    def __init__(self):
        super().__init__(
            name="My Theory",
            description="Novel gravitational theory",
            category="quantum"  # or "classical", "ugm"
        )
    
    def get_metric(self, r, M, C, G):
        # Define your g_μν components
        g_tt = -(1 - 2*G*M/(C**2 * r))
        g_rr = 1/(1 - 2*G*M/(C**2 * r))
        # ... define all components
        return Tensor("metric", [...])
```

3. **Run validation**:
```bash
albert run --theory-filter "My Theory"
```

---

## 🌍 The Vision: Open World Model

Albert is building toward a unified physics engine where:
- Every physical law is implemented and validated
- All experimental data is digitized and accessible
- Theories can be tested against all known physics
- Synthetic data generation for games and training

Future extensions will include:
- Fluid dynamics solvers
- Quantum field theory
- Condensed matter physics
- Statistical mechanics
- Plasma physics

---

## 👥 Contributing

### For Physicists
- Add new validators for your field
- Implement experimental datasets
- Verify theoretical predictions
- Contribute new baseline theories

### For Engineers
- Optimize solvers with torch.compile
- Implement GPU kernels
- Add visualization tools
- Improve caching system

### For Everyone
- Test new theories
- Report bugs
- Improve documentation
- Join discussions on Discord

---

## 📚 Documentation

- [Technical Paper](docs/paper.html) - Geodesic solver development
- [Validators](docs/validators.html) - All validation tests explained
- [Self Discovery](docs/self_discovery.html) - AI theory generation
- [API Reference](https://albert.so/documentation.html) - Full documentation

---

## 🙏 Acknowledgments

This project continues Einstein's quest for unification. Special thanks to:
- Partanen & Tulkki (2025) for the UGM framework
- The open-source physics community
- Everyone who believes in open science

---

## 📊 Architecture Diagrams

### Theory Engine Core Execution Flow

```mermaid
flowchart TD
    START[main Entry Point] --> PARSE[Parse CLI Arguments]
    PARSE --> SETUP[Setup Execution Mode]
    SETUP --> DEVICE[Determine Device & Dtype<br/>GPU/CPU, float32/float64]
    DEVICE --> ENGINE[Create TheoryEngine Instance]
    ENGINE --> LOADER[Create TheoryLoader<br/>theories_base_dir=physics_agent/theories]
    
    LOADER --> DISCOVER[Discover Theories<br/>loader.discover_theories]
    DISCOVER --> WALK[Walk Directory Tree<br/>os.walk]
    WALK --> FINDTHEORY{Find theory.py files?}
    
    FINDTHEORY -->|Yes| LOADCLASS[Dynamic Code Loading<br/>importlib.util]
    LOADCLASS --> INSPECT[Inspect Module<br/>Find GravitationalTheory subclasses]
    INSPECT --> STORE[Store Theory Class]
    STORE --> WALK
    
    FINDTHEORY -->|No| NEXTDIR[Continue Directory Walk]
    NEXTDIR --> WALK
    
    WALK -->|Complete| FILTER[Apply Theory Filter]
    FILTER --> PHASE0[PHASE 0: Baseline Simulation<br/>Run Kerr & Kerr-Newman]
    
    PHASE0 --> PHASE1[PHASE 1: Theory Validation]
    PHASE1 --> THEORY_LOOP{For each theory}
    
    THEORY_LOOP --> HASSWEEP{Has parameter sweep?}
    HASSWEEP -->|Yes| SWEEP[Multiprocessing<br/>ProcessPoolExecutor]
    HASSWEEP -->|No| VALIDATE[Run Validators]
    
    VALIDATE --> CONSTRAINTS[Constraint Validators]
    CONSTRAINTS --> OBSERVATIONAL[Observational Validators]
    OBSERVATIONAL --> QUANTUM{Is quantum theory?}
    QUANTUM -->|Yes| QUANTUM_VAL[Quantum Validators]
    
    THEORY_LOOP -->|Complete| PHASE2[PHASE 2: Full Trajectories]
    PHASE2 --> PHASE3[PHASE 3: Predictions]
    PHASE3 --> LEADERBOARD[Generate Leaderboard]
    LEADERBOARD --> END[End]
    
    style START fill:#90EE90
    style END fill:#FFB6C1
    style PHASE0 fill:#FFE4B5
    style PHASE1 fill:#FFE4B5
    style PHASE2 fill:#FFE4B5
    style PHASE3 fill:#FFE4B5
```

### Validation Pipeline

```mermaid
graph TD
    A[🎯 THEORY SPECIFICATION<br/>Define g_μν + parameters] 
    A --> B[🧪 VALIDATION SUITE<br/>10 Core Tests]
    B --> C[⚙️ THEORY ENGINE<br/>Auto-detect symmetries]
    C --> D{🔍 SOLVER SELECTION}
    
    D -->|Symmetric| E[4D Solver<br/>Conserved: E, L_z]
    D -->|General| F[6D Solver<br/>Full phase space]
    D -->|Charged| G[Charged Extension<br/>Lorentz force]
    D -->|Quantum| H[Quantum Path Integral<br/>WKB approximation]
    
    E --> I[📐 INTEGRATION<br/>RK4 with PyTorch]
    F --> I
    G --> I  
    H --> I
    
    I --> J[✅ TEST EXECUTION<br/>10 Validators Run]
    J --> K{📊 RESULTS ANALYSIS}
    
    K -->|Pass| L[✨ NOVEL PREDICTIONS]
    K -->|Fail| M[❌ THEORY REJECTED]
    
    L --> N[🏆 SCORING<br/>χ²/dof • AIC • BIC]
    N --> O[📈 LEADERBOARD]
    
    M --> P[🔄 FEEDBACK]
    P --> A
```

### Self-Discovery Flow

```mermaid
graph TD
    A[🤖 LLM Generation<br/>AI generates novel theories] -->|Python code| B[🔬 Evaluation<br/>Test against baselines]
    B --> C[🏆 Ranking<br/>Top 10 promoted]
    C --> D[📁 Candidate Storage<br/>c_timestamp_hash/]
    D --> E[🌿 Git Workflow<br/>Automated branch]
    E --> F[🔄 Pull Request<br/>Community review]
    
    D --> G[theory.py<br/>README.md<br/>trajectory.pt<br/>losses.json]
```

### Standard Model Support

```mermaid
graph LR
    A[QED Lagrangian] --> B[Precision Tests]
    B --> C[Electron g-2<br/>12 digits precision]
    B --> D[Quantum Clocks<br/>33cm test: 10^-17]
    B --> E[Lamb Shift<br/>1057.845 MHz]
    
    F[Base Theory] --> G[add_qed_corrections]
    G --> H[Matter Coupling]
    H --> I[Unified Theory]
```

### Unified Gauge Model (UGM) Architecture

```mermaid
graph TD
    A[Four U1 Gauge Fields] --> B[H^0_μ: Time U1]
    A --> C[H^1_μ: Radial U1]
    A --> D[H^2_μ: Theta U1]
    A --> E[H^3_μ: Phi U1]
    
    B --> F[Tetrad: e^a_μ = δ^a_μ + g H^a_μ]
    C --> F
    D --> F
    E --> F
    
    F --> G[Metric: g_μν = η_ab e^a_μ e^b_ν]
    G --> H[General Relativity<br/>when all α_a = 1]
```

### Geodesic Solver Architecture

```mermaid
classDiagram
    class GravitationalTheory {
        +get_metric(r, M, C, G)
        +get_lagrangian()
        +validate()
    }
    
    class GeneralGeodesicRK4Solver {
        +6D state space
        +Arbitrary metrics
        +Quantum corrections
    }
    
    class GeodesicRK4Solver {
        +4D optimized
        +Conserved E, L_z
        +Symmetric spacetimes
    }
    
    class UGMGeodesicRK4Solver {
        +Tetrad formalism
        +Gauge fields H^a_μ
        +Loop corrections
    }
    
    GravitationalTheory --> GeneralGeodesicRK4Solver
    GravitationalTheory --> GeodesicRK4Solver
    GravitationalTheory --> UGMGeodesicRK4Solver
```

### Performance Optimization

```mermaid
graph LR
    A[First Run<br/>4+ minutes] --> B[Cache Trajectory<br/>30MB PyTorch tensors]
    B --> C[Second Run<br/>8.6ms]
    
    D[10K steps] -->|1,110x speedup| E[2.75s → 2.5ms]
    F[100K steps] -->|10,674x speedup| G[25.8s → 2.4ms]
    H[1M steps] -->|29,323x speedup| I[4m 12s → 8.6ms]
```

### Future Physics Engine Vision

```mermaid
graph TD
    A[Physics Engine<br/>Base classes for all physics] --> B[Unified Testing<br/>Every experiment digitized]
    B --> C[World Simulation<br/>Game engines meet physics]
    
    A --> D[FluidDynamics<br/>QuantumField<br/>GravitationalTheory]
    B --> E[LIGO data<br/>Quantum interference<br/>Cosmology]
    C --> F[Synthetic data<br/>PyTorch backend<br/>Real-time simulation]
    
    D --> G[WorldModel]
    E --> G
    F --> G
    
    G --> H[universe.simulate<br/>→ games or training!]
```

---

<div align="center">
  <i>"I want to know God's thoughts. The rest are details."</i><br>
  — Albert Einstein
</div> 