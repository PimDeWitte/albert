# 🌌 Albert: Physics at The Speed of AI

<div align="center">
  <img src="docs/sketch.png" alt="Albert Einstein Sketch" width="200"/>
  
 An experimental differential physics engine written in PyTorch, dedicated to testable physics research
  
  [![GitHub](https://img.shields.io/badge/GitHub-View%20Code-blue?logo=github)](https://github.com/pimdewitte/albert)
  [![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord)](https://discord.gg/xdybbSk5)
  [![Status](https://img.shields.io/badge/Status-Research%20Preview-yellow)]()
  
</div>

---

## Setup

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
albert run --max-steps 100000
albert run --theories schwarzschild kerr
albert run --theories schwarzschild --black-hole-preset stellar_mass
albert run --category quantum

# Configure Albert (API keys, etc.)
albert setup

# Discover new theories automatically -- this is highly experimental and mostly just exists for fun - we don't have the observational data today to prove any of them ;) 
albert discover --initial "unified field theory"

# Discover variations of an existing theory
albert discover --from-theory theories/quantum_corrected

# Optional: Make albert available globally
sudo ln -s $(pwd)/albert /usr/local/bin/albert
# Now you can use 'albert' from anywhere
```

---

## CLI

### `albert run` - Run Theory Simulations
```bash
# Run all theories with default settings
albert run

# Run specific theories
albert run --theories schwarzschild kerr        # Multiple theories
albert run --category quantum                    # Run category
albert run --candidates                          # Include candidates

# Black hole configurations
albert run --black-hole-preset stellar_mass      # 10 solar masses
albert run --black-hole-preset primordial_mini   # Default: quantum scale
albert run --black-hole-preset sagittarius_a_star # Galactic center
albert run --black-hole-preset laboratory_micro  # Extreme quantum regime
albert run --black-hole-preset intermediate_mass # Globular clusters

# Particle simulations
albert run --particles electron photon neutrino  # Multi-particle
albert run --particles proton                    # Specific particle

# Performance options
albert run --device cuda --dtype float32         # GPU acceleration
albert run --device cpu --dtype float64          # Max precision
albert run --max-steps 1000000                   # Million-step trajectories
albert run --no-cache                            # Force recomputation (bypasses the tensor cache)
albert run --clear-cache                         # Clear cache and exit

# Advanced options
albert run --validators-only                     # Skip trajectory computation
albert run --max-parallel-workers 16             # Parallel processing
albert run --test                                # Run pre-flight tests
albert run --enable-sweeps                       # Enable parameter sweeps
albert run --sweep-only gamma                    # Sweep only specific parameter
albert run --experimental                        # Enable experimental quantum features
albert run --verbose                             # Enable verbose logging
albert run --final                               # High-quality publication mode
albert run --early-stop                          # Enable early stopping
albert run --quantum-field-content all           # Configure quantum field content
```

### `albert discover` - an experimental loop to test LLM generated theorems against observational data 
```bash
# Start discovery with default settings
albert discover

# Discovery with initial prompt
albert discover --initial "unified field theory with torsion"

# Improve existing theory
albert discover --from-theory theories/quantum_corrected

# Continuous monitoring mode
albert discover --self-monitor
```
The origination of this project was trying to identify the most likely candidates of Einstein's final bedside notes based on completions by various models. The test framework around it was built to validate generations. And the differential engine can provide feedback data. This loop is obviously highly speculative, experimental and so far hasn't produced anything of meaning :)

### `albert setup` - Configuration
```bash
# Interactive setup wizard
albert setup
```

### Other Commands
```bash
albert validate path/to/theory.py    # Validate specific theory
albert test                         # Run environment tests
albert --help                       # Show all commands
```

---


### Candidate Theory System
```
physics_agent/theories/candidates/
├── proposed/     # Theories awaiting review
├── new/          # Recently discovered theories
└── rejected/     # Theories that didn't pass validation
```

Every theory is referenced against publicly available observational data
![Screenshot 2025-10-14 at 9 49 09 PM](https://github.com/user-attachments/assets/a5504eb3-61f3-42ff-a59d-d17635aea414)
![Screenshot 2025-10-14 at 9 49 28 PM](https://github.com/user-attachments/assets/a1a01fae-4e8e-4634-baf1-6c99d8c8469e)
![Screenshot 2025-10-14 at 9 49 32 PM](https://github.com/user-attachments/assets/f14e2a6a-2c2a-4c05-864d-7e7b400ed587)

---

## Implementing A New Theory

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
            category="quantum"  # or "classical", "emergent", "baseline"
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
albert run --theories "My Theory"
```
---

## 📚 Documentation

- [Introduction & System](docs/introduction.html) - Architecture overview
- [Solvers & Validators](docs/solvers_validators.html) - Testing framework
- [Cache System](docs/cache.html) - Performance optimization
- [API Reference](https://albert.so/documentation.html) - Full documentation

---

## 📊 Architecture Diagrams

### Theory Engine Core Execution Flow

```mermaid
flowchart TD
    START[albert run] --> PARSE[Parse CLI Arguments]
    PARSE --> SETUP[Setup Execution Mode]
    SETUP --> DEVICE[Determine Device & Dtype<br/>GPU/CPU, float32/float64]
    DEVICE --> ENGINE[Create TheoryEngine Instance]
    ENGINE --> LOADER[Load 32 Theories<br/>baseline/classical/quantum/emergent]
    
    LOADER --> FILTER[Apply Filters<br/>--theories, --category, --candidates]
    FILTER --> PHASE0[PHASE 0: Baseline Tests<br/>Schwarzschild & Kerr]
    
    PHASE0 --> PHASE1[PHASE 1: Theory Validation<br/>14 Validators × 32 Theories]
    PHASE1 --> CACHE{Check Cache?}
    
    CACHE -->|Hit<br/>8.6ms| LOAD[Load Tensor]
    CACHE -->|Miss<br/>4+ min| INTEGRATE[Geodesic Integration<br/>RK8/Symplectic/Implicit]
    
    INTEGRATE --> SAVE[Save to Cache]
    SAVE --> VALIDATE
    LOAD --> VALIDATE[Run Validators]
    
    VALIDATE --> PHASE2[PHASE 2: Full Trajectories<br/>Multi-particle simulations]
    PHASE2 --> PHASE3[PHASE 3: Predictions<br/>Novel physics tests]
    PHASE3 --> LEADERBOARD[Generate Results<br/>HTML Report + 3D Viz]
    LEADERBOARD --> END[Complete]
    
    style START fill:#90EE90
    style END fill:#FFB6C1
    style CACHE fill:#FFE4B5
    style INTEGRATE fill:#FF6B6B
    style LOAD fill:#4ECDC4
```

### Self-Discovery Flow

```mermaid
graph TD
    A[🤖 LLM Generation<br/>AI generates novel theories] -->|Python code| B[🔬 Validation<br/>14 comprehensive tests]
    B --> C{Pass Rate?}
    C -->|> 85%| D[🏆 Top Performer<br/>Promoted to candidate]
    C -->|< 85%| E[❌ Rejected<br/>Logged for analysis]
    
    D --> F[📁 Candidate Storage<br/>theories/candidates/new/]
    F --> G[📊 Metadata<br/>• Test results<br/>• Trajectories<br/>• Performance metrics]
    G --> H[🌿 Git Workflow<br/>Create PR]
    H --> I[👥 Community Review<br/>Physics validation]
    
    E --> J[🔄 Feedback Loop<br/>Improve prompts]
    J --> A
```

### Cache Architecture

```mermaid
graph LR
    A[Trajectory Request] --> B{Cache Key<br/>SHA256 Hash}
    B --> C[Parameter Set<br/>• Theory name<br/>• Black hole preset<br/>• Particle type<br/>• Integration steps<br/>• Numerical params]
    
    C --> D{Cache Lookup}
    D -->|Found| E[Load PyTorch Tensor<br/>8.6ms]
    D -->|Not Found| F[Compute Trajectory<br/>4+ minutes]
    
    F --> G[Save to Cache<br/>~30MB for 1M steps]
    G --> H[Return Result]
    E --> H
    
    style E fill:#4ECDC4
    style F fill:#FF6B6B
```

---

## Prerequisites

- **Python**: 3.9+
- **GPU**: NVIDIA (CUDA), Apple Silicon (MPS), or CPU fallback
- **API Key**: xAI/Grok (recommended) or experimental support for other providers

## Configuration

Albert uses AI to generate new theories. Get your API key:

### Primary Provider (Recommended)
- **xAI/Grok**: https://x.ai/api

### Experimental Providers
- OpenAI, Anthropic, Google Gemini (limited support)

Run the setup wizard:
```bash
albert setup
```

---

<div align="center">
  <i>"I want to know God's thoughts. The rest are details."</i><br>
  — Albert Einstein
</div>
