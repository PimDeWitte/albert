# 🌌 Albert: Physics at The Speed of AI

## A Timely Agent for Gravitational Theory Research

Albert is an advanced physics engine and evaluation framework that tests gravitational theories against experimental data and theoretical constraints. It provides comprehensive analytical validation, numerical trajectory integration, and beautiful interactive visualizations for understanding how different theories of gravity behave.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/albertai/albert.git
cd albert

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run comprehensive evaluation on all theories
albert run

# Include candidate theories from the candidates/ folder
albert run --candidates

# Evaluate only candidate theories
albert run --candidates-only

# Filter theories by name
albert run --theory-filter "quantum"

# Run with longer trajectories for more detailed analysis
albert run --max-steps 10000

# Run pre-flight tests to ensure solver correctness
albert run --test
```

## 📊 What Albert Does

Albert evaluates gravitational theories through:

### 1. **Analytical Validation Tests**
- Mercury perihelion precession
- Light deflection by the Sun
- Photon sphere radius
- Parameterized Post-Newtonian (PPN) parameters
- Gravitational wave propagation
- Neutron star constraints (PSR J0740)
- Quantum effects (g-2 anomaly)
- Scattering amplitudes

### 2. **Trajectory-Based Tests**
- Kerr baseline comparison (rotating black holes)
- Schwarzschild comparison (non-rotating black holes)
- Circular orbit stability
- ISCO (Innermost Stable Circular Orbit) analysis
- Conserved quantity preservation
- Precession accuracy

### 3. **Interactive Visualizations**
- 3D trajectory viewers with WebGPU rendering
- Real-time theory comparison dashboards
- Analytical vs numerical test results
- Leaderboard rankings with sortable metrics
- Parameter sweep visualizations

## 🏆 Theory Evaluation & Ranking

The evaluation system produces two key rankings:

1. **Analytical Score**: Based on how well theories match experimental observations
2. **Combined Score**: Weighted combination of analytical (70%) and trajectory (30%) tests

Results are presented in an interactive HTML report with:
- Comprehensive test summaries
- Individual theory performance cards
- Trajectory visualizations
- Loss metrics and comparisons
- Detailed breakdowns of each test

## 🔬 Candidate Theory System

Albert supports a unique candidate theory workflow:

### Directory Structure
```
physics_agent/theories/candidates/
├── proposed/     # Theories awaiting review
├── new/          # Recently discovered theories
└── rejected/     # Theories that didn't pass validation
```

### Working with Candidates

```bash
# Include proposed candidates in evaluation
albert run --candidates

# Test all candidate statuses
albert run --candidates --candidates-status all

# Test only new candidates
albert run --candidates-only --candidates-status new
```

Candidates that score in the top 10 can be submitted via pull request for community review and potential promotion to the main theory collection.

## 🎯 Advanced Features

### Self-Discovery Mode
```bash
albert discover
```
Automatically generates and tests new theoretical variations using AI-guided exploration.

```bash
# Guide discovery toward specific physics
albert discover --initial "explore quantum corrections to the metric"

# Start from an existing theory
albert discover --from-theory theories/quantum_corrected
```

The self-discovery system:
- Uses AI to generate novel gravitational theories
- Evaluates them against experimental data
- Automatically promotes promising candidates
- Provides PR instructions for community review

### Parameter Sweeps
For theories with tunable parameters, use advanced mode:
```bash
# Enable parameter sweeps
albert run-advanced --enable-sweeps

# Sweep only specific parameters
albert run-advanced --sweep-only gamma

# Control sweep parallelization
albert run-advanced --enable-sweeps --sweep-workers 8
```

### Theory Validation
```bash
albert validate path/to/theory.py
```
Validates a single theory file for correctness and compatibility.

### Environment Testing
```bash
albert test
```
Runs comprehensive solver and environment tests to ensure numerical accuracy.

### Longer Trajectories
For more detailed analysis, increase the integration steps:
```bash
albert run --max-steps 50000
```

### Parallel Processing
Control computational resources:
```bash
albert run --max-parallel-workers 8
```

## 📈 Understanding the Results

After running evaluation, you'll find:

1. **HTML Report** (`runs/comprehensive_test_*/comprehensive_theory_validation_*.html`)
   - Interactive dashboards
   - Theory comparison charts
   - Detailed test breakdowns

2. **Trajectory Viewers** (`runs/comprehensive_test_*/trajectories/`)
   - Individual 3D visualizations for each theory
   - Side-by-side comparisons with baselines

3. **JSON Data** (`runs/comprehensive_test_*/theory_validation_*.json`)
   - Raw numerical results
   - Complete test metrics

## 🔧 Architecture Overview

### Core Components

- **evaluation.py**: Main evaluation engine orchestrating all tests
- **geodesic_integrator.py**: Numerical integration solvers for different spacetimes
- **validations/**: Analytical test implementations
- **theories/**: Collection of gravitational theories
- **ui/**: WebGPU-based 3D visualization components

### Solver Hierarchy

Albert uses specialized solvers optimized for different scenarios:
- 4D solvers for spherically symmetric spacetimes (faster)
- 6D solvers for general spacetimes
- Quantum-corrected solvers for theories with quantum effects
- Conserved quantity tracking for numerical stability

## 🌟 Key Innovations

1. **Theory as Code**: Each gravitational theory is implemented as Python code with clear physics
2. **Comprehensive Validation**: Both analytical and numerical tests ensure physical accuracy
3. **Beautiful Visualizations**: WebGPU-powered 3D renderers for intuitive understanding
4. **Candidate System**: Democratic process for discovering and validating new theories
5. **High Precision**: Careful numerical methods preserve conservation laws

## 📚 Documentation

- [Solver Architecture](docs/solvers/index.html) - Deep dive into the numerical methods
- [Validation Pipeline](docs/validators.html) - How theories are tested
- [Self-Discovery](docs/self_discovery.html) - AI-guided theory exploration
- [Scoring System](docs/scoring.html) - How theories are ranked

## 🤝 Contributing

We welcome contributions! Whether it's:
- New gravitational theories
- Additional validation tests
- Visualization improvements
- Documentation enhancements

See the [contribution guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Citations

If you use Albert in your research, please cite:
```bibtex
@software{albert2024,
  title = {Albert: Physics at The Speed of AI},
  author = {Albert AI Team},
  year = {2024},
  url = {https://github.com/albertai/albert}
}
```

## 🚧 Current Status

Albert is under active development. Current focus areas:
- Expanding the theory collection
- Adding more experimental constraints
- Improving visualization performance
- Enhancing the self-discovery system

---

*Built for Einstein's legacy - pursuing perfection in gravitational physics*