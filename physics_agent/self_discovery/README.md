# Self Discovery System

The self-discovery system for automated gravitational theory exploration, now located under `physics_agent/self_discovery/`.

## Architecture

The system is split into two main components:

1. **Theory Engine Core** (`physics_agent/theory_engine_core.py`)
   - Core simulation and evaluation functionality
   - Trajectory computation
   - Loss calculations
   - Visualization generation
   - Can be used independently for theory testing

2. **Self Discovery** (`physics_agent/self_discovery/self_discovery.py`)
   - Automated theory generation via AI APIs
   - Theory loading and management
   - Iterative refinement loops
   - Uses TheoryEngine for all simulations

## Customizable Prompt Template

The self-discovery system now uses an external prompt template that can be easily customized:

- **`prompt_template.txt`**: The main prompt template file
- **`PROMPT_TEMPLATE_README.md`**: Detailed guide on customizing the prompt

This allows you to:
- Guide theory generation toward specific physics domains
- Add custom requirements and constraints
- Explore different theoretical frameworks
- Focus on particular validation criteria

To customize theory generation, simply edit `prompt_template.txt` before running the self-discovery system. See `PROMPT_TEMPLATE_README.md` for detailed instructions and examples.

## Usage

Run self-discovery from the project root:

```bash
python physics_agent/self_discovery/self_discovery.py [options]
```

Or use the convenience script:

```bash
./run_physics_agent.sh self_discovery.py [options]
```

## Common Commands

### Test all baseline theories
```bash
python physics_agent/self_discovery/self_discovery.py
```

### Test a custom theory directory
```bash
python physics_agent/self_discovery/self_discovery.py --test-theories physics_agent/theories/my_theory
```

### Run with different loss types
```bash
python physics_agent/self_discovery/self_discovery.py --loss-type ricci
python physics_agent/self_discovery/self_discovery.py --multi-loss
```

### Enable self-monitoring and early stopping
```bash
python physics_agent/self_discovery/self_discovery.py --self-monitor --early-stop --steps 100000
```

### Generate new theories via AI

Default mode (runs baselines, generates new theory):
```bash
python physics_agent/self_discovery/self_discovery.py --self-discover
```

With custom initial prompt:
```bash
python physics_agent/self_discovery/self_discovery.py --self-discover --initial-prompt "Explore torsion-based modifications"
```

Focus on improving a specific theory:
```bash
python physics_agent/self_discovery/self_discovery.py --self-discover --theory theories/quantum_corrected --initial-prompt "Add holographic corrections"
```

Use prompt from file:
```bash
python physics_agent/self_discovery/self_discovery.py --self-discover --initial-prompt prompts/einstein_notes.txt
```

## Features

- **Early Stopping**: Automatically backs off from poorly performing theories (credit: Ben Geist)
- **Multiple Loss Types**: FFT, Ricci tensor, endpoint MSE, and more
- **Intelligent Caching**: Speeds up repeated runs
- **Progress Monitoring**: Visual snapshots during simulation
- **Dynamic Theory Loading**: Automatically finds and loads theories
- **AI-Powered Discovery**: Generate new theories via LLM APIs
- **Candidate System**: Promising theories automatically promoted to candidates/
- **PR Integration**: Helper scripts for submitting candidates to GitHub

## Directory Structure

```
physics_agent/
├── theory_engine_core.py    # Core simulation engine
├── self_discovery/
│   ├── self_discovery.py    # Self-discovery system
│   ├── cache/              # Cached trajectories
│   ├── runs/               # Run outputs
│   └── README.md           # This file
└── theories/               # Theory implementations
```

## Theory Requirements

Theories must be in `physics_agent/theories/` with:
- Required: `theory.py` containing a class inheriting from `GravitationalTheory`
- Optional: `validations/`, `papers/`, `results/`, etc.

See `physics_agent/theories/README.md` for details on creating theories.

## Candidate Workflow

The self-discovery system now includes an automated candidate promotion workflow:

1. **Discovery**: When `--self-discover` is used, the system:
   - Runs baseline theories
   - Generates a new theory via LLM
   - Evaluates the theory's performance

2. **Promotion**: If the theory scores in the top 10:
   - Automatically creates `candidates/c_{timestamp}_{hash}/`
   - Copies all run data to the candidate directory
   - Shows PR submission instructions

3. **Submission**: Two ways to submit candidates:
   ```bash
   # Automated submission
   python submit_candidate.py c_20240115_abc12345
   
   # Manual submission
   git add physics_agent/theories/candidates/
   git commit -m "Add promising candidate c_20240115_abc12345"
   git push origin your-branch
   ```

4. **Testing Candidates**: Run all theories including candidates:
   ```bash
   python -m physics_agent.theory_engine_core --candidates
   ```

## Self-Discovery Modes

- **Default**: Runs only baselines, keeps runtime open for agent
- **With --candidates**: Also loads and evaluates existing candidates
- **With --theory**: Focuses on improving a specific theory

## Using TheoryEngine Directly

For direct theory testing without self-discovery:

```python
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.my_theory.theory import MyTheory

# Initialize engine
engine = TheoryEngine(device='cuda', dtype=torch.float64)

# Create theory instance
theory = MyTheory()

# Run simulation
hist, tag, kicks = engine.run_trajectory(
    theory, r0, n_steps, dtau,
    quantum_interval=0, quantum_beta=0.0,
    M=engine.M, c=engine.C_T, G=engine.G_T,
    EPSILON=engine.EPSILON
)

# Generate visualization
engine.generate_comparison_plot(
    theory.name, hist, baseline_results,
    engine.RS.item(), "output.png"
)
``` 