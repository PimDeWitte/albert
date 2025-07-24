# Gravitational Theories Directory Structure

This directory contains implementations of various gravitational theories for testing and comparison.

## Directory Structure

Each theory should be in its own directory with the following structure:

### Required Files

- **`theory.py`** - The main theory implementation containing a class that inherits from `GravitationalTheory`

### Optional Directories/Files

- **`validations/`** - (Optional) Custom validation tests specific to this theory
- **`papers/`** - (Optional) Related research papers and documentation
- **`results/`** - (Optional) Simulation results and outputs
- **`baselines/`** - (Optional) Baseline comparisons or reference implementations
- **`source/`** - (Optional) Additional source code or legacy implementations
- **`runs/`** - (Optional) Execution logs and run histories
- **`README.md`** - (Optional) Theory-specific documentation

## Example Structure

```
my_theory/
├── theory.py              # Required: Main theory implementation
├── README.md             # Optional: Theory documentation
├── validations/          # Optional: Custom tests
├── papers/               # Optional: Research papers
├── results/              # Optional: Outputs
└── runs/                 # Optional: Execution logs
```

## Creating a New Theory

1. Create a new directory: `mkdir my_new_theory`
2. Create `theory.py` with a class inheriting from `GravitationalTheory`:

```python
from physics_agent.base_theory import GravitationalTheory

class MyTheory(GravitationalTheory):
    def __init__(self):
        super().__init__("My Theory Name", is_symmetric=True)
    
    def get_metric(self, r, M, c, G):
        # Implement your metric tensor components
        g_tt = ...
        g_rr = ...
        g_pp = ...
        g_tp = ...
        return g_tt, g_rr, g_pp, g_tp
```

3. That's it! The theory is ready to be tested.

## Special Directories

- **`defaults/`** - Contains the framework's default theories and baselines
- **`template/`** - Template structure for new theories

## Note on Self-Discovery

The self-discovery system has been moved to `/self_discovery/` at the project root for better separation of concerns. 