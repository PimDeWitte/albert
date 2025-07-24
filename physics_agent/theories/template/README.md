# Theory Template

This is a template directory for creating new gravitational theories.

## How to Create a New Theory

1. **Copy this template directory**:
   ```bash
   cp -r theories/template theories/my_new_theory
   ```

2. **Create your theory implementation** in `source/theory.py`:
   ```python
   from base_theory import GravitationalTheory, Tensor
   import torch
   
   class MyNewTheory(GravitationalTheory):
       category = "classical"  # or "quantum", "unified"
       cacheable = True
       
       def __init__(self):
           super().__init__("My New Theory")
       
       def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
           # Your metric implementation
           rs = 2 * G_param * M_param / C_param**2
           # Return g_tt, g_rr, g_pp, g_tp
           return -m, 1/m, r**2, torch.zeros_like(r)
   ```

3. **Add theoretical baselines** in `baselines/`:
   - Mathematical derivations
   - Physical motivations
   - References to papers

4. **Run your theory**:
   ```bash
   ./theories/my_new_theory/run.sh
   ```

## Directory Structure

- **source/**: Your theory implementation (required)
- **baselines/**: Theoretical foundations and derivations
- **validations/**: Custom validation tests (optional, defaults are run automatically)
- **papers/**: Related research and publications
- **results/**: Simulation outputs (created automatically)
- **self_discovery/**: AI-generated variations (created if using self-discovery mode)
- **runs/**: Individual run results (created automatically)

## What Happens When You Run

1. Your theory is automatically compared against:
   - Schwarzschild (GR) - pure gravity baseline
   - Reissner-Nordström - gravity + electromagnetism baseline
   - Any baseline theories in your theory's directory

2. Standard validations are run:
   - PSR B1913+16 pulsar timing
   - Mercury perihelion precession
   - Cassini PPN-γ parameter
   - (more to come)

3. Results are saved in `runs/` with:
   - Trajectory plots
   - Metric component visualizations
   - Loss comparisons
   - Interactive visualizations

## No Programming Required!

You only need to:
1. Write your metric equations in `source/theory.py`
2. Run `./run.sh` from your theory directory

The framework handles everything else automatically.

## Self-Discovery Mode

Let AI explore variations of your theory:

```bash
# Basic self-discovery
./run.sh --self-discover

# With custom exploration direction
./run.sh --self-discover --initial-prompt "Investigate a theory that incorporates quantum corrections to the metric tensor, guided by a Lagrangian that includes higher-order curvature invariants."

# Using different AI provider (requires appropriate API key)
./run.sh --self-discover --api-provider gemini
```

Generated variations will be saved in the `self_discovery/` subdirectory. 