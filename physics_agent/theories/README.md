# Multi-Physics Theory Framework

The Albert theory framework now supports multiple fields of physics beyond just gravitational theories. Each field has its own subdirectory containing theories specific to that domain.

## Directory Structure

```
theories/
├── gravitational/              # Gravitational theories
│   ├── defaults/               # Baseline theories (Schwarzschild, Kerr, etc.)
│   ├── aalto_gauge_gravity/
│   ├── alena_tensor/
│   ├── asymptotic_safety/
│   ├── causal_dynamical_triangulations/
│   ├── einstein_asymmetric/
│   ├── einstein_regularised_core/
│   ├── einstein_teleparallel/
│   ├── emergent/
│   ├── entropic_gravity/
│   ├── fractal/
│   ├── gauge_gravity/
│   ├── kerr/
│   ├── kerr_newman/
│   ├── log_corrected/
│   ├── loop_quantum_gravity/
│   ├── newtonian_limit/
│   ├── non_commutative_geometry/
│   ├── phase_transition/
│   ├── post_quantum_gravity/
│   ├── quantum_corrected/
│   ├── spinor_conformal/
│   ├── stochastic_noise/
│   ├── string/
│   ├── surfaceology/
│   ├── twistor_theory/
│   ├── ugm/
│   └── yukawa/
├── thermodynamic/              # Thermodynamic theories
│   └── emergent_spacetime_from_entanglement/
├── fluid_dynamics/             # Fluid dynamics theories
│   └── analog_gravity_superfluid/
├── electromagnetism/           # Electromagnetic theories
│   └── born_infeld_gravity/
├── particle_physics/           # Particle physics theories
│   └── quantum_gravity_with_anomalies/
├── cosmology/                  # Cosmological theories
│   └── (future theories)
├── candidates/                 # Candidate theories under review
│   ├── proposed/
│   ├── new/
│   └── rejected/
└── template/                   # Template for new theories
```

## Creating a Theory for a New Physics Field

Each theory must inherit from `GravitationalTheory` (which will be renamed to `BaseTheory` in future) and implement required methods.

### Example: Thermodynamic Theory

```python
from physics_agent.base_theory import GravitationalTheory
import torch

class EmergentSpacetimeFromEntanglement(GravitationalTheory):
    """Theory where spacetime emerges from quantum entanglement."""
    
    category = "thermodynamic"  # Declare the physics field
    
    def __init__(self, **kwargs):
        # Theory-specific parameters
        self.alpha_ent = kwargs.get('alpha_ent', 1.0)
        
        super().__init__(
            name="Emergent Spacetime from Entanglement",
            **kwargs
        )
    
    def get_metric(self, r, M_param, C_param, G_param, t=None, phi=None):
        """Compute metric with thermodynamic corrections."""
        # Implement metric tensor components
        # Even non-gravitational theories must provide a metric
        # for geodesic integration
        pass
    
    # Field-specific methods
    def compute_entanglement_entropy(self, r, M):
        """Compute entanglement entropy (thermodynamic-specific)."""
        pass
    
    def compute_hawking_temperature(self, M):
        """Hawking temperature with entanglement corrections."""
        pass
```

### Example: Fluid Dynamics Theory

```python
class AnalogGravitySuperfluid(GravitationalTheory):
    """Analog gravity in superfluid systems."""
    
    category = "fluid_dynamics"
    
    def get_metric(self, r, M_param, C_param, G_param, t=None, phi=None):
        """Acoustic metric for phonons in superfluid."""
        # Implement effective metric for sound waves
        pass
    
    # Fluid-specific methods
    def solve_tov_equation(self, rho_c, K, Gamma):
        """Solve TOV equation for fluid star."""
        pass
    
    def fluid_stability_analysis(self, M, r, drho_dr, dP_dr):
        """Analyze fluid stability."""
        pass
```

## Theory Categories

Each physics field can have multiple categories:

- **Gravitational**: `classical`, `quantum`, `modified`, `emergent`
- **Thermodynamic**: `black_hole`, `emergent`, `quantum`
- **Fluid Dynamics**: `relativistic`, `analog`, `quantum`
- **Electromagnetic**: `nonlinear`, `quantum`, `plasma`
- **Particle Physics**: `qft`, `anomalies`, `unified`
- **Cosmology**: `inflation`, `dark_energy`, `modified`

## Required Methods

All theories must implement:

1. `get_metric()` - Returns metric tensor components (even for non-gravitational theories)
2. `__init__()` - Initialize with theory name and parameters

## Optional Field-Specific Methods

### Thermodynamic Theories
- `compute_hawking_temperature(M)`
- `compute_black_hole_entropy(M)`
- `preserves_information()`
- `information_recovery_time(M)`

### Fluid Dynamics Theories
- `solve_tov_equation(rho_c, K, Gamma)`
- `neutron_star_mass_radius(M)`
- `accretion_disk_efficiency(M, a)`
- `fluid_stability_analysis(M, r, drho_dr, dP_dr)`
- `equation_of_state(rho)`

### Electromagnetic Theories
- `maxwell_tensor(r, Q, M)`
- `charged_black_hole_metric(r, M, Q)`
- `electromagnetic_stress_energy(B, E)`
- `maximum_magnetic_field(r, M)`
- `plasma_frequency(n_e, r, M)`

### Particle Physics Theories
- `g_minus_2_correction()`
- `scattering_amplitude_modification(s, t)`
- `beta_function(g, mu)`
- `running_coupling(Q)`
- `unification_scale()`
- `check_renormalizability()`

## Parameter Sweeps

Theories can define parameter sweeps:

```python
class MyTheory(GravitationalTheory):
    sweep = {
        'alpha': [0.1, 0.5, 1.0, 2.0],
        'beta': [0.0, 0.1, 0.2]
    }
    
    preferred_params = {
        'alpha': 1.0,  # Best value from theoretical considerations
        'beta': 0.1
    }
```

## Running Theories

### Command Line

```bash
# Run all theories in a field
albert run --physics-fields thermodynamic

# Run specific theory
albert run --theories emergent_spacetime_from_entanglement

# Test custom theory
albert run --single-theory my_custom_theory.py
```

### Python API

```python
from physics_agent.theories import instantiate_theory

# Load theory by identifier
theory = instantiate_theory('thermodynamic/emergent_spacetime_from_entanglement')

# Or load from custom file
from my_custom_theory import MyCustomTheory
theory = MyCustomTheory()
```

## Cross-Field Theories

Some theories naturally span multiple fields:

- **Black Hole Thermodynamics**: Gravitational + Thermodynamic
- **Magnetohydrodynamics**: Fluid + Electromagnetic  
- **Quantum Gravity**: Gravitational + Quantum + Particle
- **Cosmological Inflation**: Cosmology + Particle + Quantum

These theories should:
1. Choose primary field as their directory location
2. Implement methods from all relevant fields
3. Be tested with validators from all applicable fields

## Best Practices

1. **Metric Required**: Even non-gravitational theories must provide a metric for trajectory integration
2. **Field Declaration**: Always set the `category` attribute to your physics field
3. **Documentation**: Include docstrings explaining the theory's physics
4. **Validation**: Implement field-specific methods that validators expect
5. **Parameter Ranges**: Define sensible parameter sweeps based on physics

## Adding a New Physics Field

To add support for a new field of physics:

1. Create a new subdirectory under `theories/`
2. Create example theories demonstrating the field
3. Add corresponding validators in `validations/your_field/`
4. Update the framework to recognize the new field
5. Document the field-specific methods and requirements