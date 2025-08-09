# Multi-Physics Validation Framework

The Albert validation framework now supports multiple fields of physics, organized into specialized subdirectories. Each field contains validators specific to its domain while maintaining the ability to cross-validate with other fields.

## Directory Structure

```
validations/
├── base_validation.py          # Base classes for all validators
├── gravitational/              # Gravitational physics validators
│   ├── mercury_precession_validator.py
│   ├── light_deflection_validator.py
│   ├── conservation_validator.py
│   ├── metric_properties_validator.py
│   ├── ppn_validator.py
│   ├── photon_sphere_validator.py
│   ├── gw_validator.py
│   ├── hawking_validator.py
│   ├── psr_j0740_validator.py
│   ├── cow_interferometry_validator.py
│   ├── atom_interferometry_validator.py
│   ├── gravitational_decoherence_validator.py
│   ├── quantum_clock_validator.py
│   ├── quantum_lagrangian_grounding_validator.py
│   └── pta_stochastic_gw_validator.py
├── thermodynamic/              # Thermodynamics validators
│   └── black_hole_thermodynamics_validator.py
├── fluid_dynamics/             # Fluid dynamics validators
│   └── relativistic_fluid_validator.py
├── electromagnetism/           # Electromagnetic validators
│   └── electromagnetic_field_validator.py

├── particle_physics/           # Particle physics validators
│   ├── renormalizability_validator.py
│   ├── unification_scale_validator.py
│   ├── qed_precision_validator.py
│   ├── g_minus_2_validator.py
│   └── scattering_amplitude_validator.py
├── cosmology/                  # Cosmology validators
│   ├── cmb_power_spectrum_validator.py
│   ├── primordial_gws_validator.py
│   └── cosmology_validator.py
└── theoretical_physics/        # Theoretical physics validators
    └── lagrangian_validator.py
```

## Physics Fields

### Gravitational Physics
Tests classical and quantum gravitational phenomena:
- Solar system tests (Mercury precession, light deflection)
- Strong field tests (black holes, neutron stars)
- Gravitational waves
- Quantum gravity effects (quantum clocks, decoherence, quantum field theory in curved spacetime)
- Pulsar timing arrays for stochastic gravitational wave backgrounds

### Thermodynamics
Tests thermodynamic consistency and emergent phenomena:
- Black hole thermodynamics (Hawking temperature, entropy)
- Information paradox
- Emergent spacetime from entanglement
- Thermodynamic laws in curved spacetime

### Fluid Dynamics
Tests relativistic fluid behavior:
- Neutron star structure (TOV equation)
- Accretion disk physics
- Fluid stability in strong gravity
- Analog gravity in superfluids

### Electromagnetism
Tests electromagnetic fields in curved spacetime:
- Maxwell equations in curved spacetime
- Charged black holes (Reissner-Nordström, Kerr-Newman)
- Plasma physics in strong gravity
- Electromagnetic stress-energy

### Particle Physics
Tests quantum field theory and high-energy phenomena:
- Anomalous magnetic moments (g-2)
- Scattering amplitudes
- Renormalizability
- Unification scales

### Cosmology
Tests cosmological predictions:
- CMB power spectrum
- Primordial gravitational waves
- Dark energy
- Inflation



## Usage

### Running Field-Specific Validators

```python
from physics_agent.validations import get_validators_by_field

# Get all gravitational validators
grav_validators = get_validators_by_field('gravitational')

# Get all thermodynamic validators
thermo_validators = get_validators_by_field('thermodynamic')

# Test a theory with field-specific validators
for validator_class in thermo_validators:
    validator = validator_class()
    result = validator.validate(theory)
```

### Cross-Field Validation

Many theories span multiple fields. The framework supports cross-field validation:

```python
# A theory implementing both gravity and thermodynamics
theory = EmergentSpacetimeFromEntanglement()

# Test with validators from multiple fields
from physics_agent.validations import (
    BlackHoleThermodynamicsValidator,  # Thermodynamic
    PhotonSphereValidator,              # Gravitational
    ElectromagneticFieldValidator       # Electromagnetic
)

validators = [
    BlackHoleThermodynamicsValidator(),
    PhotonSphereValidator(),
    ElectromagneticFieldValidator()
]

for validator in validators:
    result = validator.validate(theory)
```

## Creating New Validators

To add a validator for a new physics field:

1. Create a subdirectory for the field if it doesn't exist
2. Create your validator class inheriting from `ObservationalValidator`
3. Implement the `validate()` method
4. Add field-specific tests relevant to that domain

Example for a new condensed matter validator:

```python
# validations/condensed_matter/superconductivity_validator.py
from physics_agent.validations.base_validation import ObservationalValidator

class SuperconductivityValidator(ObservationalValidator):
    """Tests theories that predict superconducting phenomena."""
    
    def validate(self, theory, **kwargs):
        # Test Meissner effect
        # Test Cooper pair formation
        # Test BCS gap equation
        # etc.
```

## Field Interactions

The multi-field structure enables testing unified theories that span multiple domains:

- **Gravitational + Thermodynamic**: Black hole thermodynamics, emergent gravity
- **Gravitational + Electromagnetic**: Charged black holes, magnetospheres
- **Gravitational + Fluid**: Neutron stars, accretion disks
- **Particle + Cosmology**: Early universe physics, dark matter
- **Quantum + Gravitational**: Quantum gravity, decoherence

The framework automatically selects appropriate validators based on the theory's declared field and category.