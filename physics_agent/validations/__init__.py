"""
Multi-Physics Validation Framework

This framework supports validation of theories across multiple fields of physics:
- Gravitational physics
- Thermodynamics  
- Fluid dynamics
- Quantum mechanics
- Electromagnetism
- Particle physics
- Cosmology
- Theoretical physics

<reason>chain: Restructured to support multiple physics fields beyond just gravity</reason>
"""

from .base_validation import (
    ValidationResult,
    ObservationalValidator,
    PromisingCandidateValidator,
    PredictionValidator
)

# Import validators by field
# <reason>chain: Organized imports by physics field for clarity</reason>

# Gravitational physics validators
from .gravitational.mercury_precession_validator import MercuryPrecessionValidator
from .gravitational.light_deflection_validator import LightDeflectionValidator
from .gravitational.conservation_validator import ConservationValidator
from .gravitational.metric_properties_validator import MetricPropertiesValidator
from .gravitational.ppn_validator import PpnValidator
from .gravitational.photon_sphere_validator import PhotonSphereValidator
from .gravitational.gw_validator import GwValidator
from .gravitational.hawking_validator import HawkingValidator
from .gravitational.psr_j0740_validator import PsrJ0740Validator
from .gravitational.cow_interferometry_validator import COWInterferometryValidator
from .gravitational.atom_interferometry_validator import AtomInterferometryValidator
from .gravitational.gravitational_decoherence_validator import GravitationalDecoherenceValidator

# Thermodynamic validators
from .thermodynamic.black_hole_thermodynamics_validator import BlackHoleThermodynamicsValidator
from .thermodynamic.hawking_temperature_validator import HawkingTemperatureValidator

# Fluid dynamics validators
from .fluid_dynamics.relativistic_fluid_validator import RelativisticFluidValidator
from .fluid_dynamics.energy_conditions_validator import EnergyConditionsValidator

# Electromagnetic validators
from .electromagnetism.electromagnetic_field_validator import ElectromagneticFieldValidator
from .electromagnetism.charged_black_hole_validator import ChargedBlackHoleValidator

# Quantum gravity validators (part of gravitational physics)
from .gravitational.quantum_clock_validator import QuantumClockValidator
from .gravitational.quantum_lagrangian_grounding_validator import QuantumLagrangianGroundingValidator
from .gravitational.pta_stochastic_gw_validator import PTAStochasticGWValidator

# Particle physics validators
from .particle_physics.renormalizability_validator import RenormalizabilityValidator
from .particle_physics.unification_scale_validator import UnificationScaleValidator
from .particle_physics.qed_precision_validator import QEDPrecisionValidator
from .particle_physics.g_minus_2_validator import GMinus2Validator
from .particle_physics.scattering_amplitude_validator import ScatteringAmplitudeValidator
from .particle_physics.running_couplings_validator import RunningCouplingsValidator

# Cosmology validators
from .cosmology.cmb_power_spectrum_validator import CMBPowerSpectrumValidator
from .cosmology.primordial_gws_validator import PrimordialGWsValidator
from .cosmology.cosmology_validator import CosmologyValidator
from .cosmology.hubble_parameter_validator import HubbleParameterValidator

# Theoretical physics validators
from .theoretical_physics.lagrangian_validator import LagrangianValidator

# Multi-physics validators
from .multi_physics.unified_field_validator import UnifiedFieldValidator
from .multi_physics.cosmological_thermodynamics_validator import CosmologicalThermodynamicsValidator
from .multi_physics.quantum_gravity_effects_validator import QuantumGravityEffectsValidator

# Framework utilities
from .precision_tracker import PrecisionTracker
from .uncertainty_quantifier import UncertaintyQuantifier
from .reproducibility_framework import ReproducibilityFramework
from .scientific_report_generator import ScientificReportGenerator
from .validator_registry import validator_registry
from .validator_performance_tracker import performance_tracker

# <reason>chain: Helper function to get validators by field</reason>
def get_validators_by_field(field: str) -> list:
    """Get all validators for a specific physics field."""
    field_validators = {
        'gravitational': [
            MercuryPrecessionValidator,
            LightDeflectionValidator,
            ConservationValidator,
            MetricPropertiesValidator,
            PpnValidator,
            PhotonSphereValidator,
            GwValidator,
            HawkingValidator,
            PsrJ0740Validator,
            COWInterferometryValidator,
            AtomInterferometryValidator,
            GravitationalDecoherenceValidator,
            # Quantum gravity validators
            QuantumClockValidator,
            QuantumLagrangianGroundingValidator,
            PTAStochasticGWValidator
        ],
        'thermodynamic': [
            BlackHoleThermodynamicsValidator,
            HawkingTemperatureValidator
        ],
        'fluid_dynamics': [
            RelativisticFluidValidator,
            EnergyConditionsValidator
        ],
        'electromagnetism': [
            ElectromagneticFieldValidator,
            ChargedBlackHoleValidator
        ],
        'particle_physics': [
            RenormalizabilityValidator,
            UnificationScaleValidator,
            QEDPrecisionValidator,
            GMinus2Validator,
            ScatteringAmplitudeValidator,
            RunningCouplingsValidator
        ],
        'cosmology': [
            CMBPowerSpectrumValidator,
            PrimordialGWsValidator,
            CosmologyValidator,
            HubbleParameterValidator
        ],
        'theoretical_physics': [
            LagrangianValidator
        ],
        'multi_physics': [
            UnifiedFieldValidator,
            CosmologicalThermodynamicsValidator,
            QuantumGravityEffectsValidator
        ]
    }
    
    return field_validators.get(field, [])

# <reason>chain: Get all available physics fields</reason>
def get_available_fields() -> list:
    """Get list of all available physics fields."""
    return [
        'gravitational',
        'thermodynamic', 
        'fluid_dynamics',
        'electromagnetism',
        'particle_physics',
        'cosmology',
        'theoretical_physics',
        'multi_physics'
    ]

__all__ = [
    # Base classes
    'ValidationResult',
    'ObservationalValidator',
    'PromisingCandidateValidator', 
    'PredictionValidator',
    
    # Gravitational validators
    'MercuryPrecessionValidator',
    'LightDeflectionValidator',
    'ConservationValidator',
    'MetricPropertiesValidator',
    'PpnValidator',
    'PhotonSphereValidator',
    'GwValidator',
    'HawkingValidator',
    'PsrJ0740Validator',
    'COWInterferometryValidator',
    'AtomInterferometryValidator',
    'GravitationalDecoherenceValidator',
    
    # Thermodynamic validators
    'BlackHoleThermodynamicsValidator',
    'HawkingTemperatureValidator',
    
    # Fluid dynamics validators
    'RelativisticFluidValidator',
    'EnergyConditionsValidator',
    
    # Electromagnetic validators
    'ElectromagneticFieldValidator',
    'ChargedBlackHoleValidator',
    
    # Quantum mechanics validators
    'QuantumClockValidator',
    'QuantumLagrangianGroundingValidator',
    'PTAStochasticGWValidator',
    
    # Particle physics validators
    'RenormalizabilityValidator',
    'UnificationScaleValidator',
    'QEDPrecisionValidator',
    'GMinus2Validator',
    'ScatteringAmplitudeValidator',
    'RunningCouplingsValidator',
    
    # Cosmology validators
    'CMBPowerSpectrumValidator',
    'PrimordialGWsValidator',
    'CosmologyValidator',
    'HubbleParameterValidator',
    
    # Theoretical physics validators
    'LagrangianValidator',
    
    # Multi-physics validators
    'UnifiedFieldValidator',
    'CosmologicalThermodynamicsValidator',
    'QuantumGravityEffectsValidator',
    
    # Utilities
    'PrecisionTracker',
    'UncertaintyQuantifier',
    'ReproducibilityFramework',
    'ScientificReportGenerator',
    'validator_registry',
    'performance_tracker',
    
    # Helper functions
    'get_validators_by_field',
    'get_available_fields'
]