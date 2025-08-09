# Gravitational physics validators
"""
Validators for gravitational physics theories.

These validators test theories against classical and quantum gravitational phenomena:
- Solar system tests (Mercury precession, light deflection)
- Strong field tests (black holes, neutron stars)
- Gravitational waves
- Quantum gravity effects

<reason>chain: Centralized location for all gravitational physics validators</reason>
"""

from .mercury_precession_validator import MercuryPrecessionValidator
from .light_deflection_validator import LightDeflectionValidator
from .conservation_validator import ConservationValidator
from .metric_properties_validator import MetricPropertiesValidator
from .ppn_validator import PpnValidator
from .photon_sphere_validator import PhotonSphereValidator
from .gw_validator import GwValidator
from .hawking_validator import HawkingValidator
from .psr_j0740_validator import PsrJ0740Validator
from .cow_interferometry_validator import COWInterferometryValidator
from .atom_interferometry_validator import AtomInterferometryValidator
from .gravitational_decoherence_validator import GravitationalDecoherenceValidator

# Also import simple/alternative versions if they exist
try:
    from .ppn_validator_simple import PpnValidatorSimple
except ImportError:
    PpnValidatorSimple = None

__all__ = [
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
]

if PpnValidatorSimple is not None:
    __all__.append('PpnValidatorSimple')