"""
Validation module for testing gravitational theories against observational data.

This is the core validation framework used by the physics_agent system.
Theories can also provide additional custom validations in their 
additional_validations/ subdirectory.
"""

from .base_validation import (
    ValidationResult,
    ObservationalValidator,
    PromisingCandidateValidator,
    PredictionValidator
)

from .mercury_precession_validator import MercuryPrecessionValidator
from .light_deflection_validator import LightDeflectionValidator
from .conservation_validator import ConservationValidator
from .metric_properties_validator import MetricPropertiesValidator

# Quantum-native validators
from .cow_interferometry_validator import COWInterferometryValidator
from .atom_interferometry_validator import AtomInterferometryValidator
from .gravitational_decoherence_validator import GravitationalDecoherenceValidator
from .quantum_clock_validator import QuantumClockValidator
from .quantum_lagrangian_grounding_validator import QuantumLagrangianGroundingValidator

# Unification validators
from .renormalizability_validator import RenormalizabilityValidator
from .unification_scale_validator import UnificationScaleValidator

# Prediction validators (compare against state-of-the-art)
from .cmb_power_spectrum_validator import CMBPowerSpectrumValidator
from .pta_stochastic_gw_validator import PTAStochasticGWValidator
from .primordial_gws_validator import PrimordialGWsValidator
from .qed_precision_validator import QEDPrecisionValidator
# <reason>chain: Removed unimplemented validators</reason>
# from .future_detectors_validator import FutureDetectorsValidator
# from .novel_signatures_validator import NovelSignaturesValidator

from .ppn_validator import PpnValidator
from .photon_sphere_validator import PhotonSphereValidator
from .gw_validator import GwValidator
from .hawking_validator import HawkingValidator
from .cosmology_validator import CosmologyValidator
from .precision_tracker import PrecisionTracker
from .uncertainty_quantifier import UncertaintyQuantifier
from .reproducibility_framework import ReproducibilityFramework
from .scientific_report_generator import ScientificReportGenerator
from .lagrangian_validator import LagrangianValidator
from .psr_j0740_validator import PsrJ0740Validator

__all__ = [
    'ValidationResult',
    'ObservationalValidator',
    'PromisingCandidateValidator', 
    'PredictionValidator',
    'MercuryPrecessionValidator',
    'LightDeflectionValidator',
    'ConservationValidator',
    'MetricPropertiesValidator',
    'COWInterferometryValidator',
    'AtomInterferometryValidator',
    'GravitationalDecoherenceValidator',
    'QuantumClockValidator',
    'QuantumLagrangianGroundingValidator',
    'RenormalizabilityValidator',
    'UnificationScaleValidator',
    'CMBPowerSpectrumValidator',
    'PTAStochasticGWValidator',
    'PrimordialGWsValidator',
    # 'FutureDetectorsValidator',  # Removed - not implemented
    # 'NovelSignaturesValidator',  # Removed - not implemented
    'PpnValidator',
    'PhotonSphereValidator',
    'GwValidator',
    'HawkingValidator',
    'CosmologyValidator',
    'PrecisionTracker',
    'UncertaintyQuantifier',
    'ReproducibilityFramework',
    'ScientificReportGenerator',
    'LagrangianValidator',
    'PsrJ0740Validator',
    'QEDPrecisionValidator'
] 