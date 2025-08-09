"""
Relativistic Fluid Dynamics Validator

Tests theories that include relativistic fluid dynamics, such as:
- Neutron star structure (TOV equation)
- Accretion disk dynamics
- Relativistic jets
- Fluid instabilities in strong gravity

<reason>chain: Many astrophysical systems involve relativistic fluids in strong gravitational fields</reason>
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from physics_agent.validations.base_validation import ObservationalValidator, ValidationResult
from physics_agent.constants import GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT

class RelativisticFluidValidator(ObservationalValidator):
    """
    Validates theories with relativistic fluid dynamics in curved spacetime.
    
    <reason>chain: Essential for testing theories that modify fluid behavior in strong gravity</reason>
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Relativistic Fluid Dynamics"
        self.description = "Tests relativistic fluid behavior in strong gravitational fields"
        
    def validate(self, theory, **kwargs) -> ValidationResult:
        """
        Validate theory's predictions for relativistic fluids.
        
        <reason>chain: Tests neutron stars, accretion, and fluid stability</reason>
        """
        results = {
            'tov_equation': self._test_tov_equation(theory),
            'neutron_star_mass_radius': self._test_neutron_star_constraints(theory),
            'accretion_efficiency': self._test_accretion_disk(theory),
            'fluid_stability': self._test_fluid_stability(theory),
            'eos_consistency': self._test_equation_of_state(theory)
        }
        
        # Calculate overall pass/fail
        passed = all(r.get('passed', False) for r in results.values())
        
        # Calculate aggregate score
        scores = [r.get('score', 0.0) for r in results.values()]
        overall_score = np.mean(scores) if scores else 0.0
        
        return ValidationResult(
            passed=passed,
            score=overall_score,
            details=results,
            notes=self._generate_notes(results, theory)
        )
    
    def _test_tov_equation(self, theory) -> Dict[str, Any]:
        """Test Tolman-Oppenheimer-Volkoff equation for hydrostatic equilibrium."""
        try:
            if not hasattr(theory, 'solve_tov_equation'):
                return {
                    'passed': True,  # Not required for all theories
                    'score': 0.5,
                    'notes': 'Theory does not implement TOV solver'
                }
            
            # Test with simple polytropic EOS: P = K ρ^Γ
            K = 1e13  # Polytropic constant (SI units)
            Gamma = 2.0  # Polytropic index
            
            # Central density for 1.4 solar mass neutron star
            rho_c = 1e18  # kg/m³
            
            # Solve TOV equation
            M, R, P_profile, rho_profile = theory.solve_tov_equation(rho_c, K, Gamma)
            
            # Check physical constraints
            M_solar = M / SOLAR_MASS
            R_km = R / 1000
            
            # Typical neutron star: M ~ 1.4 M_sun, R ~ 10-15 km
            mass_ok = 0.5 < M_solar < 3.0  # Below maximum NS mass
            radius_ok = 8 < R_km < 20  # Reasonable radius range
            
            # Check pressure goes to zero at surface
            surface_pressure_ok = P_profile[-1] / P_profile[0] < 1e-10
            
            passed = mass_ok and radius_ok and surface_pressure_ok
            
            score = 0.0
            if mass_ok:
                score += 0.4
            if radius_ok:
                score += 0.4
            if surface_pressure_ok:
                score += 0.2
            
            return {
                'passed': passed,
                'score': score,
                'mass_solar': M_solar,
                'radius_km': R_km,
                'surface_pressure_ratio': P_profile[-1] / P_profile[0],
                'notes': f'M={M_solar:.2f}M☉, R={R_km:.1f}km'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_neutron_star_constraints(self, theory) -> Dict[str, Any]:
        """Test against observed neutron star mass-radius relations."""
        try:
            if not hasattr(theory, 'neutron_star_mass_radius'):
                return {
                    'passed': True,
                    'score': 0.5,
                    'notes': 'Theory does not model neutron stars'
                }
            
            # Test against PSR J0740+6620 constraints
            # M = 2.08 ± 0.07 M_sun, R = 12.35 ± 0.75 km
            M_obs = 2.08 * SOLAR_MASS
            R_obs = 12.35e3  # meters
            
            M_pred, R_pred = theory.neutron_star_mass_radius(M_obs)
            
            # Check radius prediction
            R_pred_km = R_pred / 1000
            R_error = abs(R_pred_km - 12.35) / 12.35
            
            # Within 2σ (1.5 km)?
            passed = abs(R_pred_km - 12.35) < 1.5
            
            score = 1.0 - min(R_error, 1.0)
            
            return {
                'passed': passed,
                'score': score,
                'predicted_radius_km': R_pred_km,
                'observed_radius_km': 12.35,
                'error_percent': R_error * 100,
                'notes': f'R={R_pred_km:.1f}km vs observed 12.35±0.75km'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_accretion_disk(self, theory) -> Dict[str, Any]:
        """Test accretion disk physics and efficiency."""
        try:
            if not hasattr(theory, 'accretion_disk_efficiency'):
                return {
                    'passed': True,
                    'score': 0.5,
                    'notes': 'Theory does not model accretion'
                }
            
            # Test Schwarzschild and Kerr black holes
            M = 10 * SOLAR_MASS  # 10 solar mass black hole
            
            # Schwarzschild case (a=0)
            eta_schw = theory.accretion_disk_efficiency(M, a=0.0)
            
            # Kerr case (a=0.998, near extremal)
            eta_kerr = theory.accretion_disk_efficiency(M, a=0.998)
            
            # Standard thin disk efficiencies:
            # Schwarzschild: η ≈ 0.057 (5.7%)
            # Kerr (a=0.998): η ≈ 0.32 (32%)
            
            schw_ok = abs(eta_schw - 0.057) / 0.057 < 0.1  # 10% tolerance
            kerr_ok = abs(eta_kerr - 0.32) / 0.32 < 0.15  # 15% tolerance
            
            # Kerr should be more efficient
            ordering_ok = eta_kerr > eta_schw
            
            passed = schw_ok and kerr_ok and ordering_ok
            
            score = 0.0
            if schw_ok:
                score += 0.4
            if kerr_ok:
                score += 0.4
            if ordering_ok:
                score += 0.2
            
            return {
                'passed': passed,
                'score': score,
                'schwarzschild_efficiency': eta_schw,
                'kerr_efficiency': eta_kerr,
                'expected_schw': 0.057,
                'expected_kerr': 0.32,
                'notes': f'η_Schw={eta_schw:.3f}, η_Kerr={eta_kerr:.3f}'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_fluid_stability(self, theory) -> Dict[str, Any]:
        """Test fluid stability criteria in strong gravity."""
        try:
            if not hasattr(theory, 'fluid_stability_analysis'):
                return {
                    'passed': True,
                    'score': 0.5,
                    'notes': 'Theory does not analyze fluid stability'
                }
            
            # Test Rayleigh-Taylor instability in accretion
            M = 1e6 * SOLAR_MASS  # Supermassive black hole
            r = 10 * (2 * GRAVITATIONAL_CONSTANT * M / SPEED_OF_LIGHT**2)  # 10 Rs
            
            # Density and pressure gradients
            drho_dr = -1e10  # kg/m⁴ (negative = decreasing outward)
            dP_dr = -1e20  # Pa/m (negative = decreasing outward)
            
            stability = theory.fluid_stability_analysis(M, r, drho_dr, dP_dr)
            
            # Check various stability criteria
            passed = True
            score = 0.0
            
            if 'rayleigh_taylor' in stability:
                # Should be stable for normal accretion
                if stability['rayleigh_taylor']['stable']:
                    score += 0.25
                else:
                    passed = False
            
            if 'kelvin_helmholtz' in stability:
                # Check shear stability
                if stability['kelvin_helmholtz']['growth_rate'] < 1e-5:
                    score += 0.25
            
            if 'parker' in stability:
                # Magnetic stability
                if stability['parker']['stable']:
                    score += 0.25
            
            if 'convective' in stability:
                # Schwarzschild criterion
                if stability['convective']['stable']:
                    score += 0.25
            
            return {
                'passed': passed,
                'score': score,
                'stability_criteria': stability,
                'notes': 'Fluid stability analysis complete'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _test_equation_of_state(self, theory) -> Dict[str, Any]:
        """Test consistency of equation of state with observations."""
        try:
            if not hasattr(theory, 'equation_of_state'):
                return {
                    'passed': True,
                    'score': 0.5,
                    'notes': 'Theory does not specify equation of state'
                }
            
            # Test at nuclear density
            rho_nuc = 2.8e17  # kg/m³ (nuclear saturation density)
            
            # Get pressure and sound speed
            P, cs2 = theory.equation_of_state(rho_nuc)
            
            # Causality constraint: cs² < c²
            causal = cs2 < 1.0  # In units where c=1
            
            # Stability: cs² > 0
            stable = cs2 > 0
            
            # Realistic range: 0 < cs² < 1/3 for most matter
            realistic = 0 < cs2 < 0.33
            
            passed = causal and stable
            
            score = 0.0
            if causal:
                score += 0.4
            if stable:
                score += 0.4
            if realistic:
                score += 0.2
            
            return {
                'passed': passed,
                'score': score,
                'sound_speed_squared': cs2,
                'pressure': P,
                'density': rho_nuc,
                'causal': causal,
                'stable': stable,
                'notes': f'cs²/c² = {cs2:.3f}'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _generate_notes(self, results: Dict[str, Any], theory) -> str:
        """Generate summary notes."""
        notes = []
        
        for test_name, result in results.items():
            if 'passed' in result:
                symbol = "✓" if result['passed'] else "✗"
                
                if test_name == 'neutron_star_mass_radius' and 'error_percent' in result:
                    notes.append(f"{symbol} NS radius: {result['error_percent']:.1f}% error")
                elif test_name == 'accretion_efficiency' and 'schwarzschild_efficiency' in result:
                    notes.append(f"{symbol} Accretion: η={result['schwarzschild_efficiency']:.3f}")
                elif test_name == 'fluid_stability':
                    notes.append(f"{symbol} Fluid stability")
                elif test_name == 'equation_of_state' and 'sound_speed_squared' in result:
                    notes.append(f"{symbol} EOS: cs²/c²={result['sound_speed_squared']:.2f}")
        
        return "; ".join(notes)


# Import required constants that might be missing
try:
    from physics_agent.constants import SOLAR_MASS
except ImportError:
    SOLAR_MASS = 1.989e30  # kg
