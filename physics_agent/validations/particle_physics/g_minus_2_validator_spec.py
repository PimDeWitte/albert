"""
Anomalous Magnetic Moment (g-2) Validator Specification

This validator tests whether a theory correctly predicts quantum corrections
to the magnetic moment of leptons, particularly the muon where we have
the most precise measurements and a significant (4.2σ) discrepancy with
the Standard Model.

Physical Background:
- The magnetic moment of a lepton is μ = g(e/2m)S where g ≈ 2
- QED predicts g = 2 at tree level (Dirac equation)
- Quantum corrections give g = 2(1 + a) where a = (g-2)/2
- For the muon: a_μ^exp = 116592059(22) × 10^-11 (Fermilab 2023)
- SM prediction: a_μ^SM = 116591810(43) × 10^-11
- Discrepancy: Δa_μ = 249(48) × 10^-11 (4.2σ)

Implementation Strategy:
1. Extract electromagnetic and other relevant couplings from theory
2. Compute one-loop vertex corrections
3. Include vacuum polarization contributions
4. Add light-by-light scattering (if theory has appropriate vertices)
5. Compare with experimental value
"""

import numpy as np
from typing import Dict, Optional, Tuple
from ..validations.base_validation import ValidationResult, ObservationalValidator

class GMinus2Validator(ObservationalValidator):
    """
    Validator for anomalous magnetic moment predictions.
    
    Tests a theory's prediction for (g-2)/2 against precision measurements,
    particularly for the muon where there's a significant SM discrepancy.
    """
    
    def __init__(self):
        super().__init__()
        
        # Experimental values (CODATA 2022 + Fermilab Run 1-3)
        self.experimental_values = {
            'electron': {
                'value': 1159652180.73e-12,
                'error': 0.28e-12,
                'source': 'Harvard 2022'
            },
            'muon': {
                'value': 116592059e-11,
                'error': 22e-11,
                'source': 'Fermilab 2023'
            },
            'tau': {
                'value': 0.00117721,  # Less precise
                'error': 0.00000005,
                'source': 'PDG 2022'
            }
        }
        
        # Standard Model predictions
        self.sm_predictions = {
            'electron': {
                'value': 1159652181.606e-12,
                'error': 0.229e-12
            },
            'muon': {
                'value': 116591810e-11,
                'error': 43e-11
            }
        }
    
    def calculate_one_loop_correction(
        self, 
        theory,
        lepton: str = 'muon'
    ) -> Tuple[float, float]:
        """
        Calculate one-loop QED-like corrections from the theory.
        
        This includes:
        1. Vertex correction (Schwinger term): α/(2π)
        2. Vacuum polarization
        3. Light-by-light scattering (if applicable)
        
        Returns:
            (correction, uncertainty) in units of 10^-11 for muon
        """
        
        # Get coupling constants at appropriate scale
        if lepton == 'muon':
            energy_scale = 0.105658  # GeV (muon mass)
        elif lepton == 'electron':
            energy_scale = 0.000511  # GeV (electron mass)
        else:
            energy_scale = 1.777    # GeV (tau mass)
            
        # Extract couplings from theory
        try:
            # Get effective electromagnetic coupling
            alpha_eff = self._extract_em_coupling(theory, energy_scale)
            
            # Basic Schwinger correction (universal for QED-like theories)
            schwinger_term = alpha_eff / (2 * np.pi)
            
            # Vertex correction from theory-specific modifications
            vertex_correction = self._calculate_vertex_correction(
                theory, lepton, energy_scale
            )
            
            # Vacuum polarization from heavy particles
            vacuum_pol = self._calculate_vacuum_polarization(
                theory, lepton, energy_scale
            )
            
            # Light-by-light if theory has 4-photon vertices
            lbl_correction = self._calculate_light_by_light(
                theory, lepton, energy_scale
            )
            
            # Total correction
            total = schwinger_term + vertex_correction + vacuum_pol + lbl_correction
            
            # Uncertainty estimate (theory-dependent)
            uncertainty = self._estimate_uncertainty(theory, total)
            
            # Convert to appropriate units
            if lepton == 'muon':
                return total, uncertainty
            elif lepton == 'electron':
                return total * 1e3, uncertainty * 1e3  # Convert to 10^-12
                
        except Exception as e:
            # If theory doesn't support quantum corrections
            return 0.0, float('inf')
    
    def _extract_em_coupling(self, theory, energy_scale: float) -> float:
        """
        Extract the effective electromagnetic coupling at given scale.
        
        For theories that modify QED, this might differ from α ≈ 1/137.
        """
        # Check if theory provides coupling constants
        if hasattr(theory, 'get_coupling_constants'):
            couplings = theory.get_coupling_constants(energy_scale)
            if 'electromagnetic' in couplings:
                return couplings['electromagnetic']
            elif 'alpha' in couplings:
                return couplings['alpha']
        
        # Default to standard QED value at this scale
        # Running of α from 0 to energy_scale
        alpha_0 = 1/137.035999084  # Fine structure constant
        
        # Simple one-loop running (for leptons)
        if energy_scale > 0.001:  # Above 1 MeV
            # Approximate running
            log_factor = np.log(energy_scale / 0.000511)  # Scale/electron mass
            alpha_eff = alpha_0 / (1 - (alpha_0/(3*np.pi)) * log_factor)
        else:
            alpha_eff = alpha_0
            
        return alpha_eff
    
    def _calculate_vertex_correction(
        self, 
        theory, 
        lepton: str,
        energy_scale: float
    ) -> float:
        """
        Calculate theory-specific vertex corrections beyond QED.
        
        This is where modified gravity theories might predict
        different quantum corrections.
        """
        # Check if theory has specific vertex corrections
        if hasattr(theory, 'calculate_vertex_correction'):
            return theory.calculate_vertex_correction(lepton, energy_scale)
        
        # Check for anomalous dimensions or form factors
        if hasattr(theory, 'get_form_factors'):
            form_factors = theory.get_form_factors(lepton, energy_scale)
            if 'magnetic' in form_factors:
                # F2(0) - 1 gives the anomalous moment correction
                return form_factors['magnetic'](0) - 1
        
        # Default: no additional vertex correction
        return 0.0
    
    def _calculate_vacuum_polarization(
        self,
        theory,
        lepton: str,
        energy_scale: float
    ) -> float:
        """
        Calculate vacuum polarization contributions from heavy particles.
        
        In SM, this includes contributions from heavier leptons and hadrons.
        Modified theories might have additional heavy states.
        """
        # Standard Model-like contributions
        correction = 0.0
        
        # For muon g-2, main contributions are from:
        # 1. Electron loops (largest)
        # 2. Tau loops
        # 3. Hadronic loops (dominant uncertainty)
        # 4. Weak boson loops
        
        if lepton == 'muon':
            # Approximate contributions (in units of 10^-11)
            # These would be calculated properly from Feynman diagrams
            
            # Electron loop contribution
            m_e = 0.000511  # GeV
            m_mu = 0.105658  # GeV
            correction += 1.094258e-3 * (m_e/m_mu)**2
            
            # Tau contribution (suppressed by mass)
            m_tau = 1.77686  # GeV
            correction += -7.85e-6 * (m_mu/m_tau)**2
            
            # Check for additional particles in theory
            if hasattr(theory, 'get_particle_spectrum'):
                spectrum = theory.get_particle_spectrum()
                for particle in spectrum:
                    if particle['charge'] != 0:  # Charged particles contribute
                        mass = particle['mass']
                        charge = particle['charge']
                        spin = particle.get('spin', 0.5)
                        
                        # Approximate contribution (would need proper loop calc)
                        if mass > m_mu:
                            contrib = charge**2 * (m_mu/mass)**2
                            if spin == 0:  # Scalar
                                contrib *= 0.25  # Different loop factor
                            correction += contrib
        
        return correction
    
    def _calculate_light_by_light(
        self,
        theory,
        lepton: str,
        energy_scale: float
    ) -> float:
        """
        Calculate light-by-light scattering contributions.
        
        This requires 4-photon vertices which might be present
        in theories with additional fields or non-minimal couplings.
        """
        # In SM, this is a small contribution (~10^-4 of total)
        # Enhanced in theories with light scalars/pseudoscalars
        
        if hasattr(theory, 'has_four_photon_vertex'):
            if theory.has_four_photon_vertex():
                # Would need proper 2-loop calculation
                # For now, return order of magnitude estimate
                alpha = self._extract_em_coupling(theory, energy_scale)
                return (alpha/np.pi)**2 * 0.1  # Rough estimate
        
        return 0.0
    
    def _estimate_uncertainty(self, theory, total_correction: float) -> float:
        """
        Estimate theoretical uncertainty in the calculation.
        
        Sources:
        1. Higher-order corrections (2-loop, 3-loop)
        2. Hadronic uncertainties
        3. Theory-specific uncertainties
        """
        # Base uncertainty from missing higher orders
        rel_uncertainty = 0.01  # 1% from 2-loop corrections
        
        # Add theory-specific uncertainties
        if hasattr(theory, 'get_theoretical_uncertainty'):
            rel_uncertainty += theory.get_theoretical_uncertainty('g-2')
        
        return abs(total_correction * rel_uncertainty)
    
    def validate(self, theory, verbose: bool = False) -> ValidationResult:
        """
        Validate theory's g-2 predictions against experimental data.
        
        Focus on muon g-2 where we have the clearest discrepancy.
        """
        result = ValidationResult()
        result.validator_name = "Anomalous Magnetic Moment (g-2)"
        
        # Calculate for muon (most interesting case)
        lepton = 'muon'
        exp_data = self.experimental_values[lepton]
        sm_pred = self.sm_predictions[lepton]
        
        # Get theory prediction
        theory_correction, theory_uncertainty = self.calculate_one_loop_correction(
            theory, lepton
        )
        
        # Theory's total prediction
        theory_value = sm_pred['value'] + theory_correction
        theory_error = np.sqrt(sm_pred['error']**2 + theory_uncertainty**2)
        
        # Compare with experiment
        exp_value = exp_data['value']
        exp_error = exp_data['error']
        
        # Calculate chi-squared
        diff = theory_value - exp_value
        combined_error = np.sqrt(theory_error**2 + exp_error**2)
        chi2 = (diff / combined_error)**2
        
        # Also calculate improvement over SM
        sm_diff = sm_pred['value'] - exp_value
        sm_chi2 = (sm_diff / np.sqrt(sm_pred['error']**2 + exp_error**2))**2
        
        # Pass if theory improves over SM
        result.passed = chi2 < sm_chi2
        result.loss = chi2
        
        # Additional metrics
        result.extra_data = {
            'theory_a_mu': theory_value,
            'theory_uncertainty': theory_error,
            'exp_a_mu': exp_value,
            'exp_uncertainty': exp_error,
            'sm_a_mu': sm_pred['value'],
            'sm_uncertainty': sm_pred['error'],
            'theory_correction': theory_correction,
            'chi2': chi2,
            'sm_chi2': sm_chi2,
            'improvement_sigma': np.sqrt(sm_chi2) - np.sqrt(chi2)
        }
        
        # Detailed notes
        if result.passed:
            improvement = result.extra_data['improvement_sigma']
            result.notes = f"Theory improves g-2 prediction by {improvement:.1f}σ"
        else:
            result.notes = f"Theory prediction χ²={chi2:.1f} vs SM χ²={sm_chi2:.1f}"
        
        if verbose:
            print(f"\n{theory.name} g-2 Validation:")
            print(f"  Experimental a_μ: {exp_value:.2e} ± {exp_error:.2e}")
            print(f"  SM prediction: {sm_pred['value']:.2e} ± {sm_pred['error']:.2e}")
            print(f"  Theory prediction: {theory_value:.2e} ± {theory_error:.2e}")
            print(f"  Theory correction: {theory_correction:.2e}")
            print(f"  χ² comparison: Theory={chi2:.1f}, SM={sm_chi2:.1f}")
            print(f"  Result: {'PASS' if result.passed else 'FAIL'}")
        
        return result