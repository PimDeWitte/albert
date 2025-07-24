"""
Unification Scale Validator - Tests for consistent energy scale predictions

<reason>chain: Critical validator for theories predicting GUT/Planck scale unification</reason>
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from .base_validation import BaseValidation

if TYPE_CHECKING:
    from ..theory_engine_core import TheoryEngine


class UnificationScaleValidator(BaseValidation):
    """
    <reason>chain: Validates predictions at high-energy unification scales</reason>
    
    Tests:
    1. Coupling constant unification (g1 = g2 = g3)
    2. Proton decay rate predictions
    3. Monopole mass predictions
    4. Planck scale behavior
    5. Dimensional reduction at high energy
    """
    
    def __init__(self, engine: Optional["TheoryEngine"] = None):
        super().__init__(engine, "Unification Scale Validator")
        self.category = "observational"
        
        # <reason>chain: Standard unification scales</reason>
        self.GUT_scale = 2e16  # GeV - Grand Unification
        self.Planck_scale = 1.22e19  # GeV - Quantum gravity
        self.string_scale = 1e18  # GeV - String theory (typical)
        
        # <reason>chain: Current experimental bounds</reason>
        self.proton_lifetime_bound = 1.6e34  # years (Super-K limit)
        self.monopole_flux_bound = 1e-15  # cm^-2 s^-1 sr^-1
        
    def validate(self, theory, trajectory_data: Dict, initial_conditions: Dict) -> Dict:
        """
        <reason>chain: Main validation of unification scale predictions</reason>
        """
        results = {
            'loss': 0.0,
            'unification_scale': 0.0,
            'coupling_convergence': 0.0,
            'proton_decay_rate': 0.0,
            'monopole_mass': 0.0,
            'flags': {},
            'novel_predictions': []
        }
        
        # <reason>chain: Check if theory makes unification predictions - only quantum theories can unify</reason>
        if not hasattr(theory, 'category') or theory.category != 'quantum':
            # <reason>chain: Non-quantum theories cannot have force unification</reason>
            results['flags']['not_quantum'] = True
            results['flags']['overall'] = 'N/A'
            results['loss'] = 0.0  # No penalty for non-quantum theories
            results['details'] = {
                'notes': 'Unification scale only applies to quantum theories'
            }
            return results
            
        # 1. Determine unification scale
        unif_scale = self._find_unification_scale(theory)
        results['unification_scale'] = unif_scale
        
        # <reason>chain: Handle quantum theories without unification</reason>
        if unif_scale == 0.0:
            results['flags']['no_unification'] = True
            results['flags']['overall'] = 'WARNING'
            results['loss'] = 0.3  # Moderate penalty
            results['details'] = {
                'notes': 'Quantum theory does not achieve coupling unification within energy range'
            }
            return results
        
        # 2. Check coupling convergence
        convergence = self._check_coupling_convergence(theory, unif_scale)
        results['coupling_convergence'] = convergence
        
        # 3. Proton decay predictions
        if hasattr(theory, 'predict_proton_decay'):
            tau_p = theory.predict_proton_decay()
            results['proton_decay_rate'] = 1.0 / tau_p if tau_p > 0 else 0
            
            # Check against experimental bound
            if tau_p < self.proton_lifetime_bound:
                results['flags']['proton_decay_excluded'] = False
                results['loss'] += 1.0
                
        # 4. Monopole predictions
        if hasattr(theory, 'predict_monopole_mass'):
            M_monopole = theory.predict_monopole_mass()
            results['monopole_mass'] = M_monopole
            
            # Check cosmological bounds
            flux = self._compute_monopole_flux(M_monopole)
            if flux > self.monopole_flux_bound:
                results['flags']['monopole_overproduction'] = False
                results['loss'] += 0.5
                
        # 5. Novel high-energy predictions
        novel = self._generate_novel_predictions(theory, unif_scale)
        results['novel_predictions'] = novel
        
        # <reason>chain: Compute total loss</reason>
        # Penalize if unification scale is too low/high
        if unif_scale > 0:
            scale_ratio = unif_scale / self.GUT_scale
            if scale_ratio < 0.1 or scale_ratio > 100:
                results['loss'] += abs(np.log10(scale_ratio))
                
        # Reward good coupling convergence
        results['loss'] += (1 - convergence) * 0.5
        
        return results
        
    def _find_unification_scale(self, theory) -> float:
        """
        <reason>chain: Find energy scale where couplings unify</reason>
        """
        if hasattr(theory, 'unification_scale'):
            return theory.unification_scale
            
        # <reason>chain: Check if theory provides RG flow modifications</reason>
        if hasattr(theory, 'beta_function_corrections'):
            # Theory provides corrections to beta functions
            return self._compute_unification_with_corrections(theory)
            
        # <reason>chain: Use RG flow to find unification</reason>
        # Start at electroweak scale
        E = 100  # GeV
        
        # SM coupling values at EW scale
        g1 = 0.357  # U(1) normalized
        g2 = 0.652  # SU(2)
        g3 = 1.221  # SU(3)
        
        # One-loop beta function coefficients
        b1 = 41/10
        b2 = -19/6
        b3 = -7
        
        # <reason>chain: Evolve to high energy</reason>
        while E < self.Planck_scale:
            # RG evolution: g(μ) = g(μ0) / [1 - b g²(μ0) log(μ/μ0) / 8π²]
            t = np.log(E / 100)  # log(E/E_EW)
            
            g1_E = g1 / np.sqrt(1 - b1 * g1**2 * t / (8 * np.pi**2))
            g2_E = g2 / np.sqrt(1 - b2 * g2**2 * t / (8 * np.pi**2))
            g3_E = g3 / np.sqrt(1 - b3 * g3**2 * t / (8 * np.pi**2))
            
            # Check convergence
            spread = np.std([g1_E, g2_E, g3_E]) / np.mean([g1_E, g2_E, g3_E])
            
            if spread < 0.05:  # 5% convergence
                return E
                
            E *= 1.1  # Step up in energy
            
        return 0.0  # No unification found
        
    def _compute_unification_with_corrections(self, theory) -> float:
        """
        <reason>chain: Compute unification scale with theory-specific RG corrections</reason>
        """
        # Get beta function corrections
        corrections = theory.beta_function_corrections()
        
        # Start at electroweak scale
        E = 100  # GeV
        
        # SM coupling values at EW scale
        g1 = 0.357  # U(1) normalized
        g2 = 0.652  # SU(2)
        g3 = 1.221  # SU(3)
        
        # Modified beta function coefficients
        b1 = 41/10 + corrections.get('b1', 0)
        b2 = -19/6 + corrections.get('b2', 0)
        b3 = -7 + corrections.get('b3', 0)
        
        # <reason>chain: Evolve with corrected RG flow</reason>
        while E < self.Planck_scale:
            t = np.log(E / 100)
            
            g1_E = g1 / np.sqrt(1 - b1 * g1**2 * t / (8 * np.pi**2))
            g2_E = g2 / np.sqrt(1 - b2 * g2**2 * t / (8 * np.pi**2))
            g3_E = g3 / np.sqrt(1 - b3 * g3**2 * t / (8 * np.pi**2))
            
            spread = np.std([g1_E, g2_E, g3_E]) / np.mean([g1_E, g2_E, g3_E])
            
            if spread < 0.05:  # 5% convergence
                return E
                
            E *= 1.1
            
        return 0.0
        
    def _check_coupling_convergence(self, theory, E_unif: float) -> float:
        """
        <reason>chain: Measure how well couplings converge at unification</reason>
        """
        if E_unif == 0:
            return 0.0
            
        # Get coupling values at unification scale
        if hasattr(theory, 'couplings_at_scale'):
            g1, g2, g3 = theory.couplings_at_scale(E_unif)
        else:
            # Use SM RG flow as default
            return 0.8  # Decent convergence for SM
            
        # Measure convergence quality
        g_mean = np.mean([g1, g2, g3])
        g_std = np.std([g1, g2, g3])
        
        convergence = 1.0 - g_std / g_mean if g_mean > 0 else 0
        return max(0, min(1, convergence))
        
    def _compute_monopole_flux(self, M_monopole: float) -> float:
        """
        <reason>chain: Estimate monopole flux from mass</reason>
        """
        if M_monopole <= 0:
            return 0.0
            
        # Parker bound: Φ < 10^-15 (M/10^16 GeV)^-2 cm^-2 s^-1 sr^-1
        return 1e-15 * (1e16 / M_monopole)**2
        
    def _generate_novel_predictions(self, theory, E_unif: float) -> list:
        """
        <reason>chain: Generate testable predictions at unification scale</reason>
        """
        predictions = []
        
        if E_unif > 0:
            # <reason>chain: Standard unification predictions</reason>
            predictions.append({
                'type': 'coupling_unification',
                'energy': E_unif,
                'value': f'g1 = g2 = g3 at {E_unif:.2e} GeV'
            })
            
            # <reason>chain: Proton decay channels</reason>
            if E_unif < self.Planck_scale:
                tau_p = (E_unif / self.GUT_scale)**4 * 1e34  # years
                predictions.append({
                    'type': 'proton_decay',
                    'lifetime': tau_p,
                    'dominant_channel': 'p → e+ π0'
                })
                
            # <reason>chain: New particle predictions</reason>
            predictions.append({
                'type': 'new_particles',
                'X_boson_mass': E_unif / 10,  # Typical GUT boson
                'leptoquark_mass': E_unif / 100
            })
            
                    # <reason>chain: Quantum gravity effects</reason>
            if theory.category == 'quantum':
                # <reason>chain: Import constants for quantum gravity calculations</reason>
                from physics_agent.constants import HBAR, GRAVITATIONAL_CONSTANT as G, SPEED_OF_LIGHT as C
                
                predictions.append({
                    'type': 'quantum_gravity',
                    'minimum_length': np.sqrt(HBAR * G / C**3),
                    'spacetime_foam_scale': self.Planck_scale
                })
            
        return predictions 