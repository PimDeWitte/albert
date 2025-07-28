import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor, QuantumMixin
import torch
from scipy.constants import G as const_G, c as const_c, hbar
class QuantumCorrected(GravitationalTheory, QuantumMixin):
    """
    Quantum-corrected General Relativity with parameter α controlling quantum effects.
    <reason>Incorporates quantum corrections to spacetime geometry through α-dependent terms.</reason>
    <reason>Tests showed varying performance across parameter range, indicating quantum scale sensitivity.</reason>
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(alpha={'min': -2.0, 'max': 2.0, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Test with alpha=0 to see if quantum corrections are causing extreme observational failures</reason>
    preferred_params = {'alpha': 0.0}  # No quantum correction - reduces to standard GR
    cacheable = True
    def __init__(self, alpha: float = None):
        # <reason>chain: Use preferred_params if alpha not specified</reason>
        if alpha is None:
            alpha = self.preferred_params.get('alpha', 0.1)
        # <reason>chain: Define quantum field components</reason>
        
        # Matter fields
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        gamma_mu = get_symbol('γ^μ')
        
        # Gauge fields (simple EM for quantum gravity test)
        
        # Lagrangians
        matter_lagrangian = sp.I * psi_bar * gamma_mu * get_symbol('D_μ') * psi - get_symbol('m_f') * psi_bar * psi
        gauge_lagrangian = -sp.Rational(1,4) * get_symbol('F_μν') * get_symbol('F^μν')
        interaction_lagrangian = -get_symbol('q') * psi_bar * gamma_mu * psi * get_symbol('A_μ')  # Minimal coupling
        
        # <reason>chain: Build Lagrangians directly using get_symbol calls</reason>

        # <reason>chain: Warn when alpha=0 as this reduces to standard GR with quantum overhead</reason>
        if alpha == 0:
            import warnings
            warnings.warn(
                f"\nQuantum Corrected theory with α=0 is equivalent to Schwarzschild/Kerr/Kerr-Newman "
                f"but uses quantum calculations.\nThis adds computational overhead without quantum effects. "
                f"Consider using standard GR theories instead.",
                UserWarning,
                stacklevel=2
            )

        # <reason>chain: Use scientific notation for small alpha values</reason>
        if abs(alpha) < 0.01:
            name = f"Quantum Corrected (α={alpha:+.2e})"
        else:
            name = f"Quantum Corrected (α={alpha:+.2f})"
            
        super().__init__(
            name,
            lagrangian=get_symbol('R'),
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        self.alpha = alpha
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components with quantum corrections.
        <reason>Quantum effects modify metric through α-dependent corrections to curvature.</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        alpha = torch.tensor(self.alpha, device=r.device, dtype=r.dtype)
        
        # <reason>chain: Use phenomenological quantum correction for visible effects</reason>
        # Instead of Planck-scale corrections, use a correction that scales with rs/r
        # This makes quantum effects visible at astrophysical scales
        rs_over_r = rs / r
        
        # Quantum correction as a fraction of the classical term  
        # Use rs_over_r (not squared) for stronger effects at larger radii
        # α controls the strength and sign of quantum corrections
        # NOTE: When α=0, this reduces to standard Schwarzschild metric
        # <reason>chain: Fix quantum correction to modify rather than cancel classical term</reason>
        # The quantum term should multiply the classical term, not add to it linearly
        quantum_factor = 1 + alpha * rs_over_r  # Multiplicative correction
        
        m = 1 - rs_over_r * quantum_factor  # When α=0: m = 1 - rs/r (Schwarzschild)
        
        epsilon = 1e-10
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 