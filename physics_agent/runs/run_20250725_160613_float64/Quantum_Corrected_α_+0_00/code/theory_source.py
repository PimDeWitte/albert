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
    # <reason>chain: Correct default alpha=0</reason>
    preferred_params = {'alpha': 1e-5}  # Small non-zero to avoid instability
    cacheable = True
    def __init__(self, alpha: float = 0.0):
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

        super().__init__(
            f"Quantum Corrected (α={alpha:+.2f})",
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
        
        # <reason>Planck mass for dimensional consistency in quantum corrections</reason>
        m_p = torch.sqrt(torch.tensor(hbar * const_c / const_G, device=r.device, dtype=r.dtype))
        
        # <reason>Quantum correction: modifies metric with higher-order curvature terms</reason>
        # <reason>α controls strength of quantum effects relative to classical GR</reason>
        rs_over_r = rs / r
        quantum_term = alpha * (hbar / (m_p * r))**2  # <reason>Dimensionless quantum correction</reason>
        
        m = 1 - rs_over_r + quantum_term * rs_over_r**2
        
        epsilon = 1e-10
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 