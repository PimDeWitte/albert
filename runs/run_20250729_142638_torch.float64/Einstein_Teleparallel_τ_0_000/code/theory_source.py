import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor, QuantumMixin
import torch

class EinsteinTeleparallel(GravitationalTheory, QuantumMixin):
    """
    Einstein's teleparallel equivalent of GR with unification.
    <reason>Formulates gravity using torsion instead of curvature.</reason>
    <reason>Parameter tau controls torsion-matter coupling for unification.</reason>
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(tau={'min': 0.0, 'max': 0.1, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Default τ=0 recovers standard teleparallel gravity</reason>
    preferred_params = {'tau': 0.0}
    cacheable = True
    def __init__(self, tau: float = 0.0):
        # <reason>chain: Define quantum field components for teleparallel theory</reason>
        T = get_symbol('T')  # Torsion scalar
        
        # Matter fields coupled to torsion
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        gamma_mu = get_symbol('γ^μ')
        
        # Gauge fields in teleparallel formalism
        F_mn = get_symbol('F_μν')
        
        # Contorsion tensor
        
        # Lagrangians with teleparallel structure
        gravity_lagrangian = T + tau * get_symbol('T_μνρ') * get_symbol('T^μνρ')
        matter_lagrangian = sp.I * psi_bar * gamma_mu * get_symbol('D_μ') * psi - get_symbol('m_f') * psi_bar * psi + tau * get_symbol('K_μνρ') * psi_bar * get_symbol('σ^μν') * psi
        gauge_lagrangian = -sp.Rational(1,4) * F_mn * get_symbol('F^μν') + tau * T * F_mn * get_symbol('F^μν')
        interaction_lagrangian = -get_symbol('q') * psi_bar * gamma_mu * psi * get_symbol('A_μ') * (1 + tau * T)
        
        # <reason>chain: Build Lagrangians directly using get_symbol calls</reason>

        super().__init__(
            f"Einstein Teleparallel (τ={tau:.3f})",
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        # Matter Lagrangian: ψ̄(iγ^μD_μ - m)ψ
        # Gauge Lagrangian: -1/4 F_μν F^μν
        # Interaction Lagrangian: qψ̄γ^μA_μψ
        self.tau = tau
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components for teleparallel gravity.
        <reason>In teleparallel formulation, gravity manifests through torsion modifications.</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        tau = torch.tensor(self.tau, device=r.device, dtype=r.dtype)
        
        # <reason>Base Schwarzschild metric</reason>
        m_base = 1 - rs / r
        
        # <reason>Torsion modification: affects metric through vierbein fields</reason>
        # <reason>τ parameter introduces corrections from torsion scalar</reason>
        torsion_correction = tau * (rs / r)**2
        m = m_base - torsion_correction
        
        epsilon = 1e-10
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 