import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor, QuantumMixin
import torch

class EinsteinAsymmetric(GravitationalTheory, QuantumMixin):
    """
    Einstein's asymmetric unified field theory.
    <reason>Uses asymmetric metric tensor to unify gravity and electromagnetism.</reason>
    <reason>Parameter alpha controls antisymmetric field strength.</reason>
    
    Based on Einstein's nonsymmetric gravitational theory attempts.
    Source: https://en.wikipedia.org/wiki/Nonsymmetric_gravitational_theory
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(alpha={'min': -1e-3, 'max': 1e-3, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Default α=0 corresponds to symmetric metric (pure GR)</reason>
    preferred_params = {'alpha': 0.0}
    cacheable = True
    def __init__(self, alpha: float = 0.0):
        # <reason>chain: Define quantum field components for asymmetric theory</reason>
        g_mn = get_symbol('g_μν')  # Symmetric part
        f_mn = get_symbol('f_μν')  # Antisymmetric part
        
        # Matter fields
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        gamma_mu = get_symbol('γ^μ')
        
        # Electromagnetic field from antisymmetric part
        F_mn = alpha * (f_mn - get_symbol('f_νμ'))
        
        # Unified metric
        g_mn + sp.I * alpha * f_mn
        
        # Lagrangians with asymmetric unification
        gravity_lagrangian = get_symbol('R') + alpha**2 * get_symbol('f_μν') * get_symbol('f^μν')
        matter_lagrangian = sp.I * psi_bar * gamma_mu * get_symbol('D_μ') * psi - get_symbol('m_f') * psi_bar * psi
        gauge_lagrangian = -sp.Rational(1,4) * F_mn * get_symbol('F^μν')
        interaction_lagrangian = -get_symbol('q') * psi_bar * gamma_mu * psi * get_symbol('A_μ') * (1 + alpha * get_symbol('f^μν') * g_mn)
        
        # <reason>chain: Build Lagrangians directly using get_symbol calls</reason>

        super().__init__(
            f"Einstein Asymmetric (α={alpha:.1e})",
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        # Matter Lagrangian: ψ̄(iγ^μD_μ - m)ψ
        # Gauge Lagrangian: -1/4 F_μν F^μν
        # Interaction Lagrangian: qψ̄γ^μA_μψ
        self.alpha = alpha
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components for asymmetric field theory.
        <reason>Introduces off-diagonal terms representing electromagnetic-like effects.</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        alpha = torch.tensor(self.alpha, device=r.device, dtype=r.dtype)
        
        # <reason>Standard Schwarzschild diagonal components</reason>
        m = 1 - rs / r
        
        epsilon = 1e-10
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2
        
        # <reason>Key feature: non-zero off-diagonal term from asymmetry</reason>
        # <reason>α controls coupling between time and angular coordinates</reason>
        g_tp = alpha * rs / r * torch.sqrt(r)
        
        return g_tt, g_rr, g_pp, g_tp 