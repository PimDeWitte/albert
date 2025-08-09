import torch
import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor
class GaugeGravity(GravitationalTheory):
    """
    Gauge theory of gravity unifying gravitational and gauge interactions.
    <reason>Treats gravity as a gauge theory similar to other fundamental forces.</reason>
    <reason>Parameter kappa controls gauge-gravity coupling strength.</reason>
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(kappa={'min': 0.0, 'max': 2.0, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Default κ=0 corresponds to decoupled limit</reason>
    preferred_params = {'kappa': 0.0}
    cacheable = True
    def __init__(self, kappa: float = 0.0):
        # <reason>chain: Define quantum field components for gauge gravity</reason>
        
        # Matter fields
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        gamma_mu = get_symbol('γ^μ')
        
        # Gauge fields - both Abelian and non-Abelian
        
        # Gauge coupling constants
        
        # Lagrangians with gauge unification
        gravity_lagrangian = get_symbol('R') + kappa * get_symbol('tr(F_μν F^μν)')  # Gauge-like gravity term
        matter_lagrangian = sp.I * psi_bar * gamma_mu * get_symbol('D_μ') * psi - get_symbol('m_f') * psi_bar * psi
        gauge_lagrangian = -sp.Rational(1,4) * get_symbol('F_μν') * get_symbol('F^μν') - sp.Rational(1,4) * get_symbol('tr(G_μν^a G^μν_a)')
        interaction_lagrangian = -get_symbol('g_1') * psi_bar * gamma_mu * psi * get_symbol('A_μ') - get_symbol('g_2') * psi_bar * gamma_mu * get_symbol('T^a') * psi * get_symbol('A_μ^a')
        
        # <reason>chain: Build Lagrangians directly using get_symbol calls</reason>

        super().__init__(
            f"Gauge Gravity (κ={kappa:.2f})",
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        # Matter Lagrangian: ψ̄(iγ^μD_μ - m)ψ
        # Gauge Lagrangian: -1/4 F_μν F^μν
        # Interaction Lagrangian: qψ̄γ^μA_μψ
        self.kappa = kappa
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None, **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components for gauge gravity.
        
        Gauge corrections modify the standard metric.
        """
        rs = 2 * G_param * M_param / C_param**2
        kappa_tensor = torch.tensor(self.kappa, device=r.device, dtype=r.dtype)
        
        # Standard Schwarzschild
        f = 1 - rs / r
        
        # Gauge correction
        gauge_term = kappa_tensor * (rs / r)**2
        f_modified = f - gauge_term
        
        epsilon = 1e-10
        g_tt = -f_modified
        g_rr = 1 / (f_modified + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 