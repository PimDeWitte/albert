import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor, QuantumMixin
import torch

class Yukawa(GravitationalTheory, QuantumMixin):
    """
    Yukawa-type modification to gravity with exponential screening.
    <reason>Adds exponential decay term modeling short-range modifications to gravity.</reason>
    <reason>λ is the characteristic length scale of the Yukawa potential.</reason>
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(lambda_rs={'min': 1.5, 'max': 100.0, 'points': 10, 'scale': 'log'})
    # <reason>chain: Set large lambda_rs for Newtonian limit</reason>
    preferred_params = {'lambda_rs': 1e6}
    cacheable = True

    def __init__(self, lambda_rs: float = 10.0):
        # Define quantum field components
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        gamma_mu = get_symbol('γ^μ')
        D_mu = get_symbol('D_μ')
        m_f = get_symbol('m_f')
        e = get_symbol('e')
        A_mu = get_symbol('A_μ')
        
        # Define Lagrangians
        matter_lagrangian = sp.I * psi_bar * gamma_mu * D_mu * psi - m_f * psi_bar * psi
        gauge_lagrangian = -sp.Rational(1, 4) * get_symbol('F_μν') * get_symbol('F^μν')
        interaction_lagrangian = -e * psi_bar * gamma_mu * psi * A_mu
        
        # Yukawa gravitational Lagrangian
        lagrangian = get_symbol('R') + get_symbol('alpha') * sp.exp(-get_symbol('r') / get_symbol('lambda_rs'))
        
        super().__init__(
            f"Yukawa (λ={lambda_rs:.2f} RS)",
            lagrangian=lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        self.lambda_rs = lambda_rs  # Length scale in units of Schwarzschild radius
        # <reason>Coupling strength must be small for weak field validity: |α| << 1</reason>
        # <reason>Literature suggests α ~ 10^-9 to 10^-3 for Solar System constraints</reason>
        self.alpha = 1e-6  # <reason>Small coupling strength to ensure weak field approximation</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components with Yukawa-type modification.
        <reason>Yukawa potential adds exponentially screened contribution to gravity.</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        lambda_val = torch.tensor(self.lambda_rs, device=r.device, dtype=r.dtype)
        
        # <reason>Yukawa length scale λ in physical units</reason>
        lambda_phys = lambda_val * rs
        
        # <reason>Yukawa modification: gravity strengthens then weakens exponentially beyond λ</reason>
        # <reason>Standard form: V_Yukawa = -GM/r(1 + α*e^(-r/λ)) where α is coupling strength</reason>
        yukawa_factor = torch.exp(-r / lambda_phys)
        
        # <reason>Modified metric function with Yukawa correction</reason>
        # <reason>g_tt = -(1 - rs/r - α*rs/r*e^(-r/λ)) = -(1 - rs/r*(1 + α*e^(-r/λ)))</reason>
        m = 1 - rs / r * (1 + self.alpha * yukawa_factor)  # <reason>alpha=0.5 is coupling strength</reason>
        
        # <reason>chain: No epsilon needed - m is always positive at weak fields</reason>
        g_tt = -m
        # <reason>chain: For PPN validity, compute g_rr exactly without approximations</reason>
        # The issue is that approximations lose precision at weak fields
        # Use exact formula 1/m but protect against near-horizon singularities
        epsilon = 1e-30  # <reason>Very small epsilon for horizon protection only</reason>
        g_rr = torch.where(
            m > 1e-10,  # <reason>Far from horizon - use exact formula</reason>
            1 / m,
            1 / (m + epsilon)  # <reason>Near horizon protection</reason>
        )
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 