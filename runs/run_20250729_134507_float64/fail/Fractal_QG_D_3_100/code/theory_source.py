import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor, QuantumMixin
import torch
class Fractal(GravitationalTheory, QuantumMixin):
    """
    Fractal quantum gravity with non-integer dimensional spacetime.
    <reason>Spacetime has fractal structure at quantum scales.</reason>
    <reason>Parameter D controls the fractal dimension deviation from 4.</reason>
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(D={'min': 2.95, 'max': 3.05, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Default D=3.0 corresponds to standard spatial dimensions</reason>
    preferred_params = {'D': 3.1}  # Slightly off 3.0 for variation
    cacheable = True
    def __init__(self, D: float = 3.0):
        # <reason>chain: Define quantum field components with fractal structure</reason>
        
        # Matter fields on fractal spacetime
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        gamma_mu = get_symbol('γ^μ')
        
        # Gauge fields
        
        # Fractal measure
        mu_D = get_symbol('r')**(D-3)  # Fractal volume element
        
        # Lagrangians with fractal modifications
        gravity_lagrangian = get_symbol('R') * mu_D
        matter_lagrangian = (sp.I * psi_bar * gamma_mu * get_symbol('D_μ') * psi - get_symbol('m_f') * psi_bar * psi) * mu_D
        gauge_lagrangian = -sp.Rational(1,4) * get_symbol('F_μν') * get_symbol('F^μν') * mu_D
        interaction_lagrangian = -get_symbol('q') * psi_bar * gamma_mu * psi * get_symbol('A_μ') * mu_D
        
        # <reason>chain: Build Lagrangians directly using get_symbol calls</reason>

        super().__init__(
            f"Fractal QG (D={D:.3f})",
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        # Matter Lagrangian: ψ̄(iγ^μD_μ - m)ψ
        # Interaction Lagrangian: qψ̄γ^μA_μψ
        self.D = D
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components for fractal spacetime.
        <reason>Fractal dimension modifies how gravity scales with distance.</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        D = torch.tensor(self.D, device=r.device, dtype=r.dtype)
        
        # <reason>In fractal spacetime, gravitational potential scales as r^(1-D)</reason>
        # <reason>For D=3, recovers standard 1/r potential</reason>
        # <reason>D<3: stronger gravity at large r; D>3: weaker gravity at large r</reason>
        fractal_exponent = D - 3
        
        # <reason>Modified metric with fractal scaling</reason>
        m = 1 - rs / r * (r / rs)**fractal_exponent
        
        epsilon = 1e-10
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2  # <reason>Angular part unchanged in spherical symmetry</reason>
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 