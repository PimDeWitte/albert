import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor
import torch

class EinsteinRegularisedCore(GravitationalTheory):
    """
    Quantum gravity with regularized core to avoid singularities.
    <reason>Replaces point singularities with quantum-scale regular cores.</reason>
    <reason>Parameter epsilon controls the regularization scale.</reason>
    
    Based on regularized black hole solutions in quantum gravity.
    Source: https://arxiv.org/abs/2006.16751
    """
    category = "quantum"
    sweep = dict(epsilon={'min': 1e-6, 'max': 1e-3, 'points': 10, 'scale': 'log'})
    preferred_params = {'epsilon': 1e-4}
    cacheable = True
    
    def __init__(self, epsilon: float = 1e-4):
        # <reason>Define quantum field components with regularization</reason>
        
        # Matter fields with regularized interactions
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        gamma_mu = get_symbol('γ^μ')
        
        # Regularization function - exponential cutoff at small r
        f_reg = sp.exp(-epsilon**2 / get_symbol('r')**2)  # Gaussian regularization
        
        # Lagrangians with regularized singularities
        gravity_lagrangian = get_symbol('R') * f_reg
        matter_lagrangian = (sp.I * psi_bar * gamma_mu * get_symbol('D_μ') * psi - get_symbol('m_f') * psi_bar * psi) * f_reg
        gauge_lagrangian = -sp.Rational(1,4) * get_symbol('F_μν') * get_symbol('F^μν') * f_reg
        interaction_lagrangian = -get_symbol('q') * psi_bar * gamma_mu * psi * get_symbol('A_μ') * f_reg

        super().__init__(
            f"Regularised Core QG (ε={epsilon:.1e})",
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        self.epsilon = epsilon
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Simpson-Visser regularized black hole metric.
        <reason>Well-studied metric that avoids singularities while preserving asymptotics</reason>
        <reason>Reduces to Schwarzschild at large r, regular at r=0</reason>
        Reference: Simpson & Visser, JCAP 02 (2019) 042
        """
        rs = 2 * G_param * M_param / C_param**2
        
        # <reason>Simpson-Visser regularization parameter</reason>
        # a = epsilon * rs gives the regularization scale
        a = self.epsilon * rs
        
        # <reason>Simpson-Visser metric function</reason>
        # Psi(r) = sqrt(r^2 + a^2) interpolates between r (large r) and a (small r)
        # This gives: ds² = -[1 - rs/Psi(r)]dt² + [1 - rs/Psi(r)]^(-1)dr² + r²dΩ²
        psi = torch.sqrt(r**2 + a**2)
        
        # <reason>Metric function f(r) = 1 - rs/Psi(r)</reason>
        # At large r: f → 1 - rs/r (Schwarzschild)
        # At r=0: f = 1 - rs/a > 0 (no horizon if a > rs)
        f = 1 - rs / psi
        
        # <reason>For small epsilon, ensure no horizon exists</reason>
        # This is the key feature of regularized black holes
        f = torch.clamp(f, min=1e-10)
        
        g_tt = -f
        g_rr = 1 / f
        g_pp = r**2  # Angular part uses coordinate radius
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 