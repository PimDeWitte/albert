import torch
import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor
class SpinorConformal(GravitationalTheory):
    """
    Spinor-based conformal quantum gravity.
    <reason>Uses spinor fields as fundamental with conformal symmetry.</reason>
    <reason>Parameter lambda controls conformal coupling strength.</reason>
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(lambda_c={'min': 0.0, 'max': 1.0, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Default λ=0 corresponds to minimal coupling</reason>
    preferred_params = {'lambda_c': 0.0}
    cacheable = True
    def __init__(self, lambda_c: float = 0.0):
        # <reason>chain: Define quantum field components with spinor/conformal emphasis</reason>
        R = get_symbol('R')
        
        # Spinor fields (fundamental)
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        chi = get_symbol('χ')  # Second spinor
        chi_bar = get_symbol('χ̄')
        
        # Spinor masses
        
        # Dirac matrices
        gamma_mu = get_symbol('γ^μ')
        D_mu = get_symbol('D_μ')
        
        # Gauge fields
        A_mu = get_symbol('A_μ')
        
        # Conformal factor
        Omega = get_symbol('Ω')
        
        # Lagrangians with conformal coupling
        gravity_lagrangian = R + lambda_c * get_symbol('C_μνρσ') * get_symbol('C^μνρσ')
        matter_lagrangian = (sp.I * psi_bar * gamma_mu * D_mu * psi - get_symbol('m_ψ') * psi_bar * psi * Omega + 
                            sp.I * chi_bar * gamma_mu * D_mu * chi - get_symbol('m_χ') * chi_bar * chi * Omega +
                            lambda_c * sp.Rational(1,6) * R * (psi_bar * psi + chi_bar * chi))
        gauge_lagrangian = -sp.Rational(1,4) * get_symbol('F_μν') * get_symbol('F^μν') * Omega**(-2)
        interaction_lagrangian = (-get_symbol('q_psi') * psi_bar * gamma_mu * psi * A_mu - 
                                 get_symbol('q_chi') * chi_bar * gamma_mu * chi * A_mu +
                                 lambda_c * psi_bar * get_symbol('γ^5') * chi * get_symbol('φ'))  # Yukawa-like
        
        # <reason>chain: Build Lagrangians directly using get_symbol calls</reason>

        super().__init__(
            f"Spinor-Conformal QG (λ={lambda_c:.2f})",
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        # Matter Lagrangian: ψ̄(iγ^μD_μ - m)ψ
        # Interaction Lagrangian: qψ̄γ^μA_μψ
        self.lambda_c = lambda_c
        self.beta = lambda_c / 10.0 if lambda_c != 0 else 0.01  # <reason>chain: Conformal coupling parameter for metric</reason>
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components for spinor-conformal gravity.
        
        Conformal and spinor effects modify the spacetime geometry.
        """
        rs = 2 * G_param * M_param / C_param**2
        beta_tensor = torch.tensor(self.beta, device=r.device, dtype=r.dtype)
        
        # Standard Schwarzschild base
        f = 1 - rs / r
        
        # Conformal modification with spinor coupling
        conformal_factor = 1 + beta_tensor * torch.log(r / rs)
        f_modified = f * conformal_factor
        
        epsilon = 1e-10
        g_tt = -f_modified
        g_rr = 1 / (f_modified + epsilon)
        g_pp = r**2 * conformal_factor  # Conformal scaling
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 