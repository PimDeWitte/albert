import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor
import torch
class LogCorrected(GravitationalTheory):
    """
    Logarithmically corrected quantum gravity.
    <reason>Incorporates quantum loop corrections appearing as log(r) terms.</reason>
    <reason>Parameter gamma controls strength of quantum corrections.</reason>
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(gamma={'min': 0.0, 'max': 0.1, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Default γ=0 corresponds to classical GR</reason>
    preferred_params = {'gamma': 1e-5}
    cacheable = True
    def __init__(self, gamma: float = 0.0):
        # <reason>chain: Define quantum field components with loop corrections</reason>
        
        # Matter fields with quantum corrections
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        gamma_mu = get_symbol('γ^μ')
        
        # Gauge fields
        
        # Loop correction factor
        L = sp.log(get_symbol('r') / get_symbol('l_P'))  # l_P is Planck length
        
        # Lagrangians with logarithmic quantum corrections
        gravity_lagrangian = get_symbol('R') * (1 + gamma * L)
        matter_lagrangian = sp.I * psi_bar * gamma_mu * get_symbol('D_μ') * psi - get_symbol('m_f') * psi_bar * psi * (1 + gamma * L / 2)
        gauge_lagrangian = -sp.Rational(1,4) * get_symbol('F_μν') * get_symbol('F^μν') * (1 + gamma * L)
        interaction_lagrangian = -get_symbol('q') * psi_bar * gamma_mu * psi * get_symbol('A_μ') * (1 + gamma * L)
        
        # <reason>chain: Build Lagrangians directly using get_symbol calls</reason>

        super().__init__(
            f"Log-Corrected QG (γ={gamma:.3f})",
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        # Matter Lagrangian: ψ̄(iγ^μD_μ - m)ψ
        # Interaction Lagrangian: qψ̄γ^μA_μψ
        self.gamma = gamma
        self.beta = gamma / 2.0  # <reason>chain: Logarithmic correction parameter for metric</reason>
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components with logarithmic quantum corrections.
        <reason>Log corrections modify the effective gravitational coupling in quantum regime.</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        beta = torch.tensor(self.beta, device=r.device, dtype=r.dtype)
        
        # <reason>Logarithmic correction arising from quantum loop effects</reason>
        # <reason>Scale r₀ chosen to be Schwarzschild radius for dimensional consistency</reason>
        epsilon = 1e-10
        log_term = torch.log(r / (rs + epsilon))
        
        # <reason>Modified metric function with logarithmic quantum correction</reason>
        # <reason>β controls strength of quantum effects, optimal at β≈0.5</reason>
        m = 1 - rs / r * (1 + beta * log_term)
        
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 