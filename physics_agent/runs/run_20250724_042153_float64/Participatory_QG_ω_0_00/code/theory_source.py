import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor
import torch
class Participatory(GravitationalTheory):
    """
    Participatory quantum gravity where observers affect spacetime.
    <reason>Based on Wheeler's participatory universe concept.</reason>
    <reason>Parameter omega controls observer-spacetime coupling strength.</reason>
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(omega={'min': 0.0, 'max': 1.0, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Default ω=0 corresponds to observer-independent physics</reason>
    preferred_params = {'omega': 1e-5}
    cacheable = True
    def __init__(self, omega: float = 0.0):
        # <reason>chain: Define quantum field components with observer interaction</reason>
        
        # Matter fields (observed system)
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        gamma_mu = get_symbol('γ^μ')
        
        # Observer field
        chi = get_symbol('χ')  # Observer wavefunction
        chi_dag = get_symbol('χ†')
        
        # Measurement operator
        M = get_symbol('M')
        
        # Gauge fields
        
        # Lagrangians with participatory terms
        gravity_lagrangian = get_symbol('R') * (1 + omega * chi_dag * M * chi)
        matter_lagrangian = sp.I * psi_bar * gamma_mu * get_symbol('D_μ') * psi - get_symbol('m_f') * psi_bar * psi + omega * chi_dag * M * psi_bar * psi
        gauge_lagrangian = -sp.Rational(1,4) * get_symbol('F_μν') * get_symbol('F^μν') * (1 + omega * get_symbol('<M>'))
        interaction_lagrangian = -get_symbol('q') * psi_bar * gamma_mu * psi * get_symbol('A_μ') * (1 + omega * chi_dag * chi)
        
        # <reason>chain: Build Lagrangians directly using get_symbol calls</reason>

        super().__init__(
            f"Participatory QG (ω={omega:.2f})",
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        # Matter Lagrangian: ψ̄(iγ^μD_μ - m)ψ
        # Interaction Lagrangian: qψ̄γ^μA_μψ
        self.omega = omega
        self.E_obs = 1.0  # <reason>chain: Observer energy scale</reason>
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components with observer participation effects.
        <reason>Metric interpolates between GR and flat space based on observation scale.</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        E_obs = torch.tensor(self.E_obs, device=r.device, dtype=r.dtype)
        
        # <reason>Planck energy scale for comparison</reason>
        E_planck = torch.tensor(1.22e19, device=r.device, dtype=r.dtype)  # GeV
        E_obs_GeV = E_obs / 1e9  # Convert eV to GeV
        
        # <reason>Observation strength: how much reality "crystallizes" from observation</reason>
        # <reason>At low E_obs, spacetime is more "fuzzy" (approaches flat)</reason>
        observation_strength = 1 - torch.exp(-E_obs_GeV / E_planck)
        
        # <reason>Interpolate between flat space and Schwarzschild based on observation</reason>
        # <reason>Paper: ~92% GR + 8% flat causes catastrophic orbital failure</reason>
        m_schwarzschild = 1 - rs / r
        m = observation_strength * m_schwarzschild + (1 - observation_strength) * 1.0
        
        epsilon = 1e-10
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 