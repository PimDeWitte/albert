import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor
import torch
class PhaseTransition(GravitationalTheory):
    """
    Quantum gravity with phase transitions in field structure.
    <reason>Spacetime undergoes phase transitions at critical scales.</reason>
    <reason>Parameter T_c controls the critical transition temperature/energy.</reason>
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(T_c={'min': 0.1, 'max': 2.0, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Default T_c=1.0 at GUT scale</reason>
    preferred_params = {'T_c': 1.1}
    cacheable = True
    def __init__(self, T_c: float = 1.0):
        # <reason>chain: Define quantum field components with phase transitions</reason>
        T = get_symbol('T')  # Temperature/energy scale
        
        # Matter fields with phase-dependent masses
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        gamma_mu = get_symbol('γ^μ')
        
        # Gauge fields
        
        # Phase transition function
        theta = sp.tanh((T - T_c) / get_symbol('δT'))  # Smooth transition
        
        # Effective masses depend on phase
        
        # Lagrangians with phase transition
        gravity_lagrangian = get_symbol('R') * (1 + get_symbol('α') * theta)
        matter_lagrangian = (sp.I * psi_bar * gamma_mu * get_symbol('D_μ') * psi - get_symbol('m_0') * psi_bar * psi + 
                            get_symbol('∂_μφ') * get_symbol('∂^μφ') - get_symbol('V(φ,T)'))
        gauge_lagrangian = -sp.Rational(1,4) * get_symbol('F_μν') * get_symbol('F^μν') * (1 + get_symbol('β') * theta)
        interaction_lagrangian = -get_symbol('q') * psi_bar * gamma_mu * psi * get_symbol('A_μ') + get_symbol('λ') * psi_bar * psi * get_symbol('φ')
        
        # <reason>chain: Build Lagrangians directly using get_symbol calls</reason>

        super().__init__(
            f"Phase Transition QG (T_c={T_c:.1f})",
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        self.T_c = T_c
        self.r_c_rs = 3.0  # <reason>chain: Critical radius for phase transition in units of Schwarzschild radius</reason>
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components with gravitational phase transition.
        <reason>Smooth transition between two gravitational phases at r_c.</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        r_c = torch.tensor(self.r_c_rs, device=r.device, dtype=r.dtype) * rs
        
        # <reason>Phase transition function: smooth step using tanh</reason>
        # <reason>Width of transition region ~ 0.5 RS for smooth physics</reason>
        transition_width = 0.5 * rs
        phase_param = torch.tanh((r - r_c) / transition_width)
        
        # <reason>Two phases with different gravitational behavior</reason>
        # <reason>Strong phase (r < r_c): enhanced gravity</reason>
        # <reason>Weak phase (r > r_c): standard gravity</reason>
        m_strong = 1 - 1.5 * rs / r  # Enhanced gravity
        m_weak = 1 - rs / r  # Standard gravity
        
        # <reason>Smooth interpolation between phases</reason>
        m = 0.5 * ((1 - phase_param) * m_strong + (1 + phase_param) * m_weak)
        
        epsilon = 1e-10
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 