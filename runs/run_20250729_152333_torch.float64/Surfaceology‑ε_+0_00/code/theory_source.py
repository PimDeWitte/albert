import torch
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor
class Surfaceology(GravitationalTheory):
    """
    Surfaceology theory - gravity emerges from surface interactions.
    
    This theory proposes that gravitational effects arise from the
    thermodynamics of surfaces in spacetime.
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(epsilon={'min': -0.1, 'max': 0.1, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Correct default epsilon=0</reason>
    preferred_params = {'epsilon': 0.0}
    cacheable = True  # Enable caching for efficiency
    def __init__(self, epsilon: float = 0.0):
        super().__init__(f"Surfaceology‑ε={epsilon:+.2f}")  # Include parameter in name
        self.epsilon = epsilon
        self.lagrangian = get_symbol('R') + epsilon * get_symbol('K')  # K is surface term
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components for surfaceology.
        
        Surface effects modify the metric near the horizon scale.
        """
        rs = 2 * G_param * M_param / C_param**2
        epsilon_tensor = torch.tensor(self.epsilon, device=r.device, dtype=r.dtype)
        
        # Standard term with surface correction
        f_standard = 1 - rs / r
        
        # Surface correction near horizon
        surface_term = epsilon_tensor * (rs / r)**3
        f = f_standard - surface_term
        
        epsilon_val = 1e-10
        g_tt = -f
        g_rr = 1 / (f + epsilon_val)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 