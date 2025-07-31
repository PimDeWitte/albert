import torch
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor
class AlenaTensor(GravitationalTheory):
    """
    Alena tensor gravity theory with parameter γ.
    
    A modified gravity theory that introduces tensor corrections
    to Einstein's field equations. Based on recent unification attempts.
    
    Source: https://phys.org/news/2024-12-alena-tensor-unification-physics.html
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(gamma={'min': -0.5, 'max': 0.5, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Correct default gamma=0</reason>
    preferred_params = {'gamma': 0.0}
    cacheable = True  # Enable caching for efficiency
    def __init__(self, gamma: float = 0.0):
        super().__init__(f"Alena‑Tensor‑γ={gamma:+.2f}")  # Include parameter in name
        self.gamma = gamma
        # Modified Einstein-Hilbert action with tensor corrections
        self.lagrangian = get_symbol('R') + gamma * get_symbol('R_μν') * get_symbol('R^μν')
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components for Alena tensor gravity.
        
        Tensor corrections modify the effective gravitational coupling.
        """
        rs = 2 * G_param * M_param / C_param**2
        gamma_tensor = torch.tensor(self.gamma, device=r.device, dtype=r.dtype)
        
        # Standard Schwarzschild with tensor correction
        f = 1 - rs / r
        
        # Tensor correction affects the metric
        tensor_term = gamma_tensor * (rs / r)**3
        f_modified = f + tensor_term
        
        epsilon = 1e-10
        g_tt = -f_modified
        g_rr = 1 / (f_modified + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp