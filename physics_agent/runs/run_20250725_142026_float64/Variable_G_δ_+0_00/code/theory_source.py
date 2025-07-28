from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor
import torch
class VariableG(GravitationalTheory):
    """
    Theory with spatially varying gravitational constant G(r).
    <reason>Tests possibility that gravitational "constant" varies with distance from mass.</reason>
    <reason>δ parameter controls strength of variation relative to standard G.</reason>
    """
    category = "classical"
    # <reason>chain: Update to range format</reason>
    sweep = dict(delta={'min': -0.5, 'max': 0.5, 'points': 11, 'scale': 'linear'})
    # <reason>chain: Correct default delta=0 for constant G</reason>
    preferred_params = {'delta': 0.0}
    cacheable = True
    def __init__(self, delta: float = 0.0):
        super().__init__(f"Variable G (δ={delta:+.2f})")
        self.delta = delta
        # <reason>Lagrangian modified by position-dependent coupling</reason>
        self.lagrangian = get_symbol('R') / (get_symbol('G') * (1 + delta * get_symbol('r') / get_symbol('r_0')))
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components with varying gravitational coupling.
        <reason>G varies with distance: G(r) = G₀(1 + δr/r₀) where r₀ is scale length.</reason>
        """
        delta = torch.tensor(self.delta, device=r.device, dtype=r.dtype)
        
        # <reason>Scale length r₀ chosen as initial Schwarzschild radius for consistency</reason>
        rs_0 = 2 * G_param * M_param / C_param**2
        
        # <reason>Variable G: increases/decreases with distance based on δ</reason>
        G_eff = G_param * (1 + delta * r / rs_0)
        
        # <reason>Effective Schwarzschild radius varies with position</reason>
        rs_eff = 2 * G_eff * M_param / C_param**2
        
        # <reason>Standard Schwarzschild form but with position-dependent rs</reason>
        m = 1 - rs_eff / r
        
        epsilon = 1e-10
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 