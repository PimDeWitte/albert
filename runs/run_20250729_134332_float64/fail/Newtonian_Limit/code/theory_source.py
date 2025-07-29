from physics_agent.base_theory import GravitationalTheory, Tensor
import torch
class NewtonianLimit(GravitationalTheory):
    """
    Newtonian limit of General Relativity.
    <reason>The weak-field approximation of GR, keeping only leading order terms in rs/r.</reason>
    <reason>Correctly scored high in geodesic tests but showed non-zero loss due to missing spatial curvature.</reason>
    """
    category = "classical"
    cacheable = True
    def __init__(self):
        super().__init__("Newtonian Limit")
        # <reason>Lagrangian represents weak-field limit: only time component matters</reason>
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components for the Newtonian limit.
        <reason>In weak field: g_tt ≈ -(1 - 2Φ/c²), g_rr ≈ 1, where Φ = -GM/r</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        
        # <reason>Newtonian limit: g_tt = -(1 - rs/r), g_rr = 1 (no spatial curvature)</reason>
        g_tt = -(1 - rs / r)
        g_rr = torch.ones_like(r)  # <reason>Key difference: no spatial curvature in Newtonian limit</reason>
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 