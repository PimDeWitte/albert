import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory, Tensor

class Schwarzschild(GravitationalTheory):
    """
    Schwarzschild metric for a non-rotating, uncharged black hole.
    This is the simplest solution to the Einstein field equations.
    """
    
    category = "classical"  # Classical solution to Einstein equations

    def __init__(self, kappa: float = 0.0):
        """
        Initialize the Schwarzschild metric with optional kappa modification.
        """
        super().__init__(
            name=f"Schwarzschild (κ={kappa:.2e})" if kappa != 0 else "Schwarzschild",
            force_6dof_solver=False  # Always use 4D solver due to symmetry
        )
        
        # <reason>chain: Pure Einstein-Hilbert action for vacuum spacetime, with optional kappa modification</reason>
        self.lagrangian = sp.Symbol('R')
        self.kappa = kappa  # Optional modification parameter; when 0, reduces to pure GR
        
        # <reason>chain: Store init parameters for multiprocessing serialization</reason>
        self._init_params = {'kappa': kappa}

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float, 
                   t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the Schwarzschild metric components.
        
        The Schwarzschild metric in spherical coordinates is:
        ds² = -(1 - rs/r)c²dt² + (1 - rs/r)⁻¹dr² + r²(dθ² + sin²θ dφ²)
        
        In the equatorial plane (θ = π/2), this simplifies to:
        ds² = -(1 - rs/r)c²dt² + (1 - rs/r)⁻¹dr² + r²dφ²
        """
        # <reason>chain: Calculate Schwarzschild radius rs = 2GM/c²</reason>
        rs = 2 * G_param * M_param / C_param**2
        
        # <reason>chain: Define metric function f(r) = 1 - rs/r + κ/r² (κ=0 for pure GR)</reason>
        f = 1 - rs / r + self.kappa / r**2
        
        # <reason>chain: Use same epsilon as other metrics for consistency</reason>
        epsilon = 1e-10
        
        # <reason>chain: Prevent singularity at event horizon r = rs</reason>
        if torch.any(torch.abs(r - rs) < rs * epsilon):
            # If too close to horizon, add small offset
            r = torch.where(torch.abs(r - rs) < rs * epsilon,
                          rs * (1 + epsilon),
                          r)
            # Recalculate f with offset r
            f = 1 - rs / r + self.kappa / r**2
        
        # <reason>chain: Ensure f doesn't get too close to zero</reason>
        if torch.any(torch.abs(f) < epsilon):
            f = torch.where(torch.abs(f) < epsilon,
                          torch.sign(f) * epsilon,
                          f)
        
        # <reason>chain: Schwarzschild metric components in standard form</reason>
        g_tt = -f
        g_rr = 1 / f
        g_pp = r**2
        g_tp = torch.zeros_like(r)  # No frame-dragging in Schwarzschild
        
        return g_tt, g_rr, g_pp, g_tp
    
 