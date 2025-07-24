import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory, Tensor

class Kerr(GravitationalTheory):
    """
    Kerr metric for a rotating, uncharged black hole.
    
    This is an exact solution to Einstein's field equations and should
    be lossless when compared to itself.
    """
    category = "classical"  # Exact classical solution
    cacheable = True
    def __init__(self, a: float = 0.5):
        """
        Kerr metric for a rotating, uncharged black hole.
        
        Args:
            a (float): Spin parameter of the black hole (0 <= a <= M).
                       For simplicity in this model, 'a' is a dimensionless
                       ratio a/M, so 0 <= a <= 1. We will use it as a direct
                       scaling factor.
        """
        # When a=0, reduces to Schwarzschild (symmetric)
        # When a≠0, has frame-dragging (asymmetric)
        force_6dof = None if a == 0 else True
        
        super().__init__(
            name=f"Kerr (a={a:.2f})",
            force_6dof_solver=force_6dof
        )
        self.lagrangian = sp.sympify('R')  # Einstein-Hilbert action for General Relativity
        # We assume a is given as a fraction of M.
        # For the calculations, we need the actual spin parameter a_val = a * M.
        self.a_ratio = a
        self.a = a  # Keep both for compatibility
        
        # <reason>chain: Store init parameters for multiprocessing serialization</reason>
        self._init_params = {'a': a}

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float, a: float = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the Kerr metric components in the equatorial plane.
        """
        a = self.a_ratio if a is None else a
        
        # Work in SI units throughout
        rs = 2 * G_param * M_param / C_param**2  # Schwarzschild radius
        a_val = a * rs / 2  # Convert dimensionless a to SI units (a = J/(Mc) where J is angular momentum)
        
        # Boyer-Lindquist coordinates in equatorial plane (θ = π/2)
        r_sq = r**2
        a_sq = a_val**2
        
        # Key Kerr metric quantities
        Sigma = r_sq  # In equatorial plane: Σ = r² + a²cos²θ = r²
        Delta = r_sq - rs * r + a_sq
        
        # <reason>chain: Use consistent precision with Schwarzschild for proper limiting behavior</reason>
        epsilon = 1e-10  # Same as Schwarzschild metric
        
        # Check if we're near horizons where Delta = 0
        # Outer horizon: r+ = (rs/2) + sqrt((rs/2)² - a²)
        # Inner horizon: r- = (rs/2) - sqrt((rs/2)² - a²)
        discriminant = (rs/2)**2 - a_sq
        if discriminant >= 0:
            # discriminant is already a tensor, no need to reconstruct
            r_plus = rs/2 + torch.sqrt(discriminant)
            r_minus = rs/2 - torch.sqrt(discriminant)
            
            # If too close to either horizon, add offset
            horizon_epsilon = rs * epsilon  # Scale with rs
            if torch.any(torch.abs(r - r_plus) < horizon_epsilon) or (r_minus > 0 and torch.any(torch.abs(r - r_minus) < horizon_epsilon)):
                if torch.abs(r - r_plus) < torch.abs(r - r_minus):
                    r = r_plus + horizon_epsilon
                else:
                    r = r_minus - horizon_epsilon if r < r_minus else r_minus + horizon_epsilon
                # Recalculate with offset r
                r_sq = r**2
                Sigma = r_sq
                Delta = r_sq - rs * r + a_sq
        
        # Ensure Delta doesn't get too close to zero
        if torch.any(torch.abs(Delta) < epsilon):
            Delta = torch.where(torch.abs(Delta) < epsilon, 
                              torch.sign(Delta) * epsilon, 
                              Delta)
        
        # Kerr metric components in SI units
        g_tt = -(1 - rs * r / Sigma)
        g_rr = Sigma / Delta
        g_pp = ((r_sq + a_sq)**2 - a_sq * Delta) / Sigma
        g_tp = -rs * r * a_val / Sigma
        
        return g_tt, g_rr, g_pp, g_tp 