import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory, Tensor

class Kerr(GravitationalTheory):
    """
    Kerr metric for a rotating, uncharged black hole.
    <reason>chain: The Kerr metric describes the geometry of spacetime around a rotating, uncharged mass.</reason>
    """

    def __init__(self, a: float = 0.5):
        """
        Kerr metric for rotating black holes (Q=0).
        
        Parameters:
        - a: angular momentum per unit mass (0 to 1)
        
        Reduces to:
        - Schwarzschild when a = 0
        """
        # <reason>chain: Define R as Ricci scalar symbol</reason>
        R = sp.Symbol('R')
        
        # <reason>chain: Kerr metric is stationary and axisymmetric, so it has conserved E and Lz</reason>
        # Even with rotation (a≠0), we can use the efficient 4D solver with conserved quantities
        # The off-diagonal g_tp term is handled correctly by the symmetric solver
        # <reason>chain: The is_symmetric property already handles this correctly - no need to force 6D</reason>
        # <reason>chain: We now have a specialized KerrGeodesicRK4Solver that properly handles g_tp terms</reason>
        force_6dof = False  # Always use 4D solver for Kerr
        
        super().__init__(
            f"Kerr (a={a:.2f})",
            force_6dof_solver=force_6dof,
            lagrangian=R
        )
        self.a = a
        self.a_ratio = a  # For compatibility
        # <reason>chain: Store init parameters for multiprocessing serialization</reason>
        self._init_params = {'a': a}

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float, 
                   t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the Kerr metric in the equatorial plane (Q=0).
        <reason>chain: The Kerr metric is the Q=0 limit of Kerr-Newman.</reason>
        """
        # Work in SI units throughout
        rs = 2 * G_param * M_param / C_param**2  # <reason>chain: Schwarzschild radius</reason>
        a_val = self.a * rs / 2  # <reason>chain: Convert dimensionless a to SI units</reason>
        
        r_sq = r**2
        a_sq = a_val**2
        
        # In equatorial plane (θ=π/2), Σ = r²
        Sigma = r_sq
        
        # Δ = r² - rs*r + a²
        Delta = r_sq - rs * r + a_sq
        
        # <reason>chain: Use consistent precision with Schwarzschild for proper limiting behavior</reason>
        epsilon = 1e-10  # Same as Schwarzschild metric
        
        # Check if we're near horizons
        # For Kerr: r± = (rs/2) ± sqrt((rs/2)² - a²)
        discriminant = (rs/2)**2 - a_sq
        if discriminant >= 0:
            r_plus = rs/2 + torch.sqrt(torch.tensor(discriminant) if not torch.is_tensor(discriminant) else discriminant)
            r_minus = rs/2 - torch.sqrt(torch.tensor(discriminant) if not torch.is_tensor(discriminant) else discriminant)
            
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
        
        # Kerr metric components in SI units (Q=0)
        g_tt = -(1 - (rs * r) / Sigma)  # <reason>chain: Q=0, so no charge term</reason>
        g_rr = Sigma / Delta
        g_pp = ((r_sq + a_sq)**2 - a_sq * Delta) / Sigma
        g_tp = -a_val * (rs * r) / Sigma  # <reason>chain: Q=0, so no charge term</reason>
        
        return g_tt, g_rr, g_pp, g_tp 