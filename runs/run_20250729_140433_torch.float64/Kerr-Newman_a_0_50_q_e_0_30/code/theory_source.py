import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory, Tensor

class KerrNewman(GravitationalTheory):
    """
    Kerr-Newman metric for a rotating, charged black hole.
    """

    def __init__(self, M: float = 1.0, a: float = 0.5, Q: float = 0.3, G: float = 1.0, c: float = 1.0):
        """
        Kerr-Newman metric for charged rotating black holes.
        
        Parameters:
        - a: angular momentum per unit mass (0 to 1)
        - Q: dimensionless charge parameter (0 to 1)
        
        Reduces to:
        - Kerr when Q = 0
        - Reissner-Nordström when a = 0
        - Schwarzschild when a = 0 and Q = 0
        """
        # <reason>chain: Set category attribute on the instance for proper theory classification</reason>
        self.category = "classical"  # Classical solution to Einstein-Maxwell equations
        
        # <reason>chain: Using range format for parameter sweep</reason>
        sweep = dict(
            a={'min': 0.0, 'max': 0.95, 'points': 5, 'scale': 'linear'},
            q_e={'min': 0.0, 'max': 0.9, 'points': 5, 'scale': 'linear'}
        )
        # <reason>chain: Default to Schwarzschild case</reason>
        preferred_params = {'a': 0.0, 'q_e': 0.0}
        cacheable = True
        
        # Matter fields
        psi = sp.Symbol('ψ')
        psi_bar = sp.Symbol('ψ̄')
        m_f = sp.Symbol('m_f')
        gamma_mu = sp.Symbol('γ^μ')
        D_mu = sp.Symbol('D_μ')
        
        # Electromagnetic fields (exact solution includes EM)
        F_mn = sp.Symbol('F_μν')
        A_mu = sp.Symbol('A_μ')
        
        # <reason>chain: Define R as Ricci scalar symbol</reason>
        R = sp.Symbol('R')
        
        # Lagrangians for charged rotating black hole
        gravity_lagrangian = R
        matter_lagrangian = sp.I * psi_bar * gamma_mu * D_mu * psi - m_f * psi_bar * psi
        gauge_lagrangian = -sp.Rational(1,4) * F_mn * sp.Symbol('F^μν')
        interaction_lagrangian = -sp.Symbol('q') * psi_bar * gamma_mu * psi * A_mu
        
        # <reason>chain: Kerr-Newman metric is stationary and axisymmetric, so it has conserved E and Lz</reason>
        # Even with rotation (a≠0) and charge (Q≠0), we can use the efficient 4D solver
        # The off-diagonal g_tp term is handled correctly by the symmetric solver
        # <reason>chain: For non-rotating Kerr-Newman (a=0), it's Reissner-Nordström and uses 4D solver</reason>
        # <reason>chain: For rotating Kerr-Newman (a≠0), it has g_tp≠0 and needs the 6D general solver</reason>
        force_6dof = True if a != 0 else False
        
        super().__init__(
            f"Kerr-Newman (a={a:.2f}, q_e={Q:.2f})",
            force_6dof_solver=force_6dof,
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        self.a = a
        self.a_ratio = a  # For compatibility
        self.q_e = Q  # Dimensionless charge parameter
        self.Q = Q  # For compatibility
        # <reason>chain: Add symbolic parameters for Lagrangian validation</reason>
        self.q = sp.Symbol('q')  # Elementary charge
        
        # <reason>chain: Store init parameters for multiprocessing serialization</reason>
        self._init_params = {'a': a, 'q_e': Q}

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float, a: float = None, Q: float = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the Kerr-Newman metric in the equatorial plane.
        """
        a = self.a if a is None else a
        Q = self.q_e if Q is None else Q
        
        # Work in SI units throughout
        rs = 2 * G_param * M_param / C_param**2  # Schwarzschild radius
        a_val = a * rs / 2  # Convert dimensionless a to SI units
        
        r_sq = r**2
        a_sq = a_val**2
        
        # In equatorial plane (θ=π/2), Σ = r²
        Sigma = r_sq
        
        # Calculate charge radius squared using dimensionless parameter
        # r_Q = Q * rs/2, similar to how a_val = a * rs/2
        r_Q = Q * rs / 2  # Dimensionless Q scaled by rs/2
        r_Q_squared = r_Q**2
        
        # Δ = r² - rs*r + a² + r_Q²
        Delta = r_sq - rs * r + a_sq + r_Q_squared
        
        # <reason>chain: Use consistent precision with Schwarzschild for proper limiting behavior</reason>
        epsilon = 1e-10  # Same as Schwarzschild metric
        
        # Check if we're near horizons
        # For Kerr-Newman: r± = (rs/2) ± sqrt((rs/2)² - a² - r_Q²)
        discriminant = (rs/2)**2 - a_sq - r_Q_squared
        if discriminant >= 0:
            # <reason>chain: Ensure discriminant is a tensor for torch.sqrt</reason>
            discriminant_tensor = torch.tensor(discriminant) if not torch.is_tensor(discriminant) else discriminant
            r_plus = rs/2 + torch.sqrt(discriminant_tensor)
            r_minus = rs/2 - torch.sqrt(discriminant_tensor)
            
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
                Delta = r_sq - rs * r + a_sq + r_Q_squared
        
        # Ensure Delta doesn't get too close to zero
        if torch.any(torch.abs(Delta) < epsilon):
            Delta = torch.where(torch.abs(Delta) < epsilon,
                              torch.sign(Delta) * epsilon,
                              Delta)
        
        # Kerr-Newman metric components in SI units
        g_tt = -(1 - (rs * r - r_Q_squared) / Sigma)
        g_rr = Sigma / Delta
        g_pp = ((r_sq + a_sq)**2 - a_sq * Delta) / Sigma
        g_tp = -a_val * (rs * r - r_Q_squared) / Sigma
        
        return g_tt, g_rr, g_pp, g_tp
    
 