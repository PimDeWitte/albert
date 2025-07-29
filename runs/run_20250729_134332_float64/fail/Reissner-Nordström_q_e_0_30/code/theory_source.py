import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory, Tensor

class ReissnerNordstrom(GravitationalTheory):
    """
    Reissner-Nordström metric for a non-rotating, charged black hole.
    
    This is an exact solution to the Einstein-Maxwell equations representing
    a static, spherically symmetric, electrically charged black hole.
    """

    def __init__(self, M: float = 1.0, q_e: float = 0.3, G: float = 1.0, c: float = 1.0):
        """
        Reissner-Nordström metric for charged non-rotating black holes.
        
        Parameters:
        - q_e: dimensionless charge parameter (0 to 1)
        
        Reduces to:
        - Schwarzschild when q_e = 0
        """
        category = "classical"  # Classical solution to Einstein-Maxwell equations
        
        # Matter fields (for completeness, though this is a vacuum solution)
        psi = sp.Symbol('ψ')
        psi_bar = sp.Symbol('ψ̄')
        m_f = sp.Symbol('m_f')
        gamma_mu = sp.Symbol('γ^μ')
        D_mu = sp.Symbol('D_μ')
        
        # Electromagnetic fields
        F_mn = sp.Symbol('F_μν')
        A_mu = sp.Symbol('A_μ')
        
        # <reason>chain: Define R as Ricci scalar symbol</reason>
        R = sp.Symbol('R')
        
        # Lagrangians for charged black hole
        gravity_lagrangian = R
        matter_lagrangian = sp.I * psi_bar * gamma_mu * D_mu * psi - m_f * psi_bar * psi
        gauge_lagrangian = -sp.Rational(1,4) * F_mn * sp.Symbol('F^μν')
        interaction_lagrangian = -sp.Symbol('q') * psi_bar * gamma_mu * psi * A_mu
        
        super().__init__(
            f"Reissner-Nordström (q_e={q_e:.2f})",
            force_6dof_solver=False,  # Spherically symmetric - 4DOF sufficient
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        self.q_e = q_e  # Dimensionless charge parameter
        self.Q = q_e  # For compatibility
        # <reason>chain: Add symbolic parameters for Lagrangian validation</reason>
        self.q = sp.Symbol('q')  # Elementary charge
        
        # <reason>chain: Store init parameters for multiprocessing serialization</reason>
        self._init_params = {'q_e': q_e}

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float, 
                   t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the Reissner-Nordström metric components.
        
        The metric is:
        ds² = -f(r)dt² + f(r)⁻¹dr² + r²dΩ²
        where f(r) = 1 - rs/r + r_Q²/r²
        """
        # Work in SI units throughout
        rs = 2 * G_param * M_param / C_param**2  # Schwarzschild radius
        
        # Calculate charge radius using dimensionless parameter
        # r_Q = q_e * rs/2, similar to Kerr-Newman
        r_Q = self.q_e * rs / 2  # Dimensionless q_e scaled by rs/2
        r_Q_squared = r_Q**2
        
        # Metric function f(r) = 1 - rs/r + r_Q²/r²
        f = 1 - rs / r + r_Q_squared / r**2
        
        # <reason>chain: Use consistent precision with Schwarzschild for proper limiting behavior</reason>
        epsilon = 1e-10  # Same as Schwarzschild metric
        
        # Check if we're near horizons
        # For Reissner-Nordström: r± = (rs/2) ± sqrt((rs/2)² - r_Q²)
        discriminant = (rs/2)**2 - r_Q_squared
        
        # Ensure discriminant is a tensor for torch.sqrt
        if not isinstance(discriminant, torch.Tensor):
            discriminant = torch.tensor(discriminant, device=r.device, dtype=r.dtype)
            
        if torch.all(discriminant >= 0):
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
                f = 1 - rs / r + r_Q_squared / r**2
        
        # Ensure f doesn't get too close to zero
        if torch.any(torch.abs(f) < epsilon):
            f = torch.where(torch.abs(f) < epsilon,
                           torch.sign(f) * epsilon,
                           f)
        
        # Reissner-Nordström metric components in SI units
        g_tt = -f
        g_rr = 1 / f
        g_pp = r**2
        g_tp = torch.zeros_like(r)  # No frame-dragging (non-rotating)
        
        return g_tt, g_rr, g_pp, g_tp
    
 