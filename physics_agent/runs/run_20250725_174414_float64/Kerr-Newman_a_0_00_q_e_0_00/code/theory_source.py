import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory, Tensor

class KerrNewman(GravitationalTheory):
    """
    Kerr-Newman metric for charged rotating black holes.
    
    Parameters:
    - a: angular momentum per unit mass (0 to 1)
    - q_e: dimensionless charge parameter (0 to 1)
    
    Reduces to:
    - Kerr when q_e = 0
    - Reissner-Nordström when a = 0
    - Schwarzschild when a = 0 and q_e = 0
    """
    category = "classical"  # Classical solution to Einstein-Maxwell equations
    # <reason>chain: Using range format for parameter sweep</reason>
    sweep = dict(
        a={'min': 0.0, 'max': 0.95, 'points': 5, 'scale': 'linear'},
        q_e={'min': 0.0, 'max': 0.9, 'points': 5, 'scale': 'linear'}
    )
    # <reason>chain: Default to Schwarzschild case</reason>
    preferred_params = {'a': 0.0, 'q_e': 0.0}
    cacheable = True
    
    def __init__(self, a: float = 0.0, q_e: float = 0.0):
        # <reason>chain: Define quantum field components for Kerr-Newman unified metric</reason>
        R = sp.Symbol('R')
        
        # Matter fields
        psi = sp.Symbol('ψ')
        psi_bar = sp.Symbol('ψ̄')
        m_f = sp.Symbol('m_f')
        gamma_mu = sp.Symbol('γ^μ')
        D_mu = sp.Symbol('D_μ')
        
        # Electromagnetic fields (exact solution includes EM)
        F_mn = sp.Symbol('F_μν')
        A_mu = sp.Symbol('A_μ')
        
        # Lagrangians for charged rotating black hole
        gravity_lagrangian = R
        matter_lagrangian = sp.I * psi_bar * gamma_mu * D_mu * psi - m_f * psi_bar * psi
        gauge_lagrangian = -sp.Rational(1,4) * F_mn * sp.Symbol('F^μν')
        interaction_lagrangian = -sp.Symbol('q') * psi_bar * gamma_mu * psi * A_mu
        
        # When a=0, reduces to Reissner-Nordström (symmetric)
        # When a≠0, has frame-dragging (asymmetric)
        force_6dof = None if a == 0 else True
        
        super().__init__(
            f"Kerr-Newman (a={a:.2f}, q_e={q_e:.2f})",
            force_6dof_solver=force_6dof,
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        self.a = a
        self.a_ratio = a  # For compatibility
        self.q_e = q_e  # Dimensionless charge parameter
        self.Q = q_e  # For compatibility
        # <reason>chain: Add symbolic parameters for Lagrangian validation</reason>
        self.q = sp.Symbol('q')  # Elementary charge
        
        # <reason>chain: Store init parameters for multiprocessing serialization</reason>
        self._init_params = {'a': a, 'q_e': q_e}

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
    
    def get_cache_tag(self, N_STEPS, precision_tag, r0_tag):
        """Include both a and Q parameters in cache tag."""
        base = self.name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(".", "_").replace(",", "_")
        a_tag = f"a{self.a:.2f}".replace(".", "_")
        Q_tag = f"Q{self.q_e:.0e}".replace("+", "p").replace("-", "m")
        return f"{base}_{a_tag}_{Q_tag}_{N_STEPS}_{precision_tag}_r{r0_tag}" 