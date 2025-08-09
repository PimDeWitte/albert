import sympy as sp
import torch
from physics_agent.base_theory import GravitationalTheory
from physics_agent.constants import G, c, epsilon_0

class ReissnerNordstromExtended(GravitationalTheory):
    """
    Extended Reissner-Nordström solution for charged black holes.
    
    The metric is:
    ds² = -f(r)dt² + f(r)⁻¹dr² + r²(dθ² + sin²θdφ²)
    
    where f(r) = 1 - 2GM/rc² + GQ²/(4πε₀r²c⁴)
    
    This includes the full electromagnetic field tensor and
    stress-energy tensor contributions.
    """
    category = "classical"
    field = "electromagnetism"
    
    def __init__(self, name="Reissner-Nordström Extended", Q=0.5, **kwargs):
        """
        Initialize with charge parameter.
        
        Args:
            Q: Charge in geometric units (Q/√(4πε₀G) in SI)
        """
        # Symbolic variables
        r, t, M, Q_sym = sp.symbols('r t M Q')
        
        # Metric function
        rs = 2 * G * M / c**2
        # In geometric units: r_Q² = Q²G/(4πε₀c⁴)
        r_Q_squared = Q_sym**2
        
        f = 1 - rs/r + r_Q_squared/r**2
        
        # Electromagnetic 4-potential (radial electric field)
        A_t = Q_sym / r  # Electrostatic potential
        
        # Full Lagrangian including Einstein-Maxwell theory
        R_sym = sp.Symbol('R')  # Ricci scalar
        F_munu = sp.Symbol('F_munu')  # Electromagnetic field tensor
        
        # L = R/(16πG) - F_μν F^μν/(16π)
        lagrangian = R_sym - F_munu * sp.Symbol('F^munu') / 4
        
        super().__init__(
            name=name,
            lagrangian=lagrangian,
            **kwargs
        )
        self.Q = Q  # Charge parameter
        
    def get_metric(self, r: torch.Tensor, M_param: torch.Tensor, C_param: torch.Tensor, 
                   G_param: torch.Tensor, t: torch.Tensor = None, phi: torch.Tensor = None) -> tuple:
        """
        Returns Reissner-Nordström metric.
        """
        # Schwarzschild radius
        rs = 2 * G_param * M_param / C_param**2
        
        # Charge radius squared
        # In geometric units where G = c = 4πε₀ = 1, we have r_Q² = Q²
        # In SI units: r_Q² = GQ²/(4πε₀c⁴)
        # For our geometric units with variable G, c:
        Q_SI = self.Q * torch.sqrt(4 * torch.pi * epsilon_0 * G_param) / C_param**2
        r_Q_squared = G_param * Q_SI**2 / (4 * torch.pi * epsilon_0 * C_param**4)
        
        # Metric function
        f_r = 1 - rs/r + r_Q_squared/r**2
        
        # Ensure f_r doesn't go negative (naked singularity protection)
        f_r = torch.clamp(f_r, min=1e-10)
        
        g_tt = -f_r
        g_rr = 1 / f_r
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp
    
    def get_electromagnetic_field_tensor(self, r: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the electromagnetic field tensor F_μν.
        
        For Reissner-Nordström, only F_tr = -F_rt = Q/r² is non-zero.
        
        Returns:
            F_μν as a 4x4 antisymmetric tensor
        """
        batch_size = r.shape[0]
        F = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=r.device)
        
        # Electric field component E_r = Q/r²
        E_r = self.Q / r**2
        
        # F_tr = E_r/c (in geometric units where c=1)
        F[:, 0, 1] = E_r   # F_tr
        F[:, 1, 0] = -E_r  # F_rt = -F_tr
        
        return F
    
    def get_electromagnetic_4potential(self, r: torch.Tensor) -> torch.Tensor:
        """
        Get the electromagnetic 4-potential A_μ.
        
        For Reissner-Nordström: A_t = Q/r, A_r = A_θ = A_φ = 0
        
        Returns:
            A_μ as a 4-vector
        """
        batch_size = r.shape[0]
        A = torch.zeros(batch_size, 4, dtype=self.dtype, device=r.device)
        
        # Electrostatic potential
        A[:, 0] = self.Q / r  # A_t
        
        return A
    
    def get_electromagnetic_stress_energy(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute electromagnetic stress-energy tensor.
        
        T_μν^(EM) = (1/4π)[F_μα F_ν^α - (1/4)g_μν F_αβ F^αβ]
        
        Returns:
            Electromagnetic stress-energy tensor
        """
        batch_size = r.shape[0]
        T_EM = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=r.device)
        
        # For Reissner-Nordström, the electromagnetic energy density is
        # ρ_EM = Q²/(8πr⁴) and radial pressure p_r = -ρ_EM
        
        rho_EM = self.Q**2 / (8 * torch.pi * r**4)
        
        # Diagonal components
        T_EM[:, 0, 0] = rho_EM    # T_tt (energy density)
        T_EM[:, 1, 1] = -rho_EM   # T_rr (radial pressure)
        T_EM[:, 2, 2] = rho_EM    # T_θθ (tangential pressure)
        T_EM[:, 3, 3] = rho_EM    # T_φφ (tangential pressure)
        
        return T_EM
    
    def compute_horizons(self, M_param: torch.Tensor) -> tuple:
        """
        Compute inner and outer horizons.
        
        Horizons occur at r_± = GM/c² ± √[(GM/c²)² - GQ²/(4πε₀c⁴)]
        
        Returns:
            (r_plus, r_minus) - outer and inner horizons
        """
        # In geometric units
        M = M_param
        Q = self.Q
        
        # Check for naked singularity
        discriminant = M**2 - Q**2
        
        if discriminant < 0:
            # Naked singularity - no horizons
            return None, None
        
        sqrt_disc = torch.sqrt(discriminant)
        r_plus = M + sqrt_disc
        r_minus = M - sqrt_disc
        
        return r_plus, r_minus
    
    def is_extremal(self, M_param: torch.Tensor) -> torch.Tensor:
        """
        Check if the black hole is extremal (Q = M in geometric units).
        """
        return torch.abs(self.Q - M_param) < 1e-10
    
    def is_naked_singularity(self, M_param: torch.Tensor) -> torch.Tensor:
        """
        Check if we have a naked singularity (Q > M in geometric units).
        """
        return self.Q > M_param
