import sympy as sp
import torch
from physics_agent.base_theory import GravitationalTheory
from physics_agent.constants import G, c, HBAR, BOLTZMANN_CONSTANT, SOLAR_MASS

class HawkingRadiation(GravitationalTheory):
    """
    Hawking radiation theory implementing black hole thermodynamics.
    
    This theory models:
    - Hawking temperature: T_H = ħc³/(8πGMk_B)
    - Bekenstein-Hawking entropy: S = Ac³/(4Għ)
    - Black hole evaporation rate
    """
    category = "classical"  # Though quantum in nature, the metric is classical
    field = "thermodynamic"
    
    def __init__(self, name="Hawking Radiation", **kwargs):
        # Standard Schwarzschild metric as base
        r, t, M = sp.symbols('r t M')
        
        rs = 2 * G * M / c**2
        f_r = 1 - rs / r
        
        g_tt = -f_r
        g_rr = 1/f_r
        g_pp = r**2
        g_tp = 0
        
        # Lagrangian includes thermodynamic effects
        R_sym = sp.Symbol('R')  # Ricci scalar
        T_H = sp.Symbol('T_H')  # Hawking temperature
        S_BH = sp.Symbol('S_BH')  # Black hole entropy
        
        # Modified Lagrangian with quantum corrections
        lagrangian = R_sym + sp.Symbol('alpha_quantum') * T_H * S_BH
        
        super().__init__(
            name=name,
            lagrangian=lagrangian,
            **kwargs
        )
        
    def get_metric(self, r: torch.Tensor, M_param: torch.Tensor, C_param: torch.Tensor, 
                   G_param: torch.Tensor, t: torch.Tensor = None, phi: torch.Tensor = None) -> tuple:
        """Returns Schwarzschild metric with quantum corrections near horizon."""
        rs = 2 * G_param * M_param / C_param**2
        
        # Add quantum corrections near horizon
        epsilon = 1e-10  # Small parameter for quantum effects
        quantum_correction = epsilon * (HBAR * C_param / (G_param * M_param))**2
        
        f_r = 1 - rs / r + quantum_correction * (rs / r)**3
        
        g_tt = -f_r
        g_rr = 1 / (f_r + 1e-15)  # Avoid division by zero
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp
    
    def compute_hawking_temperature(self, M_param: torch.Tensor) -> torch.Tensor:
        """
        Compute Hawking temperature: T_H = ħc³/(8πGMk_B)
        
        Args:
            M_param: Black hole mass in geometric units
            
        Returns:
            Hawking temperature in Kelvin
        """
        # Convert to SI units for temperature calculation
        M_si = M_param * self.M_scale if hasattr(self, 'M_scale') else M_param * SOLAR_MASS
        
        T_H = (HBAR * c**3) / (8 * torch.pi * G * M_si * BOLTZMANN_CONSTANT)
        return T_H
    
    def compute_black_hole_entropy(self, M_param: torch.Tensor) -> torch.Tensor:
        """
        Compute Bekenstein-Hawking entropy: S = A/(4l_p²) = 4πGM²/(ħc)
        
        Args:
            M_param: Black hole mass in geometric units
            
        Returns:
            Black hole entropy in units of k_B
        """
        # Convert to SI units
        M_si = M_param * self.M_scale if hasattr(self, 'M_scale') else M_param * SOLAR_MASS
        
        # Area of event horizon: A = 4π(2GM/c²)²
        r_s = 2 * G * M_si / c**2
        area = 4 * torch.pi * r_s**2
        
        # Planck length squared
        l_p_squared = (HBAR * G) / c**3
        
        # Entropy in units of k_B
        S_BH = area / (4 * l_p_squared)
        return S_BH
    
    def compute_evaporation_rate(self, M_param: torch.Tensor) -> torch.Tensor:
        """
        Compute black hole evaporation rate: dM/dt = -ħc⁴/(15360πG²M²)
        
        Args:
            M_param: Black hole mass in geometric units
            
        Returns:
            Mass loss rate in kg/s
        """
        M_si = M_param * self.M_scale if hasattr(self, 'M_scale') else M_param * SOLAR_MASS
        
        # Stefan-Boltzmann constant
        sigma = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
        
        # Evaporation rate
        dM_dt = -(HBAR * c**4) / (15360 * torch.pi * G**2 * M_si**2)
        
        return dM_dt
    
    def compute_lifetime(self, M_param: torch.Tensor) -> torch.Tensor:
        """
        Compute black hole lifetime: τ = 5120πG²M³/(ħc⁴)
        
        Args:
            M_param: Black hole mass in geometric units
            
        Returns:
            Lifetime in seconds
        """
        M_si = M_param * self.M_scale if hasattr(self, 'M_scale') else M_param * SOLAR_MASS
        
        tau = (5120 * torch.pi * G**2 * M_si**3) / (HBAR * c**4)
        return tau
