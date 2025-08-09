import sympy as sp
import torch
from physics_agent.base_theory import GravitationalTheory
from physics_agent.constants import G, c, HBAR, BOLTZMANN_CONSTANT

class UnruhEffect(GravitationalTheory):
    """
    Unruh effect theory modeling thermal radiation seen by accelerated observers.
    
    The Unruh temperature is: T_U = ħa/(2πck_B)
    where a is the proper acceleration.
    
    This theory modifies spacetime to include thermal effects for accelerated observers.
    """
    category = "quantum"
    field = "thermodynamic"
    
    def __init__(self, name="Unruh Effect", alpha_unruh=1.0, **kwargs):
        # Rindler coordinates for uniformly accelerated observer
        # ds² = -α²x²dt² + dx² + dy² + dz²
        x, t, alpha = sp.symbols('x t alpha')
        
        # Metric components in Rindler coordinates
        g_tt = -alpha**2 * x**2
        g_xx = 1
        g_yy = 1
        g_zz = 1
        
        # Lagrangian with Unruh temperature effects
        R_sym = sp.Symbol('R')
        T_U = sp.Symbol('T_U')  # Unruh temperature
        
        # Modified Lagrangian includes thermal corrections
        lagrangian = R_sym + sp.Symbol('alpha_unruh') * T_U**2
        
        super().__init__(
            name=name,
            lagrangian=lagrangian,
            enable_quantum=True,
            **kwargs
        )
        self.alpha_unruh = alpha_unruh
        
    def get_metric(self, r: torch.Tensor, M_param: torch.Tensor, C_param: torch.Tensor, 
                   G_param: torch.Tensor, t: torch.Tensor = None, phi: torch.Tensor = None) -> tuple:
        """
        Returns metric including Unruh effect corrections.
        
        For compatibility with the framework, we use Schwarzschild as base
        and add acceleration-dependent corrections.
        """
        # Base Schwarzschild metric
        rs = 2 * G_param * M_param / C_param**2
        f_r = 1 - rs / r
        
        # Compute local acceleration at distance r
        # a = GM/r² in Newtonian approximation
        a_local = G_param * M_param / (r**2 * C_param**2)
        
        # Unruh temperature correction factor
        # Small correction proportional to acceleration
        unruh_factor = self.alpha_unruh * 1e-20 * a_local  # Very small effect
        
        # Modified metric with Unruh corrections
        g_tt = -f_r * (1 + unruh_factor)
        g_rr = 1 / (f_r + 1e-15)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp
    
    def compute_unruh_temperature(self, acceleration: torch.Tensor) -> torch.Tensor:
        """
        Compute Unruh temperature: T_U = ħa/(2πck_B)
        
        Args:
            acceleration: Proper acceleration in m/s²
            
        Returns:
            Unruh temperature in Kelvin
        """
        T_U = (HBAR * acceleration) / (2 * torch.pi * c * BOLTZMANN_CONSTANT)
        return T_U
    
    def compute_acceleration_at_radius(self, r: torch.Tensor, M_param: torch.Tensor) -> torch.Tensor:
        """
        Compute proper acceleration for a stationary observer at radius r.
        
        For Schwarzschild: a = GM/(r²√(1-2GM/rc²))
        
        Args:
            r: Radius in geometric units
            M_param: Mass in geometric units
            
        Returns:
            Proper acceleration in m/s²
        """
        # Convert to SI units
        r_si = r * self.length_scale if hasattr(self, 'length_scale') else r * c**2 / G
        M_si = M_param * self.M_scale if hasattr(self, 'M_scale') else M_param * c**3 / G
        
        # Schwarzschild radius
        rs = 2 * G * M_si / c**2
        
        # Proper acceleration
        if torch.any(r_si <= rs):
            # Handle case inside event horizon
            a = torch.where(
                r_si > rs,
                (G * M_si / r_si**2) / torch.sqrt(1 - rs / r_si),
                torch.tensor(float('inf'))
            )
        else:
            a = (G * M_si / r_si**2) / torch.sqrt(1 - rs / r_si)
        
        return a
    
    def compute_horizon_acceleration(self, M_param: torch.Tensor) -> torch.Tensor:
        """
        Compute surface gravity (acceleration at horizon).
        
        For Schwarzschild: κ = c⁴/(4GM)
        
        Args:
            M_param: Black hole mass in geometric units
            
        Returns:
            Surface gravity in m/s²
        """
        M_si = M_param * self.M_scale if hasattr(self, 'M_scale') else M_param * c**3 / G
        
        kappa = c**4 / (4 * G * M_si)
        return kappa
    
    def compute_hawking_temperature_from_unruh(self, M_param: torch.Tensor) -> torch.Tensor:
        """
        Derive Hawking temperature using Unruh effect at the horizon.
        
        This shows the deep connection between the two effects.
        
        Args:
            M_param: Black hole mass in geometric units
            
        Returns:
            Temperature in Kelvin
        """
        # Surface gravity at horizon
        kappa = self.compute_horizon_acceleration(M_param)
        
        # Unruh temperature for this acceleration
        T_H = self.compute_unruh_temperature(kappa)
        
        return T_H
