import sympy as sp
import torch
from physics_agent.base_theory import GravitationalTheory
from physics_agent.constants import G, c

class LambdaCDM(GravitationalTheory):
    """
    ΛCDM (Lambda Cold Dark Matter) cosmological model.
    
    The standard model of cosmology with:
    - Dark energy (cosmological constant Λ)
    - Cold dark matter
    - Ordinary matter
    - Radiation
    
    Uses the FLRW metric for a homogeneous, isotropic universe.
    """
    category = "classical"
    field = "cosmology"
    
    def __init__(self, name="ΛCDM Model", 
                 Omega_m=0.3, Omega_Lambda=0.7, Omega_r=0.0, h=0.7, **kwargs):
        """
        Initialize ΛCDM model with cosmological parameters.
        
        Args:
            Omega_m: Matter density parameter (dark + baryonic)
            Omega_Lambda: Dark energy density parameter
            Omega_r: Radiation density parameter
            h: Reduced Hubble constant (H0 = 100h km/s/Mpc)
        """
        # FLRW metric: ds² = -dt² + a(t)²[dr²/(1-kr²) + r²(dθ² + sin²θdφ²)]
        t, r, a, k = sp.symbols('t r a k')
        
        # Scale factor satisfies Friedmann equations
        # (ȧ/a)² = 8πG/3 * ρ - k/a²
        # ä/a = -4πG/3 * (ρ + 3p)
        
        # Lagrangian for FLRW cosmology
        R_sym = sp.Symbol('R')  # Ricci scalar
        rho = sp.Symbol('rho')  # Energy density
        
        # Einstein-Hilbert action with matter
        lagrangian = R_sym - 8 * sp.pi * G / c**4 * rho
        
        super().__init__(
            name=name,
            lagrangian=lagrangian,
            **kwargs
        )
        
        # Store cosmological parameters
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        self.Omega_r = Omega_r
        self.Omega_k = 1 - Omega_m - Omega_Lambda - Omega_r  # Curvature
        self.h = h
        self.H0 = 100 * h  # km/s/Mpc
        
    def get_metric(self, r: torch.Tensor, M_param: torch.Tensor, C_param: torch.Tensor, 
                   G_param: torch.Tensor, t: torch.Tensor = None, phi: torch.Tensor = None) -> tuple:
        """
        Returns metric for ΛCDM cosmology.
        
        For compatibility with framework, we return a spherically symmetric
        metric that approximates cosmological effects locally.
        """
        # Convert Hubble constant to geometric units
        H0_SI = self.H0 * 1e3 / 3.086e22  # Convert km/s/Mpc to 1/s
        H0_geom = H0_SI * C_param  # Convert to geometric units
        
        # Cosmological constant in geometric units
        Lambda = 3 * self.Omega_Lambda * H0_geom**2
        
        # For local physics, use Schwarzschild-de Sitter metric
        rs = 2 * G_param * M_param / C_param**2
        
        # Metric function with cosmological constant
        f_r = 1 - rs/r - Lambda * r**2 / 3
        
        g_tt = -f_r
        g_rr = 1 / (f_r + 1e-15)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp
    
    def compute_scale_factor(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute scale factor as function of redshift: a = 1/(1+z)
        
        Args:
            z: Redshift
            
        Returns:
            Scale factor a(z)
        """
        return 1 / (1 + z)
    
    def compute_hubble_parameter(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute Hubble parameter as function of redshift.
        
        H(z) = H0 * E(z)
        where E(z) = √[Ωm(1+z)³ + Ωr(1+z)⁴ + Ωk(1+z)² + ΩΛ]
        
        Args:
            z: Redshift
            
        Returns:
            H(z) in km/s/Mpc
        """
        E_squared = (
            self.Omega_m * (1 + z)**3 +
            self.Omega_r * (1 + z)**4 +
            self.Omega_k * (1 + z)**2 +
            self.Omega_Lambda
        )
        
        return self.H0 * torch.sqrt(E_squared)
    
    def compute_luminosity_distance(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute luminosity distance.
        
        For flat universe (Ωk = 0):
        d_L = (c/H0) * (1+z) * ∫[0,z] dz'/E(z')
        
        Args:
            z: Redshift
            
        Returns:
            Luminosity distance in Mpc
        """
        # For simplicity, use approximation valid for z << 1
        # d_L ≈ cz/H0 * [1 + (1-q0)z/2]
        # where q0 = Ωm/2 - ΩΛ is deceleration parameter
        
        c_km_s = c / 1000  # Speed of light in km/s
        q0 = self.Omega_m / 2 - self.Omega_Lambda
        
        d_L = (c_km_s * z / self.H0) * (1 + (1 - q0) * z / 2)
        
        # For higher accuracy, would need numerical integration
        return d_L
    
    def compute_age_of_universe(self) -> torch.Tensor:
        """
        Compute the age of the universe.
        
        t_0 = (1/H0) * ∫[0,∞] dz/[(1+z)E(z)]
        
        Returns:
            Age in Gyr
        """
        # For ΛCDM with our parameters, use fitting formula
        # t_0 ≈ (2/3H0) * 1/√ΩΛ * ln[(1+√ΩΛ)/√Ωm]
        
        if self.Omega_Lambda > 0:
            factor = (2/3) * 1/torch.sqrt(self.Omega_Lambda) * torch.log(
                (1 + torch.sqrt(self.Omega_Lambda)) / torch.sqrt(self.Omega_m)
            )
        else:
            # Matter-only universe
            factor = 2/3 / (1 - self.Omega_m + self.Omega_m**0.3)
        
        # Convert from 1/H0 to Gyr
        H0_Gyr = self.H0 * 1.023e-3  # H0 in Gyr^-1
        t_0 = factor / H0_Gyr
        
        return t_0
    
    def compute_particle_horizon(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the comoving particle horizon.
        
        χ_p(z) = c ∫[z,∞] dz'/[H(z')]
        
        Args:
            z: Redshift
            
        Returns:
            Comoving distance to particle horizon in Mpc
        """
        # For z >> 1, dominated by matter
        # χ_p ≈ 2c/(H0√Ωm) * [1 - 1/√(1+z)]
        
        c_km_s = c / 1000
        if z > 10:
            chi_p = (2 * c_km_s / (self.H0 * torch.sqrt(self.Omega_m))) * (
                1 - 1 / torch.sqrt(1 + z)
            )
        else:
            # Use simpler approximation
            chi_p = c_km_s * (1 + z) / (self.H0 * torch.sqrt(self.Omega_m))
        
        return chi_p
    
    def is_accelerating(self, z: torch.Tensor) -> torch.Tensor:
        """
        Check if universe is accelerating at given redshift.
        
        Acceleration occurs when ä > 0, which happens when:
        ΩΛ > Ωm(1+z)³/2
        
        Args:
            z: Redshift
            
        Returns:
            Boolean tensor indicating acceleration
        """
        return self.Omega_Lambda > self.Omega_m * (1 + z)**3 / 2
