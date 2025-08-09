import sympy as sp
import torch
from physics_agent.base_theory import GravitationalTheory
from physics_agent.constants import G, c

class PerfectFluid(GravitationalTheory):
    """
    Perfect fluid in curved spacetime.
    
    The stress-energy tensor for a perfect fluid is:
    T^μν = (ρ + p/c²)u^μu^ν + pg^μν
    
    where:
    - ρ is the energy density
    - p is the pressure
    - u^μ is the 4-velocity
    
    This theory models spacetime sourced by a perfect fluid distribution.
    """
    category = "classical"
    field = "fluid_dynamics"
    
    def __init__(self, name="Perfect Fluid", equation_of_state='dust', **kwargs):
        """
        Initialize perfect fluid theory.
        
        Args:
            equation_of_state: Type of fluid ('dust', 'radiation', 'stiff', or custom w)
        """
        # Fluid parameters
        rho, p, u = sp.symbols('rho p u')
        
        # Einstein field equations with fluid source
        R_sym = sp.Symbol('R')  # Ricci scalar
        T_sym = sp.Symbol('T')  # Trace of stress-energy tensor
        
        # Modified Einstein-Hilbert action with matter
        # L = R - 8πG/c⁴ L_matter
        L_matter = -rho * sp.sqrt(-sp.Symbol('g'))  # Fluid Lagrangian density
        
        lagrangian = R_sym + 8 * sp.pi * G / c**4 * L_matter
        
        super().__init__(
            name=name,
            lagrangian=lagrangian,
            **kwargs
        )
        
        # Set equation of state parameter w = p/ρc²
        if equation_of_state == 'dust':
            self.w = 0.0  # Pressureless matter
        elif equation_of_state == 'radiation':
            self.w = 1/3  # Radiation/relativistic matter
        elif equation_of_state == 'stiff':
            self.w = 1.0  # Stiff equation of state
        elif isinstance(equation_of_state, (int, float)):
            self.w = float(equation_of_state)  # Custom value
        else:
            self.w = 0.0  # Default to dust
            
        self.equation_of_state = equation_of_state
        
    def get_metric(self, r: torch.Tensor, M_param: torch.Tensor, C_param: torch.Tensor, 
                   G_param: torch.Tensor, t: torch.Tensor = None, phi: torch.Tensor = None) -> tuple:
        """
        Returns metric for spherically symmetric perfect fluid.
        
        For simplicity, we use the interior Schwarzschild solution
        for a constant density sphere.
        """
        # Schwarzschild radius
        rs = 2 * G_param * M_param / C_param**2
        
        # Radius of the fluid sphere (e.g., 2.5 times Schwarzschild radius)
        R_fluid = 2.5 * rs
        
        # Inside the fluid sphere
        inside = r < R_fluid
        
        if torch.any(inside):
            # Interior solution (constant density)
            # This is a simplified model - real fluids would have pressure gradients
            
            # For constant density sphere: ρ = 3M/(4πR³)
            # The interior metric is more complex, but we'll use an approximation
            
            # Interior metric functions
            xi = r / R_fluid
            f_interior = torch.sqrt(1 - rs * xi**2 / R_fluid)
            h_interior = (3 * torch.sqrt(1 - rs/R_fluid) - torch.sqrt(1 - rs * xi**2 / R_fluid)) / 2
            
            # Exterior Schwarzschild
            f_exterior = 1 - rs / r
            
            # Combine interior and exterior solutions
            f_r = torch.where(inside, f_interior, f_exterior)
            g_tt_factor = torch.where(inside, h_interior**2, f_exterior)
            
            g_tt = -g_tt_factor
            g_rr = 1 / (f_r**2 + 1e-15)
            g_pp = r**2
            g_tp = torch.zeros_like(r)
        else:
            # Pure exterior Schwarzschild
            f_r = 1 - rs / r
            g_tt = -f_r
            g_rr = 1 / (f_r + 1e-15)
            g_pp = r**2
            g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp
    
    def get_fluid_stress_energy_tensor(self, r: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the stress-energy tensor for the perfect fluid.
        
        Returns:
            T_μν as a 4x4 tensor at each point
        """
        batch_size = r.shape[0]
        T = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=r.device)
        
        # For a static, spherically symmetric fluid
        # T^t_t = -ρc² (energy density)
        # T^r_r = T^θ_θ = T^φ_φ = p (pressure)
        
        # Simple model: constant density inside a sphere
        R_fluid = 5.0  # Fluid radius in geometric units
        inside = r < R_fluid
        
        # Energy density (geometric units)
        rho = torch.where(inside, torch.ones_like(r), torch.zeros_like(r))
        
        # Pressure from equation of state
        p = self.w * rho
        
        # Fill tensor components (contravariant)
        T[:, 0, 0] = -rho  # T^t_t
        T[:, 1, 1] = p     # T^r_r  
        T[:, 2, 2] = p     # T^θ_θ
        T[:, 3, 3] = p     # T^φ_φ
        
        return T
    
    def compute_sound_speed(self) -> torch.Tensor:
        """
        Compute the speed of sound in the fluid: c_s² = ∂p/∂ρ
        
        For barotropic fluid with p = wρc²: c_s² = wc²
        """
        c_s_squared = self.w  # In units where c=1
        return torch.sqrt(torch.tensor(c_s_squared))
    
    def check_energy_conditions(self, rho: torch.Tensor, p: torch.Tensor) -> dict:
        """
        Check if the fluid satisfies various energy conditions.
        
        Args:
            rho: Energy density
            p: Pressure
            
        Returns:
            Dictionary with boolean values for each condition
        """
        # In units where c=1
        return {
            'null_energy': torch.all(rho + p >= 0),  # ρ + p ≥ 0
            'weak_energy': torch.all(rho >= 0) and torch.all(rho + p >= 0),  # ρ ≥ 0 and ρ + p ≥ 0
            'strong_energy': torch.all(rho + 3*p >= 0) and torch.all(rho + p >= 0),  # ρ + 3p ≥ 0 and ρ + p ≥ 0
            'dominant_energy': torch.all(rho >= torch.abs(p))  # ρ ≥ |p|
        }
