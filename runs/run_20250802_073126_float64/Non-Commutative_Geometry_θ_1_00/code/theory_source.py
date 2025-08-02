"""
Non-commutative geometry approach to quantum gravity.
Based on https://arxiv.org/pdf/2305.03671

Non-commutative spacetime where coordinates satisfy [x^μ, x^ν] = iθ^μν.
This introduces a fundamental length scale (Planck scale) below which
spacetime becomes fuzzy/quantum.
"""

import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory, Tensor, QuantumMixin
from physics_agent.constants import get_symbol

class NonCommutativeGeometry(GravitationalTheory, QuantumMixin):
    """
    Non-commutative geometry approach to quantum gravity.
    <reason>Modifies spacetime at Planck scale through non-commuting coordinates.</reason>
    <reason>Introduces minimal length scale through θ parameter (non-commutativity scale).</reason>
    """
    category = "quantum"
    
    # <reason>chain: Sweep over non-commutativity parameter θ</reason>
    sweep = dict(theta={'min': 0.0, 'max': 10.0, 'points': 11, 'scale': 'linear'})
    preferred_params = {'theta': 1.0}  # In Planck length units
    cacheable = True
    
    def __init__(self, theta: float = None):
        """
        Initialize non-commutative geometry theory.
        
        Args:
            theta: Non-commutativity parameter (in Planck length squared units).
                   Controls the scale at which spacetime becomes non-commutative.
        """
        if theta is None:
            theta = self.preferred_params.get('theta', 1.0)
            
        # <reason>chain: Define quantum field components for NC geometry</reason>
        # Matter fields on NC space
        psi = get_symbol('ψ')
        psi_bar = get_symbol('ψ̄')
        gamma_mu = get_symbol('γ^μ')
        
        # NC modifications appear through star product in field equations
        # Moyal-Weyl star product: f ⋆ g = f·g + (i/2)θ^μν∂_μf∂_νg + O(θ²)
        
        # Lagrangians with NC modifications
        matter_lagrangian = sp.I * psi_bar * gamma_mu * get_symbol('D_μ') * psi - get_symbol('m_f') * psi_bar * psi
        gauge_lagrangian = -sp.Rational(1,4) * get_symbol('F_μν') * get_symbol('F^μν')
        # NC effects enter through modified commutation relations in gauge sector
        interaction_lagrangian = -get_symbol('q') * psi_bar * gamma_mu * psi * get_symbol('A_μ')
        
        name = f"Non-Commutative Geometry (θ={theta:.2f})"
        
        super().__init__(
            name,
            lagrangian=get_symbol('R') + theta * get_symbol('R²'),  # Higher-order curvature from NC
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        self.theta = theta
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, 
                   t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculate metric with non-commutative geometry corrections.
        <reason>NC effects modify the effective geometry at small scales.</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        theta = torch.tensor(self.theta, device=r.device, dtype=r.dtype)
        
        # <reason>chain: Planck length in geometric units (G=c=1)</reason>
        # Convert G_param and C_param to tensors if they aren't already
        G_tensor = torch.tensor(G_param, device=r.device, dtype=r.dtype) if not isinstance(G_param, torch.Tensor) else G_param
        C_tensor = torch.tensor(C_param, device=r.device, dtype=r.dtype) if not isinstance(C_param, torch.Tensor) else C_param
        l_p = torch.sqrt(G_tensor / C_tensor**3) * torch.sqrt(torch.tensor(1.616e-35, device=r.device, dtype=r.dtype))
        
        # <reason>chain: NC correction scale - becomes significant when r ~ sqrt(θ) * l_p</reason>
        # The correction modifies the metric at scales where non-commutativity matters
        nc_scale = torch.sqrt(theta) * l_p
        
        # <reason>chain: NC modification of Schwarzschild metric</reason>
        # Based on effective metric from NC field theory
        # The 1/(1 + (r/nc_scale)²) factor suppresses corrections at large r
        nc_correction = theta * (rs / r) * torch.exp(-r**2 / (4 * nc_scale**2))
        
        # Modified metric components
        g_tt = -(1 - rs/r) * (1 + nc_correction)
        g_rr = 1 / ((1 - rs/r) * (1 - nc_correction))
        g_pp = r**2 * (1 + theta * nc_correction / r**2)  # Angular part also modified
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp
    
    def compute_hawking_temperature(self, M: Tensor, C_param: Tensor, G_param: Tensor) -> Tensor:
        """
        Hawking temperature with NC corrections.
        <reason>NC geometry modifies the horizon structure and thus Hawking radiation.</reason>
        """
        # Standard Hawking temperature
        T_H = C_param**3 / (8 * torch.pi * G_param * M)
        
        # <reason>chain: NC corrections modify temperature through horizon deformation</reason>
        # Small increase in temperature due to fuzzy horizon
        theta = torch.tensor(self.theta, device=M.device, dtype=M.dtype)
        l_p = torch.sqrt(G_param / C_param**3)
        nc_correction = 1 + theta * (l_p * C_param / (G_param * M))**2
        
        return T_H * nc_correction
    
    def unification_scale(self) -> float:
        """
        Energy scale where NC effects become important.
        <reason>NC geometry naturally introduces a UV cutoff at the NC scale.</reason>
        """
        # In Planck units, unification occurs when E ~ 1/sqrt(θ)
        return 1.0 / (self.theta**0.5) if self.theta > 0 else float('inf')