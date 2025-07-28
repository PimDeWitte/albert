import torch
import numpy as np
import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor, QuantumMixin

class WeylEM(GravitationalTheory, QuantumMixin):
    name = "Weyl-EM Quantum"
    """
    <reason>chain: Weyl conformal gravity unified with electromagnetic field</reason>
    <reason>chain: Exhibits quantum deformation through modified Planck-scale corrections</reason>
    <reason>chain: Now includes full QED Lagrangian for precision tests</reason>
    """
    category = "quantum"
    
    def __init__(self, omega: float = 0.0):
        # <reason>chain: Define quantum field components for quantum theory</reason>
        # Get QED symbols
        from physics_agent.base_theory import QuantumUnifiedMixin
        qum = QuantumUnifiedMixin()
        qum.add_quantum_field_components()
        
        # Build complete Lagrangian
        
        # Gravity part: Weyl-squared term
        C_munu = get_symbol('C_μν')  # Weyl tensor (covariant)
        C_munu_up = get_symbol('C^μν')  # Weyl tensor (contravariant)
        gravity_lagrangian = C_munu * C_munu_up
        
        # Use QED components from mixin
        matter_lagrangian = qum.matter_lagrangian
        gauge_lagrangian = qum.gauge_lagrangian  
        interaction_lagrangian = qum.interaction_lagrangian
        
        super().__init__(
            f"Weyl-EM Quantum (ω={omega:.2f})",
            lagrangian=gravity_lagrangian,
            matter_lagrangian=matter_lagrangian,
            gauge_lagrangian=gauge_lagrangian,
            interaction_lagrangian=interaction_lagrangian
        )
        self.omega = omega
        
        # <reason>chain: Store QED symbols for g-2 calculations</reason>
        self.qed_symbols = qum.qed_symbols
        
    def get_metric(self, r: torch.Tensor, M_param: torch.Tensor, C_param: torch.Tensor, G_param: torch.Tensor, t: torch.Tensor = None, phi: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        <reason>chain: Weyl-EM metric with conformal factor</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        
        # Conformal factor from Weyl symmetry
        omega = torch.tensor(self.omega, device=r.device, dtype=r.dtype)
        conformal = 1 + omega * (rs / r)**2
        
        # Base Schwarzschild modified by conformal factor
        m = conformal * (1 - rs / r)
        
        epsilon = 1e-10
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2 * conformal  # Conformal scaling of angular part
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp
        
    def compute_g2_correction(self, **params) -> float:
        """
        <reason>chain: Compute theory-specific correction to electron g-2</reason>
        
        In Weyl gravity, conformal symmetry can modify QED vertex corrections.
        """
        alpha = params.get('alpha', 1.0/137.035999084)
        omega = self.omega
        
        # <reason>chain: Weyl gravity introduces log corrections to running coupling</reason>
        # δa_Weyl ~ α²ω log(M_P/m_e)
        m_e = params.get('m_e', 9.109e-31)
        M_P = 2.176e-8  # Planck mass in kg
        
        weyl_correction = (alpha**2) * omega * np.log(M_P / m_e) / (4 * np.pi)
        
        return weyl_correction 