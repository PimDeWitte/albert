#!/usr/bin/env python3
"""Exact theory implementation used for prediction"""

import numpy as np
import torch
from physics_agent.base_theory import GravitationalTheory

class TestDivergent(GravitationalTheory):
    """
    A test theory that intentionally produces divergent behavior.
    Used for testing numerical stability and error handling.
    
    WARNING: This theory is not physical and will cause simulations to fail!
    """
    category = "test"
    def __init__(self, alpha: float = 1e-3):
        super().__init__(f"TestDivergent-α={alpha:.2e}")  # Auto-detects symmetry
        self.alpha = alpha
        self.lagrangian = sp.sympify('R')  # Dummy Lagrangian
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns a metric that diverges rapidly.
        """
        rs = 2 * G_param * M_param / C_param**2
        
        # Create a metric that diverges
        # The alpha parameter controls how quickly it diverges
        divergence_factor = torch.exp(self.alpha * rs / r)
        
        f = (1 - rs / r) * divergence_factor
        
        # This will cause numerical issues!
        epsilon = 1e-10
        g_tt = -f
        g_rr = 1 / (f + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 


# Instantiation with exact parameters
theory = TestDivergent()
theory.alpha = 0.001
theory.beta = "β"
theory.gamma = "γ"
