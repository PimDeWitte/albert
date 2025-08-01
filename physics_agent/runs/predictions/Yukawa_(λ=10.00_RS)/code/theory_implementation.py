#!/usr/bin/env python3
"""Exact theory implementation used for prediction"""

import numpy as np
import torch
from physics_agent.base_theory import GravitationalTheory

class Yukawa(GravitationalTheory):
    """
    Yukawa-type modification to gravity with exponential screening.
    <reason>Adds exponential decay term modeling short-range modifications to gravity.</reason>
    <reason>λ is the characteristic length scale of the Yukawa potential.</reason>
    """
    category = "quantum"
    # <reason>chain: Update to range format</reason>
    sweep = dict(lambda_rs={'min': 1.5, 'max': 100.0, 'points': 10, 'scale': 'log'})
    # <reason>chain: Set large lambda_rs for Newtonian limit</reason>
    preferred_params = {'lambda_rs': 1e6}
    cacheable = True
    def __init__(self, lambda_rs: float = 10.0):
        super().__init__(f"Yukawa (λ={lambda_rs:.2f} RS)")
        self.lambda_rs = lambda_rs  # Length scale in units of Schwarzschild radius
        self.alpha = 0.5  # <reason>Coupling strength for Yukawa modification, matching the value used in get_metric</reason>
        # <reason>Update Lagrangian to use lambda_rs which matches the attribute name</reason>
        self.lagrangian = get_symbol('R') + get_symbol('alpha') * sp.exp(-get_symbol('r') / get_symbol('lambda_rs'))
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the metric components with Yukawa-type modification.
        <reason>Yukawa potential adds exponentially screened contribution to gravity.</reason>
        """
        rs = 2 * G_param * M_param / C_param**2
        lambda_val = torch.tensor(self.lambda_rs, device=r.device, dtype=r.dtype)
        
        # <reason>Yukawa length scale λ in physical units</reason>
        lambda_phys = lambda_val * rs
        
        # <reason>Yukawa modification: gravity weakens exponentially beyond λ</reason>
        # <reason>Standard form: V_Yukawa = -α(e^(-r/λ)/r) added to Newtonian potential</reason>
        yukawa_factor = torch.exp(-r / lambda_phys)
        
        # <reason>Modified metric function with Yukawa screening</reason>
        m = 1 - rs / r * (1 - 0.5 * yukawa_factor)  # <reason>0.5 is coupling strength</reason>
        
        epsilon = 1e-10
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 


# Instantiation with exact parameters
theory = Yukawa()
theory.alpha = 0.5
theory.beta = "β"
theory.gamma = "γ"
