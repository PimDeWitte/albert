import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    category = "quantum"
    
    def __init__(self, coupling=0.1):
        super().__init__(name="Quantum Unified Theory")
        self.coupling = coupling
        
        # Define Lagrangian symbols
        R = sp.Symbol('R')  # Ricci scalar
        F = sp.Symbol('F')  # EM field strength
        T = sp.Symbol('T')  # Torsion (if requested)
        
        # Your Lagrangian here - must include quantum unification terms
        self.lagrangian = R - (1/4) * F**2 + self.coupling * T * F + (1/2) * T**2
        
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        
        # Implement your modified metric
        g_tt = -(1 - rs/r + self.coupling / r**2)  # modify as needed
        g_rr = 1/(1 - rs/r + self.coupling / r**2) # modify as needed
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp