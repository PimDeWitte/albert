import sympy as sp
from gravitational_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    def __init__(self):
        super().__init__("TorsionGravity")
        self.kappa = 0.5  # Torsion coupling constant
        
        # Lagrangian with torsion contribution
        R = sp.Symbol('R')  # Ricci scalar
        T = sp.Symbol('T')  # Torsion scalar
        self.lagrangian = R + self.kappa * T**2
        
    def get_metric(self, r, theta, M, a=0, Q=0):
        """
        Returns the metric tensor components for the theory with torsion effects.
        For simplicity, we assume a spherically symmetric metric with modifications.
        """
        c = sp.Symbol('c')
        G = sp.Symbol('G')
        rs = 2 * G * M / c**2
        
        # Metric components with torsion-inspired correction
        g_tt = -(1 - rs/r) * (1 + self.kappa / r**2)
        g_rr = 1 / (1 - rs/r) * (1 - self.kappa / r**2)
        g_pp = r**2 * sp.sin(theta)**2
        g_tp = 0
        
        return {'g_tt': g_tt, 'g_rr': g_rr, 'g_pp': g_pp, 'g_tp': g_tp}