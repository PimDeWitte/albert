import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Modified Reissner-Nordström metric with quantum corrections
    
    Lagrangian: L = R/(16πG) - F^{μν}F_{μν}/4 + α/r² * log(r/r_p)
    where α is a quantum correction parameter and r_p is the Planck length.
    """
    
    def __init__(self, alpha: float = 0.1):
        # Use force_6dof_solver=False to indicate symmetric metric
        super().__init__(name=f"Quantum Corrected RN (α={alpha})", force_6dof_solver=False)
        self.alpha = alpha
        self.category = "quantum"
    
    def get_metric(self, r, M_param, C_param, G_param, Q_param=1e-5, **kwargs):
        """Returns the metric tensor components with quantum corrections"""
        
        rs = 2 * G_param * M_param / C_param**2
        rq = G_param * Q_param**2 / (4 * torch.pi * 8.854e-12 * C_param**4)
        
        # Quantum correction term
        r_planck = 1.616e-35  # Planck length in meters
        quantum_term = self.alpha / r**2 * torch.log(r / r_planck + 1)
        
        # Modified metric components
        f = 1 - rs/r + rq/r**2 + quantum_term
        
        g_tt = -f
        g_rr = 1/f
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp
