"""
Example theory implementation.
Copy this file and modify for your own theory.
"""

import torch
import sympy as sp
from physics_agent.base_theory import GravitationalTheory, Tensor
class MyTheory(GravitationalTheory):
    """
    Template for a new gravitational theory.
    """
    def __init__(self, some_parameter=1.0):
        # The name will be used for plots and filenames.
        super().__init__(name=f"MyTheory (p={some_parameter})")
        
        # Set the Lagrangian (symbolic expression)
        self.lagrangian = sp.sympify('R')  # Placeholder: Einstein-Hilbert action
        
        # Store parameters
        self.some_parameter = some_parameter
        
        # Note: The framework will automatically detect if your metric is symmetric
        # (g_tp = 0) and choose the appropriate solver. If you need to force the
        # 6-DOF general solver for some reason, you can pass force_6dof_solver=True
        # to the parent constructor.
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        This is where you define the metric tensor of your theory.
        It must return a tuple of 4 Tensors: (g_tt, g_rr, g_pp, g_tp).
        
        For a simple modification to Schwarzschild, you might do something like this:
        """
        rs = 2 * G_param * M_param / C_param**2
        
        # Example modification: add a term based on self.some_parameter
        modification = 1 + self.some_parameter / r
        
        g_tt = -(1 - rs / r) * modification
        g_rr = 1 / (1 - rs / r)
        g_pp = r**2
        g_tp = torch.zeros_like(r) # No time-phi coupling
        
        return g_tt, g_rr, g_pp, g_tp
    def get_cache_tag(self, N_STEPS: int, precision_tag: str, r0_tag: int) -> str:
        """
        Generate unique cache tag including parameters.
        """
        return f"{self.name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')}_{N_STEPS}_{precision_tag}_r{r0_tag}"

# Example of torchphysics integration (commented)
# def solve_geodesic_pinn(self):
#     tau = tp.domains.Interval(tp.spaces.R1('tau'), 0, 10)
#     // ... setup PINN 