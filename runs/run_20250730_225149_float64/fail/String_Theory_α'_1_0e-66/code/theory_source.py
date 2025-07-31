from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor, QuantumMixin
import torch
class StringTheory(GravitationalTheory, QuantumMixin):
    """
    String Theory: Unifies all forces including gravity by treating particles as vibrating strings.
    
    Fundamental objects are 1D strings, leading to quantum gravity via graviton mode.
    Includes supersymmetry and extra dimensions.
    
    Category: quantum (quantum gravity + gauge theories)
    
    Parameters:
    - alpha_prime: String tension parameter (Regge slope), related to string length ls = sqrt(alpha')
    
    Lagrangian: Polyakov action for strings or low-energy effective supergravity action.
    
    Metric: In low-energy limit, similar to GR with stringy corrections; for black holes, has extremal limits.
    
    Web reference: https://en.wikipedia.org/wiki/String_theory
    Additional source: https://arxiv.org/abs/hep-th/9711200 (String theory review by Polchinski)
    
    Novel predictions:
    - Extra dimensions (10 for superstring)
    - Supersymmetric partners
    - Black hole entropy from string microstates
    - AdS/CFT correspondence
    
    Quantum effects: Fully quantum, includes loop corrections automatically.
    """
    category = "quantum"
    sweep = dict(alpha_prime={'min': 1e-68, 'max': 1e-64, 'points': 11, 'scale': 'log'})  # Planck length squared range
    preferred_params = {'alpha_prime': 1e-66}  # Typical value ls ~ Planck length
    def __init__(self, alpha_prime: float = 1e-66, enable_quantum: bool = True, **kwargs):
        super().__init__(name=f"String Theory (α'={alpha_prime:.1e})", enable_quantum=enable_quantum, **kwargs)
        self.alpha_prime = alpha_prime
        
        # Low-energy effective Lagrangian: supergravity + string corrections
        R = get_symbol('R')
        self.lagrangian = R + get_symbol('alpha_prime') * (get_symbol('R_mnrs') * get_symbol('R^mnrs'))  # Leading correction
        
        # Add stringy matter: bosonic and fermionic modes
        self.matter_lagrangian = get_symbol('X_mu') * get_symbol('partial^alpha X^mu') * get_symbol('partial_alpha X_mu')  # Simplified Polyakov
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        String-corrected black hole metric (simplified, leading order correction).
        For extremal black holes in string theory, but here approximate for Schwarzschild-like.
        """
        rs = 2 * G_param * M_param / C_param**2
        alpha_p = torch.tensor(self.alpha_prime, device=r.device, dtype=r.dtype)
        
        # String correction: modifies near-horizon
        correction = alpha_p / r**2  # Dimensional, placeholder for alpha' R^2 term effect
        
        f = 1 - rs / r + correction * (rs / r)**2
        
        g_tt = -f
        g_rr = 1 / f
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 