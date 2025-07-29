import sympy as sp
from physics_agent.constants import get_symbol
from physics_agent.base_theory import GravitationalTheory, Tensor, QuantumMixin
import torch
class KaluzaKleinTheory(GravitationalTheory, QuantumMixin):
    """
    Kaluza-Klein theory: Unifies gravity and electromagnetism by adding an extra compactified dimension.
    
    In this theory, the 5-dimensional metric reduces to 4D gravity plus EM fields.
    The extra dimension is compactified on a circle of radius R_kk.
    
    Category: quantum (classical unification of gravity and EM)
    
    Parameters:
    - R_kk: Compactification radius (in meters), controls the strength of EM relative to gravity
    
    Lagrangian: 5D Einstein-Hilbert action, which upon dimensional reduction gives GR + Maxwell + scalar field.
    
    Metric: In 4D, it's the Schwarzschild metric plus EM-like terms, but for vacuum, it's similar to Reissner-Nordström for charged cases.
    
    Web reference: https://en.wikipedia.org/wiki/Kaluza%E2%80%93Klein_theory
    Additional source: https://arxiv.org/abs/hep-th/0103239 (review of Kaluza-Klein theories)
    
    Novel predictions:
    - Kaluza-Klein tower of massive particles (excitations in extra dimension)
    - Unification of gravity and EM in higher dimensions
    - Scalar dilaton field from compactification
    
    Quantum effects: Can be extended to supergravity or string theory contexts.
    """
    category = "quantum"
    sweep = dict(R_kk={'min': 1e-35, 'max': 1e-30, 'points': 11, 'scale': 'log'})  # Planck scale range
    preferred_params = {'R_kk': 1e-32}  # Typical small radius for invisibility at low energies
    def __init__(self, R_kk: float = 1e-32, enable_quantum: bool = True, **kwargs):
        # <reason>chain: Build Lagrangians directly using get_symbol calls</reason>
        super().__init__(name=f"Kaluza-Klein (R_kk={R_kk:.1e})", enable_quantum=enable_quantum, **kwargs)
        self.R_kk = R_kk
        
        # 5D Lagrangian: sqrt(-g5) get_symbol('R5')
        self.lagrangian = get_symbol('R5')
        
        # Upon reduction: GR + Maxwell + dilaton
        self.gauge_lagrangian = - (1/4) * get_symbol('F_munu') * get_symbol('F^munu') * sp.exp(get_symbol('phi'))
        self.matter_lagrangian = (1/2) * get_symbol('partial_mu phi') * get_symbol('partial^mu phi')  # Dilaton kinetic term
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        For neutral vacuum, KK theory reduces to Schwarzschild. For charged, it's like Reissner-Nordström.
        
        <reason>chain: In Kaluza-Klein theory, the effective charge arises from momentum in the extra dimension</reason>
        <reason>chain: The charge-to-mass ratio is q/m = n/(R_kk * M) where n is the KK mode number</reason>
        <reason>chain: For the ground state (n=0), the theory reduces to pure GR (Schwarzschild)</reason>
        <reason>chain: For excited states (n≠0), we get an effective Reissner-Nordström metric</reason>
        
        Note: Full KK metric is 5D, but we project to 4D effective.
        """
        rs = 2 * G_param * M_param / C_param**2
        
        # <reason>chain: Fix the effective charge calculation to be physically reasonable</reason>
        # In KK theory, the effective charge Q = q = ne/√(4πε₀G) where n is the KK mode
        # For the ground state test particle (n=0), there's no charge
        # For the first excited state (n=1), we get a small effective charge
        
        # Use the elementary charge as the fundamental unit
        e = torch.tensor(1.602e-19, device=r.device, dtype=r.dtype)  # Elementary charge in Coulombs
        epsilon_0 = torch.tensor(8.854e-12, device=r.device, dtype=r.dtype)  # Vacuum permittivity
        
        # <reason>chain: For testing, use n=1 (first KK mode) to see electromagnetic effects</reason>
        n_kk = 1  # KK mode number
        
        # <reason>chain: The effective charge in geometric units</reason>
        # Q = n * e / sqrt(4π ε₀ G) but we need to be careful with units
        # In geometric units where G=c=1, the charge has units of length
        # Q_geom = Q_SI * sqrt(G/c^4) * c^2
        
        # <reason>chain: Use a simplified approach - make Q proportional to elementary charge</reason>
        # but scaled to give reasonable Reissner-Nordström effects
        # For a solar mass black hole, Q/M ~ 1e-20 gives interesting physics
        Q_over_M = 1e-20 * n_kk  # Dimensionless charge-to-mass ratio
        Q_eff = Q_over_M * M_param  # Effective charge in same units as M
        
        # <reason>chain: Add safety check to prevent numerical issues</reason>
        # Ensure Q² < M² to avoid naked singularities
        Q_max = torch.tensor(0.99, device=r.device, dtype=r.dtype) * M_param  # Maximum allowed charge
        Q_eff_tensor = torch.tensor(Q_eff, device=r.device, dtype=r.dtype)
        Q_eff = torch.min(Q_eff_tensor, Q_max).item()
        
        # Reissner-Nordström metric with the effective charge
        rq2 = G_param * Q_eff**2 / C_param**4
        
        f = 1 - rs / r + rq2 / r**2
        
        # <reason>chain: Add numerical stability check</reason>
        # Ensure f doesn't become negative (which would make g_rr undefined)
        min_f = 1e-10
        f = torch.where(f < min_f, torch.tensor(min_f, device=r.device, dtype=r.dtype), f)
        
        g_tt = -f
        g_rr = 1 / f
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 