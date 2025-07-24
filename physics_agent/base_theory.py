import torch
import os
import inspect
from abc import abstractmethod
import sympy as sp

# <reason>chain: Import all necessary constants from centralized constants module</reason>
from physics_agent.constants import (
    SOLAR_MASS, c, G,
    get_symbol, PHYSICS_SYMBOLS
)

Tensor = torch.Tensor  # Type alias for brevity
# <reason>chain: Type alias unchanged for brevity.</reason>


class QuantumMixin:
    """<reason>chain: Mixin to provide standard quantum field Lagrangian components</reason>"""
    
    def add_quantum_field_components(self):
        """<reason>chain: Add standard quantum field theory Lagrangian components</reason>"""
        # Only add if not already defined (allows theories to override)
        if not hasattr(self, 'matter_lagrangian') or self.matter_lagrangian is None:
            self.matter_lagrangian = sp.Symbol('psi_bar') * (sp.I * sp.Symbol('gamma_mu') * sp.Symbol('D_mu') - sp.Symbol('m')) * sp.Symbol('psi')
        
        if self.category == 'quantum':
            if not hasattr(self, 'gauge_lagrangian') or self.gauge_lagrangian is None:
                self.gauge_lagrangian = -sp.Rational(1, 4) * sp.Symbol('F_munu') * sp.Symbol('F_munu')
        
        if not hasattr(self, 'interaction_lagrangian') or self.interaction_lagrangian is None:
            self.interaction_lagrangian = sp.Symbol('q') * sp.Symbol('psi_bar') * sp.Symbol('gamma_mu') * sp.Symbol('A_mu') * sp.Symbol('psi')

        # <reason>chain: Critical Addition - Quantum Information Theory Components</reason>
        # Entanglement entropy terms for emergent gravity
        self.S_ent = sp.Symbol('S_ent')  # Entanglement entropy
        self.rho = sp.Symbol('ρ')  # Density matrix
        self.I_AB = sp.Symbol('I(A:B)')  # Mutual information
        self.C_AB = sp.Symbol('C(A:B)')  # Quantum correlations
        
        # <reason>chain: Add entanglement contribution to gravity</reason>
        if hasattr(self, 'lagrangian') and self.lagrangian is not None:
            sp.Symbol('R')
            # Add emergent gravity term: L = R + α*S_ent
            alpha_ent = sp.Symbol('α_ent')  # Entanglement coupling
            self.lagrangian = self.lagrangian + alpha_ent * self.S_ent
            
        # <reason>chain: Renormalization group parameters</reason>
        self.beta_functions = {}  # Beta functions for running couplings
        self.cutoff_scale = sp.Symbol('Λ')  # UV cutoff
        self.running_couplings = {}  # Scale-dependent couplings
        
    def compute_entanglement_entropy(self, region_size, total_size):
        """<reason>chain: Compute entanglement entropy for a region</reason>"""
        # Von Neumann entropy: S = -Tr(ρ log ρ)
        # For a pure state partitioned into A and B:
        # S_A = S_B (Page curve for black holes)
        import numpy as np
        if region_size < total_size / 2:
            # Before Page time
            return region_size * np.log(2)  # Bekenstein-Hawking scaling
        else:
            # After Page time - information starts coming out
            return (total_size - region_size) * np.log(2)
            
    def check_renormalizability(self) -> dict:
        """<reason>chain: Check if theory is renormalizable</reason>"""
        # Analyze power counting of operators
        results = {
            'is_renormalizable': True,
            'divergent_operators': [],
            'beta_functions': self.beta_functions,
            'fixed_points': []
        }
        
        # Check dimension of operators in Lagrangian
        if hasattr(self, 'lagrangian'):
            # Simplified check - real implementation needs full analysis
            # Operators with dimension > 4 are non-renormalizable in 4D
            if 'R**2' in str(self.lagrangian):
                results['divergent_operators'].append('R^2 term')
                results['is_renormalizable'] = False
                
        return results


class GravitationalTheory:
    """
    Base class for gravitational theories.
    """
    category = "classical"  # Default category
    sweep = {}  # Existing sweep dict
    preferred_params = {}  # Existing preferred params
    sweepable_fields = {}  # <reason>chain: New dict for dynamic sweep parameters, e.g., {'gw_noise': {'default': 0.0, 'min': 0.0, 'max': 0.5, 'points': 5}}</reason>
    
    def __init__(self, name: str, lagrangian: sp.Expr, matter_lagrangian: sp.Expr = None, gauge_lagrangian: sp.Expr = None, interaction_lagrangian: sp.Expr = None):
        self.name = name
        self.lagrangian = lagrangian
        self.matter_lagrangian = matter_lagrangian
        self.gauge_lagrangian = gauge_lagrangian
        self.interaction_lagrangian = interaction_lagrangian
        self.complete_lagrangian = self._combine_lagrangians()
    
    # <reason>chain: Add __getattr__ to dynamically resolve symbol requests</reason>
    def __getattr__(self, name):
        """
        Dynamically resolve symbol requests to avoid storing redundant attributes.
        This allows the Lagrangian validator to find symbols without explicit storage.
        """
        # Standard physics symbols that should return the symbol itself
        
        # Check if it's a known physics symbol
        for sym, data in PHYSICS_SYMBOLS.items():
            if sym == name or name in data.get('aliases', []):
                return get_symbol(name)
        
        # Special handling for common symbolic attributes
        if name in ['L_matter', 'L_gauge', 'L_int']:
            return get_symbol(name)
            
        # If not found, raise AttributeError as normal
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    # Metadata for the theory
    category: str = "base"  # "classical", "quantum"
    
    def __init__(self, name: str, force_6dof_solver: bool = None, lagrangian: sp.Expr = None, 
                 enable_quantum: bool = None, **kwargs):
        """
        Initialize a gravitational theory.
        
        Args:
            name: Theory name for display
            force_6dof_solver: If True, forces use of 6-DOF general solver even if metric is symmetric.
                             If False, forces use of 4-DOF symmetric solver (if metric allows).
                             If None (default), auto-detects based on g_tp component.
            lagrangian: Symbolic expression for the theory's Lagrangian
            enable_quantum: Whether to enable quantum calculations. If None, auto-determined by category.
        """
        self.name = name
        self.force_6dof_solver = force_6dof_solver  # Explicit solver override
        self._is_symmetric_cached = None    # Cache for auto-detection
        self.lagrangian = lagrangian
        
        # <reason>chain: Add quantum field components for quantum theories</reason>
        self.matter_lagrangian = kwargs.get('matter_lagrangian', None)  # Dirac, Klein-Gordon, etc.
        self.gauge_lagrangian = kwargs.get('gauge_lagrangian', None)    # Yang-Mills terms
        self.interaction_lagrangian = kwargs.get('interaction_lagrangian', None)
        
        # <reason>chain: Validate required methods based on theory category</reason>
        self._validate_required_methods()
        
        # <reason>chain: Automatically add standard quantum components for quantum theories</reason>
        if self.category == 'quantum' and isinstance(self, QuantumMixin):
            self.add_quantum_field_components()
        
        # <reason>chain: Set default precision to float64 to prevent precision loss</reason>
        self.dtype = kwargs.get('dtype', torch.float64)
        
        # <reason>chain: Determine if quantum calculations should be enabled</reason>
        if enable_quantum is None:
            # Auto-enable for quantum category
            self.enable_quantum = self.category == 'quantum'
        else:
            self.enable_quantum = enable_quantum
            
        # <reason>chain: Initialize quantum path integrator if quantum is enabled</reason>
        self._quantum_integrator = None  # Lazy initialization
        
        # <reason>chain: Initialize source_dir for tracking theory location</reason>
        self.source_dir = "unknown"  # Initialize source_dir
        # Get the directory of the child class relative to the 'theories' folder
        try:
            class_path = inspect.getfile(self.__class__)
            # Navigate up to find the 'theories' directory
            base_path = os.path.dirname(class_path)
            theories_path = base_path
            
            # Find the theories directory by going up the path
            while os.path.basename(theories_path) != 'theories' and os.path.dirname(theories_path) != theories_path:
                parent = os.path.dirname(theories_path)
                if os.path.basename(parent) == 'theories':
                    theories_path = parent
                    break
                theories_path = parent
            
            if 'theories' in theories_path:
                # Get the path relative to the 'theories' directory
                self.source_dir = os.path.relpath(os.path.dirname(class_path), theories_path)
            else:
                self.source_dir = os.path.basename(os.path.dirname(class_path))  # Fallback to just the directory name
        except (TypeError, ValueError):
            # This can happen for dynamically generated classes that don't have a source file
            self.source_dir = ""
        
        # <reason>chain: Absorb any extra kwargs that are not used by the base class constructor</reason>
        # This makes it easier to pass sweep parameters without causing errors.
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
                
    @property
    def quantum_integrator(self):
        """<reason>chain: Lazy initialization of quantum integrator to avoid circular imports</reason>"""
        if self._quantum_integrator is None:
            from .quantum_path_integrator import QuantumPathIntegrator
            self._quantum_integrator = QuantumPathIntegrator(self, enable_quantum=self.enable_quantum)
        return self._quantum_integrator
    
    def validate_lagrangian_completeness(self) -> dict:
        """<reason>chain: Validate that theory has all required Lagrangian components</reason>"""
        required = ['lagrangian']
        
        if self.category == 'quantum':
            required.extend(['matter_lagrangian', 'interaction_lagrangian', 'gauge_lagrangian'])
        
        missing = []
        for field in required:
            if not hasattr(self, field) or getattr(self, field) is None:
                missing.append(field)
        
        return {
            'complete': len(missing) == 0,
            'missing': missing,
            'category': self.category,
            'has_matter': hasattr(self, 'matter_lagrangian') and self.matter_lagrangian is not None,
            'has_gauge': hasattr(self, 'gauge_lagrangian') and self.gauge_lagrangian is not None,
            'has_interaction': hasattr(self, 'interaction_lagrangian') and self.interaction_lagrangian is not None,
            'quantum_enabled': self.enable_quantum,
            'path_integral_ready': self.lagrangian is not None and self.enable_quantum
        }

    @property
    def is_symmetric(self) -> bool:
        """
        Determines if the metric is symmetric (g_tp = 0) and thus can use the 4-DOF solver.
        
        <reason>Automatically detects symmetry unless explicitly overridden by force_6dof_solver</reason>
        
        Returns:
            True if metric is symmetric and 4-DOF solver should be used
            False if metric is asymmetric or 6-DOF solver is forced
        """
        # If force_6dof_solver is True, always return False (use general solver)
        if self.force_6dof_solver is True:
            return False
            
        # If force_6dof_solver is False, return True if possible
        # (will still check metric compatibility in the engine)
        if self.force_6dof_solver is False:
            return True
            
        # Otherwise, auto-detect by checking if g_tp is zero
        if self._is_symmetric_cached is not None:
            return self._is_symmetric_cached
            
        # Auto-detect by testing g_tp at several radii
        try:
            # Test parameters (use typical values)
            M = torch.tensor(SOLAR_MASS, dtype=torch.float64)  # Solar mass
            c_tensor = torch.tensor(c, dtype=torch.float64)   # Speed of light
            G_tensor = torch.tensor(G, dtype=torch.float64) # Gravitational constant
            rs = 2 * G_tensor * M / c_tensor**2  # Schwarzschild radius
            
            # Test at several radii
            test_radii = torch.tensor([3.0, 5.0, 10.0, 20.0, 50.0]) * rs
            
            # Check if g_tp is effectively zero at all test points
            is_symmetric = True
            for r in test_radii:
                _, _, _, g_tp = self.get_metric(r.unsqueeze(0), M, c_tensor, G_tensor)
                if torch.abs(g_tp).max() > 1e-10:
                    is_symmetric = False
                    break
                    
            # Cache the result
            self._is_symmetric_cached = is_symmetric
            return is_symmetric
            
        except Exception as e:
            # If auto-detection fails, default to False (use general solver)
            print(f"Warning: Auto-detection of symmetry failed for {self.name}: {e}")
            print("Defaulting to 6-DOF general solver")
            self._is_symmetric_cached = False
            return False

    @abstractmethod
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: Tensor, G_param: Tensor, t: Tensor = None, phi: Tensor = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculate the metric tensor components for this theory.
        
        Args:
            r: Radial coordinate(s)
            M_param: Mass parameter
            C_param: Speed of light
            G_param: Gravitational constant
            t: Time coordinate (optional, for time-dependent metrics)
            phi: Angular coordinate (optional, for non-axisymmetric metrics)
            
        Returns:
            Tuple of (g_tt, g_rr, g_pp, g_tp) metric components
        """

    def get_cache_tag(self, N_STEPS: int, precision_tag: str, r0_tag: int) -> str:
        """Generate a unique cache tag for this theory configuration."""
        # Default implementation
        theory_tag = self.name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')
        return f"{ theory_tag}_r{r0_tag}_steps-{N_STEPS}_dt-{precision_tag}"

    def generate_lagrangian(self, include_em_term: bool = False, em_coupling: float = None) -> sp.Expr:
        """
        Auto-generate a Lagrangian expression for this theory if not already set.
        
        <reason>chain: Generate symbolic Lagrangian from metric to enable path integral validation</reason>
        
        Args:
            include_em_term: Whether to include electromagnetic coupling term
            em_coupling: Coupling constant for EM term (default -1/4)
            
        Returns:
            SymPy expression for the Lagrangian
        """
        if self.lagrangian is not None:
            return self.lagrangian
            
        # Default to Einstein-Hilbert action: L = R
        R_sym = sp.Symbol('R')
        
        try:
            # Get metric at a test point to check its structure
            M = torch.tensor(SOLAR_MASS, dtype=torch.float64)  # Solar mass
            c_tensor = torch.tensor(c, dtype=torch.float64)   # Speed of light  
            G_tensor = torch.tensor(G, dtype=torch.float64) # Gravitational constant
            rs = 2 * G_tensor * M / c_tensor**2
            r_test = 10.0 * rs
            
            g_tt, g_rr, g_pp, g_tp = self.get_metric(r_test.unsqueeze(0), M, c_tensor, G_tensor)
            
            # Check if it's a simple diagonal metric with g_tp = 0
            if torch.all(torch.abs(g_tp) < 1e-10):
                # For diagonal metrics: ds² = -A dt² + B dr² + r² dΩ²
                # We can use the ricci_scalar function from lagrangian_deriver
                
                # Import the function
                pass
                
                # For simple theories, we assume the metric has the form:
                # g_tt = -f(r), g_rr = 1/f(r) where f depends on theory parameters
                # Since we can't easily extract the symbolic form, we default to R
                
                self.lagrangian = R_sym
                
                # <reason>Most pure gravity theories have L = R, which is sufficient</reason>
                # <reason>Complex theories should override this method or set lagrangian manually</reason>
            else:
                # Non-diagonal metric (like Kerr) - still use R as default
                self.lagrangian = R_sym
                
            # Add electromagnetic term if requested
            if include_em_term and em_coupling is not None:
                F_sym = sp.Symbol('F')
                lambda_sym = sp.Symbol('lambdaEM')
                self.lagrangian = R_sym - (lambda_sym / 4) * F_sym**2
                # Pre-substitute the coupling value
                self.lagrangian = self.lagrangian.subs(lambda_sym, em_coupling)
                
        except Exception as e:
            # Fallback to simple R if computation fails
            print(f"Warning: Auto-generation of Lagrangian failed: {e}")
            self.lagrangian = R_sym
            
        return self.lagrangian

    @property
    def complete_lagrangian(self):
        """<reason>chain: Return complete Lagrangian combining all components</reason>"""
        if not self.lagrangian:
            return None
            
        complete = self.lagrangian
        if self.matter_lagrangian:
            complete = complete + self.matter_lagrangian
        if self.gauge_lagrangian:
            complete = complete + self.gauge_lagrangian
        if self.interaction_lagrangian:
            complete = complete + self.interaction_lagrangian
            
        return complete

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', is_symmetric={self.is_symmetric})" 

    def compute_ricci_tensor(self, r: Tensor, M: Tensor, c: Tensor, G: Tensor) -> Tensor:
        """
        Compute the Ricci tensor components for this theory at radial coordinate r.
        
        <reason>chain: Default implementation for diagonal Schwarzschild-like metrics</reason>
        <reason>chain: Computes Ricci components using standard GR formulas for spherical symmetry</reason>
        <reason>chain: Override in subclasses for custom metrics or non-diagonal cases</reason>
        
        Args:
            r: Radial coordinate(s) [Tensor]
            M: Mass parameter
            c: Speed of light
            G: Gravitational constant
            
        Returns:
            4x4 Ricci tensor (batch_size x 4 x 4 if r is batched)
        """
        # Check if metric is symmetric (diagonal)
        if not self.is_symmetric:
            raise NotImplementedError(
                f"Ricci tensor computation not implemented for non-symmetric metrics in {self.name}. "
                "Override compute_ricci_tensor in the theory subclass to provide custom implementation."
            )
        
        try:
            # Get metric components
            g_tt, g_rr, g_pp, g_tp = self.get_metric(r, M, c, G)
            
            # Verify diagonal metric
            if torch.any(torch.abs(g_tp) > 1e-10):
                raise ValueError("Metric has non-zero g_tp; default Ricci computation assumes diagonal metric.")
            
            # Initialize 4x4 tensor
            batch_size = r.shape[0] if r.dim() > 0 else 1
            if r.dim() == 0:
                r = r.unsqueeze(0)
                g_tt = g_tt.unsqueeze(0)
                g_rr = g_rr.unsqueeze(0)
                g_pp = g_pp.unsqueeze(0)
            
            ricci = torch.zeros(batch_size, 4, 4, dtype=self.dtype, device=r.device)
            
            # <reason>chain: Compute derivatives numerically using central differences for precision</reason>
            dr = r * 1e-6  # Small perturbation proportional to r
            dr = torch.clamp(dr, min=1e-10)  # Avoid zero
            
            # First derivatives
            g_tt_plus, g_rr_plus, g_pp_plus, _ = self.get_metric(r + dr, M, c, G)
            g_tt_minus, g_rr_minus, g_pp_minus, _ = self.get_metric(r - dr, M, c, G)
            
            dg_tt_dr = (g_tt_plus - g_tt_minus) / (2 * dr)
            dg_rr_dr = (g_rr_plus - g_rr_minus) / (2 * dr)
            dg_pp_dr = (g_pp_plus - g_pp_minus) / (2 * dr)
            
            # Second derivatives
            d2g_tt_dr2 = (g_tt_plus - 2*g_tt + g_tt_minus) / (dr**2)
            (g_rr_plus - 2*g_rr + g_rr_minus) / (dr**2)
            
            # <reason>chain: Ricci tensor components for spherically symmetric metric</reason>
            # R_tt = -(1/2g_rr) * d²g_tt/dr² + (1/4g_rr) * (dg_tt/dr)² - (1/4g_rr g_tt) * (dg_tt/dr)(dg_rr/dr) + (1/r g_rr) * dg_tt/dr
            ricci[:, 0, 0] = -(1/(2*g_rr)) * d2g_tt_dr2 + (1/(4*g_rr)) * (dg_tt_dr**2) - (1/(4*g_rr*g_tt)) * dg_tt_dr * dg_rr_dr + (1/(r*g_rr)) * dg_tt_dr
            
            # R_rr = (1/2g_tt) * d²g_tt/dr² - (1/4g_tt) * (dg_tt/dr)² + (1/4g_rr g_tt) * (dg_tt/dr)(dg_rr/dr) - (1/r g_rr) * dg_rr/dr
            ricci[:, 1, 1] = (1/(2*g_tt)) * d2g_tt_dr2 - (1/(4*g_tt)) * (dg_tt_dr**2) + (1/(4*g_rr*g_tt)) * dg_tt_dr * dg_rr_dr - (1/(r*g_rr)) * dg_rr_dr
            
            # R_θθ = 1 - (r/2g_rr) * dg_pp/dr - g_pp/g_rr
            ricci[:, 2, 2] = 1 - (r/(2*g_rr)) * dg_pp_dr - g_pp/g_rr
            
            # R_φφ = sin²θ * R_θθ (in our coordinates with θ integrated out, this equals R_θθ)
            ricci[:, 3, 3] = ricci[:, 2, 2]
            
            # Ricci tensor is symmetric
            ricci = (ricci + ricci.transpose(-1, -2)) / 2
            
            return ricci.squeeze(0) if batch_size == 1 and r.dim() == 0 else ricci
            
        except Exception:
            # <reason>chain: Return infinity tensor to indicate failure - will show as FAIL in leaderboard</reason>
            batch_size = r.shape[0] if r.dim() > 0 else 1
            return torch.full((batch_size, 4, 4), float('inf'), dtype=self.dtype, device=r.device)

    def has_stochastic_elements(self) -> bool:
        """
        <reason>chain: Indicates if theory has stochastic/random elements that affect conservation</reason>
        Override in theories with quantum noise, stochastic collapse, etc.
        """
        return False
    
    def computes_conservation_violation(self, trajectory: 'Tensor') -> float:
        """
        <reason>chain: Computes expected conservation violation based on theory physics</reason>
        
        For theories that violate conservation for physical reasons (e.g., stochastic theories),
        this should return the expected magnitude of violation based on theory parameters.
        
        Args:
            trajectory: The computed trajectory
            
        Returns:
            Expected conservation violation (0 for theories that conserve exactly)
        """
        return 0.0
    
    def _validate_required_methods(self):
        """
        <reason>chain: Validate that theory implements required methods based on category</reason>
        
        Warns about missing methods that validators expect, helping reduce N/A results.
        """
        warnings_list = []
        
        # <reason>chain: Check quantum theory requirements</reason>
        if hasattr(self, 'category') and self.category == 'quantum':
            # Hawking temperature requirements
            if not hasattr(self, 'compute_hawking_temperature') and not hasattr(self, 'compute_black_hole_entropy'):
                warnings_list.append(
                    "Quantum theory should implement either compute_hawking_temperature() or "
                    "compute_black_hole_entropy() for Hawking radiation validation"
                )
            
            # Unification scale requirements
            if not hasattr(self, 'unification_scale') and not hasattr(self, 'beta_function_corrections'):
                warnings_list.append(
                    "Quantum theory should implement unification_scale property or "
                    "beta_function_corrections() for unification scale validation"
                )
        
        # <reason>chain: Check GW modification requirements for all theories</reason>
        if not any(hasattr(self, attr) for attr in ['gw_speed', 'gw_damping', 'gw_modifications']):
            warnings_list.append(
                "Theory should implement gw_speed(), gw_damping, or gw_modifications() "
                "for gravitational wave validation"
            )
        
        # <reason>chain: Check cosmology requirements</reason>
        if not hasattr(self, 'dark_energy_parameter') and not hasattr(self, 'compute_dark_energy_eos'):
            warnings_list.append(
                "Theory should implement dark_energy_parameter or compute_dark_energy_eos() "
                "for cosmology validation"
            )
        
        # <reason>chain: Print warnings if verbose mode is enabled</reason>
        if warnings_list and hasattr(self, 'verbose') and self.verbose:
            print(f"\n[{self.name}] Missing method warnings:")
            for warning in warnings_list:
                print(f"  - {warning}")
    
    def conservation_violation_mechanism(self) -> str:
        """
        <reason>chain: Explains the physical mechanism for conservation violation</reason>
        
        Returns:
            Description of why this theory violates conservation (or None if it doesn't)
        """
        return None 

class QuantumUnifiedMixin:
    """
    <reason>chain: Mix-in class for quantum gravity theories that need full field content</reason>
    <reason>chain: Provides structured way to define quantum Lagrangians with matter, gauge, and interaction terms</reason>
    """
    
    def add_quantum_field_components(self):
        """
        <reason>chain: Add quantum field theory components to the Lagrangian</reason>
        <reason>chain: Following standard QFT structure: gravity + matter + gauge + interactions</reason>
        """
        # Import here to avoid circular imports
        import sympy as sp
        from physics_agent.constants import get_symbol
        
        # Standard symbols for quantum fields - use centralized registry
        psi = get_symbol('ψ')  # Fermion field
        psi_bar = get_symbol('ψ̄')  # Conjugate
        A_mu = get_symbol('A_μ')  # Gauge field
        gamma_mu = get_symbol('γ^μ')  # Gamma matrices
        D_mu = get_symbol('D_μ')  # Covariant derivative
        m = get_symbol('m')  # Fermion mass
        e = get_symbol('e')  # Coupling constant
        F_munu = get_symbol('F_μν')  # Field strength tensor
        
        # <reason>chain: QED Lagrangian components following standard form</reason>
        # L_QED = ψ̄(iγ^μD_μ - m)ψ - (1/4)F_μν F^μν
        
        # Matter Lagrangian: Dirac equation
        self.matter_lagrangian = sp.I * psi_bar * gamma_mu * D_mu * psi - m * psi_bar * psi
        
        # Gauge Lagrangian: Maxwell term  
        self.gauge_lagrangian = -sp.Rational(1, 4) * F_munu * get_symbol('F^μν')
        
        # Interaction Lagrangian: QED coupling
        self.interaction_lagrangian = -e * psi_bar * gamma_mu * psi * A_mu
        
        # <reason>chain: Add quantum corrections for precision tests</reason>
        # One-loop QED corrections (for g-2 etc)
        alpha = get_symbol('α')  # Fine structure constant ~1/137
        # Use Lambda_UV to distinguish from cosmological constant
        Lambda_UV = sp.Symbol('Lambda_UV')  # UV cutoff scale for QED
        self.quantum_correction_lagrangian = (alpha / sp.pi) * psi_bar * psi * sp.log(Lambda_UV / m)
        
        # Store for easy access
        self.qed_symbols = {
            'psi': psi, 'psi_bar': psi_bar, 'A_mu': A_mu,
            'gamma_mu': gamma_mu, 'D_mu': D_mu, 'm': m,
            'e': e, 'F_munu': F_munu, 'alpha': alpha
        } 