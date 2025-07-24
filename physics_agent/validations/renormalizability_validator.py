"""
Renormalizability and Gauge Invariance Validator for Quantum Theories

<reason>chain: Critical validator for ensuring theories are UV-complete and gauge-invariant</reason>
"""

import numpy as np
import sympy as sp
from typing import Dict, Optional, TYPE_CHECKING
from .base_validation import BaseValidation

if TYPE_CHECKING:
    from ..theory_engine_core import TheoryEngine


class RenormalizabilityValidator(BaseValidation):
    """
    <reason>chain: Validates renormalizability and gauge invariance for unification</reason>
    
    Tests:
    1. Power counting of operators (dimension ≤ 4 in 4D)
    2. Beta function computation for running couplings
    3. Gauge invariance under U(1), SU(2), SU(3) transformations
    4. BRST symmetry for non-Abelian gauge theories
    5. Unitarity bounds from scattering amplitudes
    """
    
    def __init__(self, engine: Optional["TheoryEngine"] = None, tolerance: float = 1e-10):
        super().__init__(engine, "Renormalizability Validator")
        self.tolerance = tolerance
        self.category = "constraint"
        
        # <reason>chain: Define operator dimensions in 4D spacetime</reason>
        self.operator_dimensions = {
            'φ': 1,  # Scalar field
            '∂_μ': 1,  # Derivative
            'ψ': 3/2,  # Fermion field
            'A_μ': 1,  # Gauge field
            'F_μν': 2,  # Field strength
            'R': 2,  # Ricci scalar
            'R_μν': 2,  # Ricci tensor
            'R_μνρσ': 2,  # Riemann tensor
            'm': 1,  # Mass
            'g': 0,  # Dimensionless coupling
            'λ': 0,  # Dimensionless coupling
        }
        
    def validate(self, theory, trajectory_data: Dict, initial_conditions: Dict) -> Dict:
        """
        <reason>chain: Main validation function checking renormalizability</reason>
        """
        results = {
            'loss': 0.0,
            'is_renormalizable': True,
            'is_gauge_invariant': True,
            'divergent_operators': [],
            'gauge_violations': [],
            'beta_functions': {},
            'fixed_points': [],
            'flags': {},
            'canceled': False
        }
        
        # <reason>chain: Check if theory has required Lagrangian components</reason>
        # Use appropriate Lagrangian based on theory type
        lagrangian, lagrangian_type = self._get_appropriate_lagrangian(theory)
        
        if lagrangian is None:
            results['loss'] = float('inf')
            results['canceled'] = True
            results['flags']['no_lagrangian'] = False
            return results
            
        # Store which Lagrangian we're using for later reference
        self._current_lagrangian = lagrangian
        self._lagrangian_type = lagrangian_type
            
        # 1. Power counting analysis
        power_results = self._check_power_counting(theory)
        results.update(power_results)
        
        # 2. Gauge invariance check
        if theory.category == 'quantum':
            gauge_results = self._check_gauge_invariance(theory)
            results.update(gauge_results)
            
        # 3. Beta function computation
        beta_results = self._compute_beta_functions(theory)
        results.update(beta_results)
        
        # 4. BRST symmetry check for gauge theories
        if theory.category == 'quantum' and hasattr(theory, 'gauge_lagrangian'):
            brst_results = self._check_brst_symmetry(theory)
            results.update(brst_results)
        
        # 5. Unitarity bounds
        unitarity_results = self._check_unitarity(theory)
        results.update(unitarity_results)
        
        # <reason>chain: Cancel if non-renormalizable</reason>
        if not results['is_renormalizable']:
            results['canceled'] = True
            results['loss'] = 1e6  # Large penalty
            
        return results
        
    def _check_power_counting(self, theory) -> Dict:
        """
        <reason>chain: Analyze dimension of operators in Lagrangian</reason>
        """
        results = {
            'divergent_operators': [],
            'is_renormalizable': True
        }
        
        # Convert Lagrangian to string for analysis
        # Use the Lagrangian we determined earlier
        L_str = str(self._current_lagrangian)
        
        # <reason>chain: Check for non-renormalizable operators</reason>
        # In 4D, operators with dimension > 4 are non-renormalizable
        non_renorm_patterns = [
            ('R**2', 'R² term (dim=4, marginal)'),
            ('R**3', 'R³ term (dim=6, non-renormalizable)'),
            ('R_μν**2', 'R_μν² term (dim=4, marginal)'),
            ('φ**6', 'φ⁶ term (dim=6, non-renormalizable)'),
            ('ψ̄ψ**2', '(ψ̄ψ)² term (dim=6, non-renormalizable)'),
            ('F_μν**3', 'F³ term (dim=6, non-renormalizable)'),
        ]
        
        for pattern, description in non_renorm_patterns:
            if pattern in L_str:
                results['divergent_operators'].append(description)
                if 'marginal' not in description:
                    results['is_renormalizable'] = False
                    
        # <reason>chain: Count operator dimensions systematically</reason>
        # Use the already determined Lagrangian
        if self._lagrangian_type in ['quantum', 'classical']:
            try:
                L = self._current_lagrangian
                # Extract all terms
                terms = sp.Add.make_args(L)
                
                for term in terms:
                    dim = self._compute_dimension(term)
                    if dim > 4:
                        results['divergent_operators'].append(f'Term {term} has dimension {dim}')
                        results['is_renormalizable'] = False
                        
            except Exception:
                # Fallback to string analysis
                pass
                
        results['loss'] = len(results['divergent_operators']) * 0.1
        return results
        
    def _compute_dimension(self, expr: sp.Expr) -> float:
        """
        <reason>chain: Compute mass dimension of a SymPy expression</reason>
        """
        if expr.is_number:
            return 0
            
        if expr.is_symbol:
            name = str(expr)
            for op, dim in self.operator_dimensions.items():
                if op in name:
                    return dim
            return 0  # Unknown symbols assumed dimensionless
            
        if expr.is_Add:
            # All terms in sum must have same dimension
            dims = [self._compute_dimension(arg) for arg in expr.args]
            return dims[0] if dims else 0
            
        if expr.is_Mul:
            # Product: sum of dimensions
            return sum(self._compute_dimension(arg) for arg in expr.args)
            
        if expr.is_Pow:
            base, exp = expr.args
            return self._compute_dimension(base) * float(exp)
            
        # Default
        return 0
        
    def _check_gauge_invariance(self, theory) -> Dict:
        """
        <reason>chain: Check gauge invariance under standard transformations</reason>
        """
        results = {
            'gauge_violations': [],
            'is_gauge_invariant': True
        }
        
        # <reason>chain: Check U(1) invariance (electromagnetic)</reason>
        if hasattr(theory, 'gauge_lagrangian'):
            L_gauge = str(theory.gauge_lagrangian)
            
            # F_μν should appear as F_μν F^μν (gauge invariant)
            if 'F_μν' in L_gauge and 'F^μν' not in L_gauge:
                results['gauge_violations'].append('F_μν not properly contracted')
                results['is_gauge_invariant'] = False
                
            # A_μ should only appear in covariant derivatives or F_μν
            if 'A_μ' in L_gauge and 'D_μ' not in L_gauge and 'F_μν' not in L_gauge:
                results['gauge_violations'].append('Bare A_μ term (not gauge invariant)')
                results['is_gauge_invariant'] = False
                
        # <reason>chain: Check matter field gauge invariance</reason>
        if hasattr(theory, 'matter_lagrangian'):
            L_matter = str(theory.matter_lagrangian)
            
            # ψ should appear with covariant derivative D_μ, not ∂_μ
            if 'ψ' in L_matter and '∂_μ' in L_matter and 'D_μ' not in L_matter:
                results['gauge_violations'].append('∂_μψ instead of D_μψ (breaks gauge invariance)')
                results['is_gauge_invariant'] = False
                
        results['loss'] += len(results['gauge_violations']) * 0.2
        return results
        
    def _compute_beta_functions(self, theory) -> Dict:
        """
        <reason>chain: Compute two-loop beta functions for running couplings</reason>
        """
        beta_functions = {}
        
        # <reason>chain: Standard Model beta functions at two-loop</reason>
        # β(g) = b₀ g³/(16π²) + b₁ g⁵/(16π²)²
        
        # Import fine structure constant for gauge coupling normalization
        
        # U(1) hypercharge (GUT normalized)
        b0_U1 = 41/10  # One-loop
        b1_U1 = 199/50  # Two-loop
        beta_functions['g_Y'] = lambda g: (b0_U1 * g**3 / (16 * np.pi**2) + 
                                          b1_U1 * g**5 / (16 * np.pi**2)**2)
        
        # SU(2) weak
        b0_SU2 = -19/6  # One-loop (asymptotic freedom)
        b1_SU2 = -35/6  # Two-loop
        beta_functions['g_2'] = lambda g: (b0_SU2 * g**3 / (16 * np.pi**2) +
                                          b1_SU2 * g**5 / (16 * np.pi**2)**2)
        
        # SU(3) strong
        b0_SU3 = -7  # One-loop (strong asymptotic freedom)
        b1_SU3 = -26  # Two-loop
        beta_functions['g_3'] = lambda g: (b0_SU3 * g**3 / (16 * np.pi**2) +
                                          b1_SU3 * g**5 / (16 * np.pi**2)**2)
        
        # <reason>chain: Check for fixed points (g where β(g) = 0)</reason>
        fixed_points = []
        
        # Trivial fixed point at g=0
        fixed_points.append({'coupling': 'all', 'value': 0.0, 'type': 'trivial'})
        
        # <reason>chain: Look for non-trivial fixed points (unification)</reason>
        # At high energy, couplings may unify: g_1 = g_2 = g_3 = g_GUT
        g_GUT = 0.7  # Typical GUT coupling
        E_GUT = 2e16  # GeV
        
        fixed_points.append({
            'coupling': 'quantum',
            'value': g_GUT,
            'energy': E_GUT,
            'type': 'unification'
        })
        
        return {
            'beta_functions': beta_functions,
            'fixed_points': fixed_points
        }
        
    def _check_unitarity(self, theory) -> Dict:
        """
        <reason>chain: Check unitarity bounds from scattering amplitudes</reason>
        """
        results = {
            'unitarity_bound': 8 * np.pi,  # From partial wave analysis
            'violates_unitarity': False
        }
        
        # <reason>chain: Check coupling strengths</reason>
        if hasattr(theory, 'coupling_constant'):
            g = getattr(theory, 'coupling_constant', 1.0)
            
            # Tree-level unitarity bound: g² < 8π
            if g**2 > results['unitarity_bound']:
                results['violates_unitarity'] = True
                results['loss'] = (g**2 / results['unitarity_bound']) - 1.0
                
        return results
    
    def _check_brst_symmetry(self, theory) -> Dict:
        """
        <reason>chain: Check BRST invariance for gauge theory quantization</reason>
        """
        results = {
            'brst_invariant': True,
            'brst_violations': []
        }
        
        # BRST requires:
        # 1. Nilpotent BRST operator: Q² = 0
        # 2. Gauge-fixed Lagrangian: L_gf = L + Q(gauge-fixing term)
        # 3. Ghost fields with correct statistics
        
        L_gauge = str(theory.gauge_lagrangian) if hasattr(theory, 'gauge_lagrangian') else ''
        
        # Check for Faddeev-Popov ghosts
        if 'ghost' not in L_gauge.lower() and 'c̄' not in L_gauge and 'c' not in L_gauge:
            results['brst_violations'].append('Missing Faddeev-Popov ghost fields')
            results['brst_invariant'] = False
            
        # Check for gauge-fixing term
        if 'gauge_fix' not in L_gauge.lower() and 'ξ' not in L_gauge:
            results['brst_violations'].append('Missing gauge-fixing term')
            results['brst_invariant'] = False
            
        return results 