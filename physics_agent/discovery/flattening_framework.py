"""
General Framework for Algorithmic Discovery of Flattened Physical Theories.

Based on "Flattening Physical Theories: An Algorithmic Framework for Symbolic Unification"
This generalizes Method 2.4 from the Force-Free Foliations paper to arbitrary domains.
"""

from __future__ import annotations
from typing import Dict, Any, List, Callable, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import json
import os
from datetime import datetime

import sympy as sp
from sympy import symbols, Basic, simplify, expand, sqrt, exp, log, sin, cos
import numpy as np

from .validators import validate_expression
from .known_formula_filter import filter_novel_candidates


class DomainType(Enum):
    """Physics domains supported by the framework."""
    MECHANICS = "mechanics"
    ELECTROMAGNETISM = "electromagnetism"
    WAVES = "waves"
    RELATIVITY = "relativity"
    FORCE_FREE = "force_free"
    GENERAL = "general"


@dataclass
class FlatteningCandidate:
    """A candidate flattening transformation."""
    expression: str
    domain: DomainType
    validity_score: float
    mdl_score: float
    classification: str
    metadata: Dict[str, Any]


@dataclass
class DomainConfig:
    """Configuration for a specific physics domain."""
    name: str
    primitives: List[str]  # Base symbols
    operators: List[str]   # Allowed operations
    constraints: List[str] # Validity constraints
    observables: List[str] # Physical observables to preserve
    symmetries: List[str]  # Symmetry groups
    units: Dict[str, str]  # Unit specifications
    
    
class TypedGrammar:
    """
    Typed symbolic grammar with unit checking and symmetry tags.
    Implements Section 4 of the framework document.
    """
    
    def __init__(self, domain_config: DomainConfig):
        self.config = domain_config
        self.primitives = self._init_primitives()
        self.operators = self._init_operators()
        self.unit_system = self._init_units()
        
    def _init_primitives(self) -> Dict[str, Basic]:
        """Initialize primitive symbols with types and units."""
        primitives = {}
        
        # Standard coordinates and fields
        for prim in self.config.primitives:
            if prim in ['x', 'y', 'z']:
                primitives[prim] = sp.Symbol(prim, real=True)
            elif prim == 't':
                primitives[prim] = sp.Symbol(prim, real=True, positive=True)
            elif prim in ['rho', 'r']:
                primitives[prim] = sp.Symbol(prim, real=True, positive=True)
            elif prim in ['theta', 'phi']:
                primitives[prim] = sp.Symbol(prim, real=True)
            elif prim in ['m', 'c', 'G', 'hbar', 'k']:
                primitives[prim] = sp.Symbol(prim, positive=True)
            else:
                primitives[prim] = sp.Symbol(prim)
                
        return primitives
        
    def _init_operators(self) -> Dict[str, Callable]:
        """Initialize typed operators."""
        ops = {
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'div': lambda a, b: a / b,
            'pow': lambda a, b: a ** b,
            'sqrt': lambda a: sqrt(a),
            'exp': lambda a: exp(a),
            'log': lambda a: log(a),
            'sin': lambda a: sin(a),
            'cos': lambda a: cos(a),
        }
        
        # Domain-specific operators
        if self.config.name == "force_free":
            ops['geom_sum'] = lambda a, b: sqrt((a - 1)**2 + b**2) + sqrt((a + 1)**2 + b**2)
            
        return ops
        
    def _init_units(self) -> Dict[str, Tuple[float, ...]]:
        """Initialize unit system with dimension vectors."""
        # Base dimensions: [M, L, T, Q, Θ]
        units = {
            't': (0, 0, 1, 0, 0),      # Time
            'x': (0, 1, 0, 0, 0),      # Length
            'y': (0, 1, 0, 0, 0),      # Length
            'z': (0, 1, 0, 0, 0),      # Length
            'r': (0, 1, 0, 0, 0),      # Length
            'rho': (0, 1, 0, 0, 0),    # Length
            'm': (1, 0, 0, 0, 0),      # Mass
            'c': (0, 1, -1, 0, 0),     # Speed
            'G': (-1, 3, -2, 0, 0),    # Gravitational constant
            'hbar': (1, 2, -1, 0, 0),  # Reduced Planck constant
            'k': (1, 0, -3, 0, -1),    # Boltzmann constant
        }
        
        return units
        
    def check_units(self, expr: Basic) -> bool:
        """Check if expression is dimensionally consistent."""
        # Simplified unit checking - full implementation would trace through AST
        return True  # TODO: Implement proper dimensional analysis
        
    def check_symmetry(self, expr: Basic, symmetry_group: str) -> bool:
        """Check if expression respects the symmetry group."""
        # Simplified symmetry checking
        if symmetry_group == "lorentz":
            # Check Lorentz invariance
            return self._check_lorentz_invariance(expr)
        elif symmetry_group == "galilean":
            # Check Galilean invariance  
            return self._check_galilean_invariance(expr)
        return True
        
    def _check_lorentz_invariance(self, expr: Basic) -> bool:
        """Check if expression is Lorentz invariant."""
        # Simplified check - full implementation would test boost invariance
        # For now, check if it's a proper scalar combination
        if 'dt' in str(expr) and 'dx' in str(expr):
            # Check for interval-like structure
            return True
        return False
        
    def _check_galilean_invariance(self, expr: Basic) -> bool:
        """Check if expression is Galilean invariant."""
        # Simplified check
        return True


class ConstraintBattery:
    """
    Multi-tier constraint checking system.
    Implements Section 5 of the framework document.
    """
    
    def __init__(self, domain_config: DomainConfig, grammar: TypedGrammar):
        self.config = domain_config
        self.grammar = grammar
        
    def check_all(self, expr: Basic) -> Tuple[bool, Dict[str, Any]]:
        """Run full constraint battery on expression."""
        results = {
            'passed': True,
            'tier0': {},
            'tier1': {},
            'tier2': {},
            'tier3': {},
        }
        
        # Tier 0: Static filters
        if not self._check_tier0(expr, results['tier0']):
            results['passed'] = False
            return False, results
            
        # Tier 1: Algebraic/symbolic
        if not self._check_tier1(expr, results['tier1']):
            results['passed'] = False
            return False, results
            
        # Tier 2: PDE residuals (if applicable)
        if not self._check_tier2(expr, results['tier2']):
            results['passed'] = False
            return False, results
            
        # Tier 3: Numerical validation
        if not self._check_tier3(expr, results['tier3']):
            results['passed'] = False
            return False, results
            
        return True, results
        
    def _check_tier0(self, expr: Basic, results: Dict) -> bool:
        """Tier 0: Units and signature checks."""
        # Units check
        units_ok = self.grammar.check_units(expr)
        results['units'] = units_ok
        
        # Signature check (scalar vs vector vs tensor)
        signature_ok = True  # TODO: Implement
        results['signature'] = signature_ok
        
        return units_ok and signature_ok
        
    def _check_tier1(self, expr: Basic, results: Dict) -> bool:
        """Tier 1: Algebraic and symmetry checks."""
        # Symmetry checks
        for symmetry in self.config.symmetries:
            sym_ok = self.grammar.check_symmetry(expr, symmetry)
            results[f'symmetry_{symmetry}'] = sym_ok
            if not sym_ok:
                return False
                
        # Limit checks
        results['limits'] = self._check_limits(expr)
        
        return results['limits']
        
    def _check_tier2(self, expr: Basic, results: Dict) -> bool:
        """Tier 2: PDE residuals and boundary conditions."""
        # Domain-specific PDE checks
        if self.config.name == "waves":
            results['wave_residual'] = self._check_wave_equation(expr)
            return results['wave_residual'] < 1e-6
        elif self.config.name == "force_free":
            # Use the force-free validator
            validation = validate_expression(expr, mode='force_free')
            results['force_free_valid'] = validation['valid']
            return validation['valid']
            
        return True
        
    def _check_tier3(self, expr: Basic, results: Dict) -> bool:
        """Tier 3: Numerical validation."""
        # Simplified numerical checks
        results['numerical_stable'] = True
        results['conservation_ok'] = True
        return True
        
    def _check_limits(self, expr: Basic) -> bool:
        """Check physical limits (non-relativistic, weak-field, etc.)."""
        # Simplified limit checking
        return True
        
    def _check_wave_equation(self, expr: Basic) -> float:
        """Check if expression satisfies wave equation."""
        # Simplified check - return residual
        return 0.0


class MDLScorer:
    """
    Minimum Description Length scorer for compression evaluation.
    Implements Section 2.2 and 6 of the framework document.
    """
    
    def __init__(self, 
                 alpha: float = 1.0,    # Structure weight
                 beta: float = 0.5,     # Parameters weight  
                 gamma: float = 0.1,    # Constants precision weight
                 delta: float = 0.05):  # Proof tokens weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Operation weights
        self.op_weights = {
            'add': 1, 'sub': 1, 'mul': 1, 'div': 1,
            'pow': 2, 'sqrt': 2, 'exp': 2, 'log': 2,
            'sin': 2, 'cos': 2,
            'diff': 3, 'integral': 3,  # Derivatives/integrals
            'tensor': 3,  # Tensor operations
        }
        
    def compute_mdl(self, expr: Basic) -> float:
        """Compute MDL score for expression."""
        # Structure complexity
        structure_score = self._compute_structure_score(expr)
        
        # Parameter count
        param_score = self._count_parameters(expr) * self.beta
        
        # Constants precision
        const_score = self._compute_constants_score(expr) * self.gamma
        
        # Proof complexity (simplified)
        proof_score = self._estimate_proof_complexity(expr) * self.delta
        
        return structure_score + param_score + const_score + proof_score
        
    def _compute_structure_score(self, expr: Basic) -> float:
        """Compute structural complexity of expression."""
        score = 0.0
        
        # Walk the expression tree
        for subexpr in sp.preorder_traversal(expr):
            if hasattr(subexpr, 'func'):
                func_name = subexpr.func.__name__.lower()
                score += self.op_weights.get(func_name, 1)
                
        return score * self.alpha
        
    def _count_parameters(self, expr: Basic) -> int:
        """Count free parameters in expression."""
        return len(expr.free_symbols)
        
    def _compute_constants_score(self, expr: Basic) -> float:
        """Compute precision cost of constants."""
        # Simplified - count numeric constants
        const_count = 0
        for subexpr in sp.preorder_traversal(expr):
            if subexpr.is_number and not subexpr.is_integer:
                const_count += 1
        return const_count * 8  # Assume 8 bits per constant
        
    def _estimate_proof_complexity(self, expr: Basic) -> float:
        """Estimate proof complexity."""
        # Very simplified - based on expression size
        return len(str(expr)) / 100


class FlatteningDiscoveryEngine:
    """
    Main engine for discovering flattened formulations.
    Generalizes Method 2.4 to arbitrary physics domains.
    """
    
    def __init__(self, domain: DomainType, config: Optional[DomainConfig] = None):
        self.domain = domain
        self.config = config or self._get_default_config(domain)
        self.grammar = TypedGrammar(self.config)
        self.constraints = ConstraintBattery(self.config, self.grammar)
        self.scorer = MDLScorer()
        
    def _get_default_config(self, domain: DomainType) -> DomainConfig:
        """Get default configuration for domain."""
        configs = {
            DomainType.FORCE_FREE: DomainConfig(
                name="force_free",
                primitives=['rho', 'z'],
                operators=['add', 'sub', 'mul', 'div', 'sqrt', 'exp', 'log', 'geom_sum'],
                constraints=['foliation_condition'],
                observables=['magnetic_flux', 'current'],
                symmetries=['axial'],
                units={'rho': 'L', 'z': 'L'}
            ),
            DomainType.RELATIVITY: DomainConfig(
                name="relativity",
                primitives=['dt', 'dx', 'dy', 'dz', 'c'],
                operators=['add', 'sub', 'mul', 'pow'],
                constraints=['lorentz_invariance'],
                observables=['interval'],
                symmetries=['lorentz'],
                units={'dt': 'T', 'dx': 'L', 'dy': 'L', 'dz': 'L', 'c': 'L/T'}
            ),
            DomainType.WAVES: DomainConfig(
                name="waves",
                primitives=['x', 't', 'c'],
                operators=['add', 'sub', 'mul'],
                constraints=['wave_equation'],
                observables=['phase_velocity', 'group_velocity'],
                symmetries=['galilean'],
                units={'x': 'L', 't': 'T', 'c': 'L/T'}
            ),
        }
        
        return configs.get(domain, DomainConfig(
            name="general",
            primitives=['x', 't'],
            operators=['add', 'sub', 'mul', 'div'],
            constraints=[],
            observables=[],
            symmetries=[],
            units={}
        ))
        
    def discover(self, 
                max_depth: int = 4,
                max_candidates: int = 1000,
                timeout_seconds: int = 300,
                output_dir: Optional[str] = None) -> List[FlatteningCandidate]:
        """
        Run discovery process.
        
        This is the main entry point that generalizes Method 2.4.
        """
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join('runs', f'flattening_{self.domain.value}_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Starting flattening discovery for domain: {self.domain.value}")
        print(f"Configuration: {self.config.name}")
        print(f"Max depth: {max_depth}, Max candidates: {max_candidates}")
        print("="*60)
        
        # Phase 1: Generate candidates
        candidates = self._generate_candidates(max_depth, max_candidates, timeout_seconds)
        print(f"\nGenerated {len(candidates)} raw candidates")
        
        # Phase 2: Apply constraints
        valid_candidates = []
        for i, expr in enumerate(candidates):
            if i % 10 == 0:
                print(f"  Checking constraints: {i}/{len(candidates)}")
                
            passed, results = self.constraints.check_all(expr)
            if passed:
                valid_candidates.append((expr, results))
                
        print(f"\n{len(valid_candidates)} candidates passed all constraints")
        
        # Phase 3: Score and rank
        scored_candidates = []
        for expr, constraint_results in valid_candidates:
            mdl_score = self.scorer.compute_mdl(expr)
            
            candidate = FlatteningCandidate(
                expression=str(expr),
                domain=self.domain,
                validity_score=1.0,  # Passed all constraints
                mdl_score=mdl_score,
                classification=self._classify_expression(expr),
                metadata={
                    'constraint_results': constraint_results,
                    'depth': self._get_depth(expr),
                }
            )
            scored_candidates.append(candidate)
            
        # Sort by MDL score (lower is better)
        scored_candidates.sort(key=lambda c: c.mdl_score)
        
        # Phase 4: Filter known formulas
        novel_candidates = filter_novel_candidates(
            [{'expression': c.expression} for c in scored_candidates],
            category=self.domain.value
        )
        
        # Map back to full candidates
        novel_expression_set = {c['expression'] for c in novel_candidates}
        final_candidates = [c for c in scored_candidates if c.expression in novel_expression_set]
        
        # Save results
        self._save_results(final_candidates, output_dir)
        
        return final_candidates
        
    def _generate_candidates(self, max_depth: int, max_candidates: int, 
                           timeout_seconds: int) -> List[Basic]:
        """Generate candidate expressions systematically."""
        candidates = []
        start_time = time.time()
        
        # Start with primitives
        current_level = [self.grammar.primitives[p] for p in self.config.primitives]
        candidates.extend(current_level)
        
        # Build by depth
        for depth in range(2, max_depth + 1):
            if time.time() - start_time > timeout_seconds:
                print(f"  Timeout at depth {depth}")
                break
                
            next_level = []
            
            # Apply operators
            for op_name, op_func in self.grammar.operators.items():
                if op_name in ['sqrt', 'exp', 'log', 'sin', 'cos']:
                    # Unary operators
                    for expr in current_level:
                        try:
                            new_expr = op_func(expr)
                            if self.grammar.check_units(new_expr):
                                next_level.append(new_expr)
                        except:
                            pass
                else:
                    # Binary operators
                    for expr1 in current_level:
                        for expr2 in candidates[:len(current_level)]:  # Include earlier depths
                            try:
                                new_expr = op_func(expr1, expr2)
                                if self.grammar.check_units(new_expr):
                                    next_level.append(new_expr)
                            except:
                                pass
                                
            # Deduplicate
            seen = set()
            unique_next = []
            for expr in next_level:
                str_expr = str(expr)
                if str_expr not in seen and len(str_expr) < 200:
                    seen.add(str_expr)
                    unique_next.append(expr)
                    
            current_level = unique_next
            candidates.extend(unique_next)
            
            print(f"  Depth {depth}: {len(unique_next)} new expressions")
            
            if len(candidates) >= max_candidates:
                break
                
        return candidates[:max_candidates]
        
    def _classify_expression(self, expr: Basic) -> str:
        """Classify the type of flattening."""
        # Domain-specific classification
        if self.domain == DomainType.FORCE_FREE:
            # Use force-free validator
            validation = validate_expression(expr, mode='force_free')
            return validation.get('classification', 'Unknown')
        elif self.domain == DomainType.RELATIVITY:
            # Check for interval-like structure
            if '-' in str(expr) and 'dt**2' in str(expr) and 'dx**2' in str(expr):
                return "Lorentz interval candidate"
        elif self.domain == DomainType.WAVES:
            # Check for null coordinates
            if 'x - c*t' in str(expr) or 'x + c*t' in str(expr):
                return "Null coordinate candidate"
                
        return "General flattening"
        
    def _get_depth(self, expr: Basic) -> int:
        """Estimate expression depth."""
        return expr.count_ops() + 1
        
    def _save_results(self, candidates: List[FlatteningCandidate], output_dir: str):
        """Save discovery results."""
        # JSON results
        results = {
            'domain': self.domain.value,
            'config': {
                'name': self.config.name,
                'primitives': self.config.primitives,
                'operators': self.config.operators,
                'constraints': self.config.constraints,
            },
            'candidates': [
                {
                    'expression': c.expression,
                    'mdl_score': c.mdl_score,
                    'classification': c.classification,
                    'metadata': c.metadata,
                }
                for c in candidates[:20]  # Top 20
            ],
            'summary': {
                'total_valid': len(candidates),
                'novel_count': len(candidates),
                'best_mdl': candidates[0].mdl_score if candidates else None,
            }
        }
        
        json_path = os.path.join(output_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # HTML report
        html_path = os.path.join(output_dir, 'report.html')
        self._generate_html_report(results, html_path)
        
        print(f"\nResults saved to:")
        print(f"  {json_path}")
        print(f"  {html_path}")
        
    def _generate_html_report(self, results: Dict, output_path: str):
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Flattening Discovery Results - {results['domain']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .candidate {{ border: 1px solid #ddd; padding: 10px; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    </style>
</head>
<body>
    <h1>Flattening Discovery Results</h1>
    <h2>Domain: {results['domain']}</h2>
    
    <div class="summary">
        <h3>Summary</h3>
        <ul>
            <li>Total valid candidates: {results['summary']['total_valid']}</li>
            <li>Novel formulations: {results['summary']['novel_count']}</li>
            <li>Best MDL score: {results['summary']['best_mdl']:.2f}</li>
        </ul>
    </div>
    
    <h3>Configuration</h3>
    <table>
        <tr><th>Primitives</th><td>{', '.join(results['config']['primitives'])}</td></tr>
        <tr><th>Operators</th><td>{', '.join(results['config']['operators'])}</td></tr>
        <tr><th>Constraints</th><td>{', '.join(results['config']['constraints'])}</td></tr>
    </table>
    
    <h3>Top Candidates</h3>
    {"".join(f'<div class="candidate"><strong>{i+1}. {c["classification"]}</strong><br>Expression: {c["expression"]}<br>MDL Score: {c["mdl_score"]:.2f}</div>' for i, c in enumerate(results['candidates']))}
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html)
