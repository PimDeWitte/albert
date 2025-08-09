"""
Test that the discovery engine actually finds valid physical expressions.
"""

import pytest
from sympy import symbols, sqrt, exp, log, sin, cos
from physics_agent.discovery import DiscoveryEngine
from physics_agent.discovery.validators import validate_expression, validate_force_free_foliation
from physics_agent.discovery.known_formula_filter import filter_novel_candidates


def test_discovery_finds_valid_lagrangians():
    """Test that discovery can find simple valid Lagrangians."""
    # Use a restricted token set to make finding valid expressions more likely
    engine = DiscoveryEngine(
        max_symbols=5,
        device='cpu',
        operands_override=['m', 'v', 'x', 'k'],
        unary_override=[],
        binary_override=['MUL', 'DIV', 'ADD', 'SUB', 'POW'],
        mode='lagrangian'
    )
    
    results = engine.run(num_candidates=50, mcts_sims=10)
    
    # Check we found some results
    assert len(results) > 0, "No valid expressions found"
    
    # Check all results have required fields
    for r in results:
        assert 'expression' in r
        assert 'classification' in r
        assert 'mode' in r
        assert r['mode'] == 'lagrangian'
        
    # Check at least one is classified as a known type
    classifications = [r['classification'] for r in results]
    known_types = [
        'Newtonian mechanics', 
        'Special relativistic mechanics',
        'Field theory',
        'Unknown Lagrangian'
    ]
    assert any(c in known_types for c in classifications), \
        f"No known Lagrangian types found. Got: {set(classifications)}"


def test_discovery_finds_force_free_foliations():
    """Test that discovery can find valid force-free foliations."""
    # Use tokens specific to force-free problem
    engine = DiscoveryEngine(
        max_symbols=4,
        device='cpu',
        operands_override=['rho', 'z', '1', '0'],
        unary_override=['SQRT', 'EXP'],
        binary_override=['MUL', 'ADD', 'SUB', 'DIV', 'POW'],
        mode='force_free'
    )
    
    results = engine.run(num_candidates=100, mcts_sims=20)
    
    # Check we found some results
    assert len(results) > 0, "No valid force-free foliations found"
    
    # Verify each result passes force-free validation
    for r in results:
        expr_str = r['expression']
        # Skip trivial constants
        if expr_str in ['0', '1', 'E']:
            continue
            
        # Parse and validate
        from sympy import sympify
        try:
            expr = sympify(expr_str)
            validation = validate_force_free_foliation(expr)
            assert validation['valid'], f"Expression {expr_str} failed validation: {validation}"
        except Exception as e:
            pytest.fail(f"Failed to validate {expr_str}: {e}")
            
    # Check we found some known solutions
    expr_strings = [r['expression'] for r in results]
    known_solutions = ['rho**2', 'rho**2*z']  # Vertical and X-point
    found_known = [s for s in known_solutions if any(s in e for e in expr_strings)]
    assert len(found_known) > 0, \
        f"Did not find any known solutions. Got: {expr_strings[:10]}..."


def test_novel_candidate_filtering():
    """Test that known formula filtering works."""
    # Create some mock candidates
    candidates = [
        {'expression': 'm*v**2/2', 'classification': 'Newtonian'},  # Known
        {'expression': 'm*v**3/3', 'classification': 'Unknown'},    # Novel
        {'expression': 'rho**2', 'classification': 'Force-free'},   # Known
        {'expression': 'rho**3', 'classification': 'Unknown'},      # Novel
    ]
    
    # Filter for Lagrangians
    novel = filter_novel_candidates(candidates, category='lagrangians')
    novel_exprs = [c['expression'] for c in novel]
    assert 'm*v**2/2' not in novel_exprs  # Known kinetic energy filtered
    assert 'm*v**3/3' in novel_exprs      # Novel expression kept
    
    # Filter for force-free
    novel = filter_novel_candidates(candidates, category='force_free_foliations')
    novel_exprs = [c['expression'] for c in novel]
    assert 'rho**2' not in novel_exprs    # Known vertical field filtered
    assert 'rho**3' in novel_exprs        # Novel expression kept


def test_expression_parsing_consistency():
    """Test that token parsing produces valid SymPy expressions."""
    from physics_agent.discovery.dsl import SYMBOL_TABLE, is_operand
    
    # Test operands parse correctly
    test_operands = ['rho', 'z', '1', '0', 't', 'x']
    for op in test_operands:
        if op in SYMBOL_TABLE and is_operand(op):
            sym = SYMBOL_TABLE[op]
            assert sym is not None, f"Operand {op} has None in symbol table"
            # Verify it's a valid SymPy object
            assert hasattr(sym, 'free_symbols') or str(sym) in ['0', '1'], \
                f"Operand {op} -> {sym} is not a valid SymPy expression"
                
    # Test a simple expression builds correctly
    engine = DiscoveryEngine(max_symbols=3, device='cpu')
    
    # Manually test token sequence: rho z MUL -> rho*z
    tokens = ['rho', 'z', 'MUL']
    token_ids = [engine.token_to_idx.get(t, -1) for t in tokens]
    
    # Skip if tokens not in vocabulary
    if all(tid >= 0 for tid in token_ids):
        expr = engine._tokens_to_expr(tokens)
        assert expr is not None, "Failed to parse rho z MUL"
        assert str(expr) == 'rho*z', f"Expected 'rho*z', got '{expr}'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
