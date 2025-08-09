import json
import os
import pytest

from physics_agent.discovery import DiscoveryEngine


def test_discovery_basic_runs_small():
    engine = DiscoveryEngine(max_symbols=4, device='cpu')
    results = engine.run(num_candidates=3, mcts_sims=2)
    assert isinstance(results, list)
    for r in results:
        assert 'expression' in r and 'classification' in r


def test_discovery_with_token_restriction():
    # Restrict to a tiny token space that is guaranteed valid
    operands = ['t']
    unary = ['SIN']
    binary = ['ADD']
    engine = DiscoveryEngine(max_symbols=3, device='cpu', operands_override=operands, unary_override=unary, binary_override=binary)
    results = engine.run(num_candidates=5, mcts_sims=2)
    assert isinstance(results, list)


