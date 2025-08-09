import pytest

from physics_agent.discovery.dsl import (
    VOCAB,
    OPERANDS,
    UNARY_FUNCS,
    BINARY_OPS,
    SYMBOL_TABLE,
    is_operand,
    is_unary,
    is_binary,
)
from physics_agent.constants import DISCOVERY_TOKENS, get_symbol
from sympy import Derivative, sin


def test_vocab_consistency_with_constants():
    # Ensure token sets are sourced from constants
    assert set(OPERANDS) == set(DISCOVERY_TOKENS['operands'])
    assert set(UNARY_FUNCS) == set(DISCOVERY_TOKENS['unary'])
    assert set(BINARY_OPS) == set(DISCOVERY_TOKENS['binary'])
    # VOCAB is ordered concatenation
    assert VOCAB[: len(OPERANDS)] == OPERANDS


def test_symbol_table_has_all_operands_and_ops():
    for tok in OPERANDS:
        assert tok in SYMBOL_TABLE, f"missing operand {tok}"
    for tok in UNARY_FUNCS + BINARY_OPS:
        assert tok in SYMBOL_TABLE, f"missing op {tok}"


def test_predicate_helpers_match_sets():
    for tok in OPERANDS:
        assert is_operand(tok)
    for tok in UNARY_FUNCS:
        assert is_unary(tok)
    for tok in BINARY_OPS:
        assert is_binary(tok)


def test_dt_unary_derivative_uses_registry_time():
    t = get_symbol('t')
    # Build a simple derivative DT(sin(t))
    kind, dt_fn = SYMBOL_TABLE['DT']
    assert kind == 'UNARY'
    expr = dt_fn(sin(t))
    assert isinstance(expr, Derivative)
    # Derivative of sin(t) wrt t simplifies to cos(t)
    assert str(expr.doit()) == 'cos(t)'


