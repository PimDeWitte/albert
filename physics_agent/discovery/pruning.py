"""
Physics-aware pruning utilities for symbolic expression search.

These checks reduce the search space before expensive simplification.
"""

from __future__ import annotations

from typing import Any
from sympy import Basic


def is_trivial(expr: Basic) -> bool:
    if expr is None:
        return True
    if expr.is_Number:
        return True
    try:
        if expr.simplify() == 0:
            return True
    except Exception:
        pass
    return False


def passes_pruning(expr: Basic) -> bool:
    """Fast, conservative pruning.

    - Reject None, numbers, or zero after cheap simplify
    - Avoid excessive nesting early
    """
    if is_trivial(expr):
        return False

    # Limit tree depth and size to prevent blow-ups
    try:
        size = expr.count_ops()
        # Use string representation as proxy for complexity
        str_repr = str(expr)
        depth_proxy = str_repr.count("(")
    except Exception:
        size = 0
        depth_proxy = 0

    if size > 200:
        return False
    if depth_proxy > 40:
        return False

    return True


__all__ = ["passes_pruning", "is_trivial"]


