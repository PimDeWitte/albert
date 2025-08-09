"""
Heuristic Lagrangian classifier for math-space candidates.
"""

from __future__ import annotations

from sympy import Basic, Derivative, symbols, Function


def classify_lagrangian(expr: Basic) -> str:
    """Classify expression into a rough Lagrangian type.

    Categories:
      - Velocity-dependent Lagrangian: explicit dependence on time derivatives
      - Potential-energy-like Lagrangian: depends on coordinate but not velocity
      - General/Other Lagrangian: otherwise
    """
    # Look for any time derivatives
    if any(isinstance(a, Derivative) for a in expr.atoms(Derivative)):
        return "Velocity-dependent Lagrangian"

    # Weak check for coordinate dependence
    t = symbols("t")
    q = Function("q")(t)
    if q in expr.atoms() or q in getattr(expr, "free_symbols", set()):
        return "Potential-energy-like Lagrangian"

    return "General/Other Lagrangian"


__all__ = ["classify_lagrangian"]


