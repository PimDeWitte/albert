"""
Domain-Specific Language (DSL) for math-space discovery.

Provides a finite set of physics primitives (symbols, fields, constants)
and a minimal token vocabulary for stack-based (RPN) expression building.
"""

from __future__ import annotations

from sympy import (
    Function,
    Derivative,
    sin,
    cos,
    sqrt,
    exp,
)
import sympy
from physics_agent.constants import get_symbol, register_physics_symbol, DISCOVERY_TOKENS

"""
Use unified physics symbols from constants.py for consistency.
If a symbol is missing from the registry, get_symbol(name) will create it.
"""

# Coordinates and time (registry-backed)
t = get_symbol("t")
r = get_symbol("r")
theta = get_symbol("theta")
phi = get_symbol("phi")
x = get_symbol("x")
y = get_symbol("y")
z = get_symbol("z")

# Core constants as symbols (not numeric values here)
m = get_symbol("m")
c = get_symbol("c")
G = get_symbol("G")
hbar = get_symbol("hbar")
k_B = get_symbol("k_B")
e = get_symbol("e")
Lambda = get_symbol("Lambda")
M = get_symbol("M")
rs = get_symbol("rs")

# Generalized coordinate for Lagrangians and its time derivative
# Avoid collision with electric charge 'q' from constants by using 'q_gen'
register_physics_symbol(
    symbol="q_gen",
    description="Generalized coordinate",
    category="parameter",
    test_value=1.0,
    units=None,
    aliases=["q_general"]
)
q_fn = Function("q_gen")
q = q_fn(t)
dq = Derivative(q, t)

# Field placeholders (scalar/vector/tensor-like as generic functions)
psi_fn = Function("psi")  # scalar field
A_fn = Function("A")      # potential-like field
g_fn = Function("g")      # metric-like field
R_fn = Function("R")      # curvature-like
T_fn = Function("T")      # stress-energy-like

psi = psi_fn(t, r)
A = A_fn(t, r)
g = g_fn(t, r)
R = R_fn(t, r)
T = T_fn(t, r)

# Operand tokens (terminals) from constants
OPERANDS = list(DISCOVERY_TOKENS['operands'])
# Unary function tokens from constants
UNARY_FUNCS = list(DISCOVERY_TOKENS['unary'])
# Binary operator tokens from constants
BINARY_OPS = list(DISCOVERY_TOKENS['binary'])

# Full vocabulary in a deterministic order
VOCAB = OPERANDS + UNARY_FUNCS + BINARY_OPS

# Map token string -> concrete implementation
SYMBOL_TABLE = {
    # Operands from registry-backed symbols
    "t": t, "r": r, "theta": theta, "phi": phi, "x": x, "y": y, "z": z,
    "m": m, "c": c, "G": G, "hbar": hbar, "k_B": k_B, "e": e,
    "Lambda": Lambda, "M": M, "rs": rs,
    # Generalized coordinate and its derivative
    "q_gen": q, "dq_gen": dq,
    # Fields
    "psi": psi, "A": A, "g": g, "R": R, "T": T,
    # Force-free specific
    "rho": get_symbol("rho"),
    "1": 1,
    "0": 0,

    # Unary
    "SIN": ("UNARY", sin),
    "COS": ("UNARY", cos),
    "SQRT": ("UNARY", sqrt),
    "EXP": ("UNARY", exp),
    "DT": ("UNARY", lambda a: Derivative(a, t)),
    "LOG": ("UNARY", lambda a: sympy.log(a)),

    # Binary
    "ADD": ("BINARY", lambda a, b: a + b),
    "SUB": ("BINARY", lambda a, b: a - b),
    "MUL": ("BINARY", lambda a, b: a * b),
    "DIV": ("BINARY", lambda a, b: a / b),
    "POW": ("BINARY", lambda a, b: a ** b),
    "GEOM_SUM": ("BINARY", lambda a, b: sqrt((a - 1)**2 + b**2) + sqrt((a + 1)**2 + b**2)),
}


def is_operand(token: str) -> bool:
    return token in OPERANDS


def is_unary(token: str) -> bool:
    return token in UNARY_FUNCS


def is_binary(token: str) -> bool:
    return token in BINARY_OPS


__all__ = [
    # Core symbol objects (registry-backed)
    "t", "r", "theta", "phi", "x", "y", "z",
    "m", "c", "G", "hbar", "k_B", "Lambda", "M", "rs",
    # Generalized coordinate and fields
    "q", "dq", "psi", "A", "g", "R", "T",
    # DSL interface
    "VOCAB", "SYMBOL_TABLE", "OPERANDS", "UNARY_FUNCS", "BINARY_OPS",
    "is_operand", "is_unary", "is_binary",
]


