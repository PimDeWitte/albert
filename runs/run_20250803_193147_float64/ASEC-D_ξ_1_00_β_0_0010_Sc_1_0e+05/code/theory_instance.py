#!/usr/bin/env python3
"""Exact theory instance used in this run"""

# This file shows how the theory was instantiated for this run

from theory_asec_theory import ASEC_Decoherence

# Instantiation with exact parameters
theory = ASEC_Decoherence(xi=1.0, beta=0.001, S_crit=100000.0)

# Theory name: ASEC-D (ξ=1.00, β=0.0010, Sc=1.0e+05)
# Category: quantum
