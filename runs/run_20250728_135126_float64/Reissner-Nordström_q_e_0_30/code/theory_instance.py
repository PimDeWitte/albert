#!/usr/bin/env python3
"""Exact theory instance used in this run"""

# This file shows how the theory was instantiated for this run

from theory_baselines_reissner_nordstrom import ReissnerNordstrom

# Instantiation with exact parameters
theory = ReissnerNordstrom(M=M, q_e=0.3, G=G, c=c)

# Theory name: Reissner-Nordström (q_e=0.30)
# Category: base
