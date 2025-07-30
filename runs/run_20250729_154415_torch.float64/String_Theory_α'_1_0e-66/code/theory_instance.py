#!/usr/bin/env python3
"""Exact theory instance used in this run"""

# This file shows how the theory was instantiated for this run

from physics_agent.theories.string.theory import StringTheory

# Instantiation with exact parameters
theory = StringTheory(alpha_prime=1e-66, enable_quantum=True)

# Theory name: String Theory (α'=1.0e-66)
# Category: quantum
