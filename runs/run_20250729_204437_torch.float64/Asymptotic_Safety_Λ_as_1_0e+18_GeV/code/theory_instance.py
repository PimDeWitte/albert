#!/usr/bin/env python3
"""Exact theory instance used in this run"""

# This file shows how the theory was instantiated for this run

from physics_agent.theories.asymptotic_safety.theory import AsymptoticSafetyTheory

# Instantiation with exact parameters
theory = AsymptoticSafetyTheory(Lambda_as=1e+18, enable_quantum=True)

# Theory name: Asymptotic Safety (Λ_as=1.0e+18 GeV)
# Category: quantum
