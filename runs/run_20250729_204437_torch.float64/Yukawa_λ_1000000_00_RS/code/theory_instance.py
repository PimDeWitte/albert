#!/usr/bin/env python3
"""Exact theory instance used in this run"""

# This file shows how the theory was instantiated for this run

from physics_agent.theories.yukawa.theory import Yukawa

# Instantiation with exact parameters
theory = Yukawa(lambda_rs=1000000.0)

# Theory name: Yukawa (λ=1000000.00 RS)
# Category: quantum
