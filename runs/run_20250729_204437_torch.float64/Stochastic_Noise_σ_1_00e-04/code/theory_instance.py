#!/usr/bin/env python3
"""Exact theory instance used in this run"""

# This file shows how the theory was instantiated for this run

from physics_agent.theories.stochastic_noise.theory import StochasticNoise

# Instantiation with exact parameters
theory = StochasticNoise(sigma=0.0001)

# Theory name: Stochastic Noise (σ=1.00e-04)
# Category: quantum
