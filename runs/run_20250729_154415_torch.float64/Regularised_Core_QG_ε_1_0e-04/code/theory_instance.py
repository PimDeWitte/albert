#!/usr/bin/env python3
"""Exact theory instance used in this run"""

# This file shows how the theory was instantiated for this run

from physics_agent.theories.einstein_regularised_core.theory import EinsteinRegularisedCore

# Instantiation with exact parameters
theory = EinsteinRegularisedCore(epsilon=0.0001)

# Theory name: Regularised Core QG (ε=1.0e-04)
# Category: quantum
