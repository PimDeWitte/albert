#!/usr/bin/env python3
"""Exact theory instance used in this run"""

# This file shows how the theory was instantiated for this run

from physics_agent.theories.defaults.baselines.kerr_newman import KerrNewman

# Instantiation with exact parameters
theory = KerrNewman(M=M, a=0.5, Q=0.3, G=G, c=c)

# Theory name: Kerr-Newman (a=0.50, q_e=0.30)
# Category: classical
