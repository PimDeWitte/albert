#!/usr/bin/env python3
"""Exact theory instance used in this run"""

# This file shows how the theory was instantiated for this run

from physics_agent.theories.kaluza_klein.theory import KaluzaKleinTheory

# Instantiation with exact parameters
theory = KaluzaKleinTheory(R_kk=1e-32, enable_quantum=True)

# Theory name: Kaluza-Klein (R_kk=1.0e-32)
# Category: quantum
