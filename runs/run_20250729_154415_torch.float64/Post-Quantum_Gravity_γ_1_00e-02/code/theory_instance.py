#!/usr/bin/env python3
"""Exact theory instance used in this run"""

# This file shows how the theory was instantiated for this run

from physics_agent.theories.post_quantum_gravity.theory import PostQuantumGravityTheory

# Instantiation with exact parameters
theory = PostQuantumGravityTheory(gamma_pqg=0.01, enable_quantum=True)

# Theory name: Post-Quantum Gravity (γ=1.00e-02)
# Category: quantum
