#!/usr/bin/env python3
"""Stress test for quantum path computation"""

import torch
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
from physics_agent.geodesic_integrator_stable import create_geodesic_solver
from physics_agent.quantum_path_integrator import QuantumPathIntegrator
import torch

# Create theory
theory = QuantumCorrected(alpha=1e-5)

# Create solver
M = 1.989e30
solver = create_geodesic_solver(theory, M_phys=M)

# Near horizon start/end for stress test - with radial motion
start = (0.0, 3.0, torch.pi/2, 0.0)
end = (10.0, 10.0, torch.pi/2, 0.5)

qi = solver.quantum_integrator

try:
    amplitude = qi.compute_amplitude_wkb(start, end, M=M)
    print(f"✓ SUCCESS: Computed amplitude near horizon without crash: {abs(amplitude)}")
    
    # Check if path has curvature
    path = qi._compute_geodesic_path(start, end, num_points=50, M=M)
    r_values = [p[1] for p in path]
    r_std = torch.std(torch.tensor(r_values))
    if r_std > 1e-6:
        print(f"✓ SUCCESS: Path shows curvature (r_std = {r_std:.6f})")
    else:
        print(f"✗ FAIL: Path is straight (r_std = {r_std:.6f})")
except Exception as e:
    print(f"✗ FAIL: Error computing near horizon: {e}") 