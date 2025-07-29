#!/usr/bin/env python3
"""Debug quantum integration issue"""

import sys
sys.path.append('.')
import torch
import numpy as np
import importlib.util

from physics_agent.unified_trajectory_calculator import UnifiedTrajectoryCalculator
from physics_agent.constants import c, G, SOLAR_MASS

# Load theory
theory_path = "runs/run_20250728_225529_float64/Regularised_Core_QG_ε_1_0e-04/code/theory_source.py"
spec = importlib.util.spec_from_file_location("theory_module", theory_path)
theory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(theory_module)
theory = theory_module.EinsteinRegularisedCore()

print(f"Testing: {theory.name}")
print(f"Category: {theory.category}")

# Create calculator
calc = UnifiedTrajectoryCalculator(
    theory=theory,
    M=SOLAR_MASS,
    c=c,
    G=G,
    enable_classical=True,
    enable_quantum=True
)

# Initial conditions (in geometric units)
initial_conditions = {
    'r': 12.0,  # 12M
    't': 0.0,
    'phi': 0.0,
    'E': 0.95,
    'Lz': 3.9,
    'u_t': 1.14,
    'u_r': 0.0,
    'u_phi': 0.0278,
    'particle_name': 'electron',
    'particle_mass': 9.109e-31,
    'particle_charge': -1.602e-19,
    'particle_spin': 0.5
}

# Test direct trajectory computation
print("\nComputing trajectory...")
result = calc.compute_classical_trajectory(
    initial_conditions,
    time_steps=10,
    step_size=0.01
)

print(f"\nResult keys: {list(result.keys())}")
if 'trajectory' in result:
    traj = result['trajectory']
    print(f"Trajectory shape: {traj.shape}")
    print(f"Trajectory points: {len(traj)}")
    print("\nFirst 5 points:")
    for i in range(min(5, len(traj))):
        r_m = traj[i, 1]  # Radius in meters
        r_geom = r_m / (G * SOLAR_MASS / c**2)  # Convert to geometric units
        print(f"  Step {i}: t={traj[i,0]:.6f}s, r={r_geom:.6f}M, phi={traj[i,2]:.6f}")
    
    # Check motion
    r_all = traj[:, 1] / (G * SOLAR_MASS / c**2)
    print(f"\nRadial evolution:")
    print(f"  Initial: {r_all[0]:.6f}M")
    print(f"  Final: {r_all[-1]:.6f}M")
    print(f"  Change: {abs(r_all[-1] - r_all[0]):.6f}M")
    print(f"  Std dev: {np.std(r_all):.6f}M")
else:
    print(f"ERROR: {result.get('error', 'No trajectory in result')}")

# Check the solver
print(f"\nSolver type: {result.get('solver_type', 'Unknown')}")

# Debug the solver creation
print(f"\nClassical solver: {type(calc.classical_solver).__name__}")
print(f"Has quantum solver: {hasattr(calc, 'quantum_solver')}")

# Check if it's creating a quantum solver
from physics_agent.geodesic_integrator_stable import is_quantum_theory
print(f"Is quantum theory: {is_quantum_theory(theory)}")

# Try creating the solver directly
if is_quantum_theory(theory):
    from physics_agent.geodesic_integrator_stable import GeodesicRK4Solver, QuantumGeodesicSolver
    
    # Create classical solver
    classical = GeodesicRK4Solver(theory)
    
    # Create quantum solver
    quantum = QuantumGeodesicSolver(
        theory=theory,
        classical_solver=classical,
        enable_qed_corrections=True,
        method='wkb',
        M_phys=SOLAR_MASS,
        c=c,
        G=G
    )
    
    print(f"\nDirect solver test:")
    y0 = torch.tensor([0.0, 12.0, 0.0, 0.0], dtype=torch.float64)
    h = torch.tensor(0.01, dtype=torch.float64)
    
    y1 = quantum.rk4_step(y0, h, particle_mass=9.109e-31)
    print(f"  y0: {y0}")
    print(f"  y1: {y1}")
    print(f"  Motion: {y1 is not None and not torch.allclose(y0, y1)}") 