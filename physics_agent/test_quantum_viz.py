#!/usr/bin/env python3
"""
Test quantum trajectory visualization with PennyLane-based simulator.
Shows quantum corrections in trajectory plots.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from physics_agent.geodesic_integrator_stable import create_geodesic_solver
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.string.theory import StringTheory
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT


def run_quantum_visualization_test():
    """Run a test showing quantum vs classical trajectories"""
    
    print("="*60)
    print("Quantum Trajectory Visualization Test")
    print("="*60)
    
    # Initialize theories
    classical_theory = Schwarzschild()
    quantum_theory = StringTheory()  # A quantum theory
    
    # Set up parameters
    M_sun = SOLAR_MASS
    rs = 2 * GRAVITATIONAL_CONSTANT * M_sun / SPEED_OF_LIGHT**2
    r0 = 10 * rs  # Start at 10 Schwarzschild radii
    
    # Set up initial conditions for circular orbit
    from physics_agent.geodesic_integrator import ConservedQuantityGeodesicSolver
    temp_solver = ConservedQuantityGeodesicSolver(classical_theory, M_phys=torch.tensor(M_sun))
    r0_geom = temp_solver.to_geometric_length(torch.tensor(r0))
    E_geom, L_geom = temp_solver.compute_circular_orbit_params(r0_geom)
    
    # Initial state: [t, r, phi, dr/dtau] for 4D motion
    y0 = torch.tensor([0.0, r0_geom.item(), 0.0, 0.0], dtype=torch.float64)
    
    # Time parameters
    n_steps = 5000
    h = 0.01
    
    print(f"\nInitial conditions:")
    print(f"  Radius: {r0/rs:.1f} Schwarzschild radii")
    print(f"  Energy: {E_geom.item():.6f}")
    print(f"  Angular momentum: {L_geom.item():.6f}")
    
    # Run classical trajectory
    print("\nComputing classical trajectory...")
    classical_solver = create_geodesic_solver(
        classical_theory, 
        M_phys=torch.tensor(M_sun),
        verbose=True
    )
    # Set conserved quantities for classical solver
    classical_solver.E = E_geom.item()
    classical_solver.Lz = L_geom.item()
    
    classical_trajectory = [y0.numpy()]
    state = y0.clone()
    for i in range(n_steps):
        state = classical_solver.rk4_step(state, torch.tensor(h))
        if state is None:
            print(f"Classical integration failed at step {i}")
            break
        classical_trajectory.append(state.detach().numpy() if torch.is_tensor(state) else state.numpy())
        if i % 1000 == 0:
            print(f"  Step {i}/{n_steps}")
    
    classical_trajectory = np.array(classical_trajectory)
    print(f"Classical trajectory complete: {len(classical_trajectory)} points")
    
    # Run quantum trajectory with PennyLane
    print("\nComputing quantum trajectory with PennyLane...")
    quantum_solver = create_geodesic_solver(
        quantum_theory,
        M_phys=torch.tensor(M_sun), 
        use_pennylane_quantum=True,
        num_qubits=4,
        verbose=True
    )
    
    # Convert to 6D state for quantum solver: [t, r, phi, u^t, u^r, u^phi]
    y0_6d = torch.tensor([0.0, r0_geom.item(), 0.0, E_geom.item(), 0.0, L_geom.item()/r0_geom.item()], 
                         dtype=torch.float64)
    
    quantum_trajectory = [y0_6d.numpy()]
    state = y0_6d.clone()
    for i in range(n_steps):
        state = quantum_solver.rk4_step(state, torch.tensor(h))
        if state is None:
            print(f"Quantum integration failed at step {i}")
            break
        quantum_trajectory.append(state.detach().numpy())
        if i % 1000 == 0:
            print(f"  Step {i}/{n_steps}")
    
    quantum_trajectory = np.array(quantum_trajectory)
    print(f"Quantum trajectory complete: {len(quantum_trajectory)} points")
    
    # Create visualization
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(18, 12), facecolor='black')
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d', facecolor='black')
    
    # Convert to Cartesian coordinates
    def to_cartesian(traj):
        t = traj[:, 0]
        r = traj[:, 1] 
        phi = traj[:, 2]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y, t
    
    x_c, y_c, t_c = to_cartesian(classical_trajectory)
    x_q, y_q, t_q = to_cartesian(quantum_trajectory)
    
    # Plot trajectories
    ax1.plot(x_c, y_c, t_c, 'cyan', linewidth=2, alpha=0.8, label='Classical')
    ax1.plot(x_q, y_q, t_q, 'magenta', linewidth=2, alpha=0.8, label='Quantum (PennyLane)')
    
    # Add start/end markers
    ax1.scatter(*[x_c[0], y_c[0], t_c[0]], c='green', s=100, label='Start')
    ax1.scatter(*[x_c[-1], y_c[-1], t_c[-1]], c='red', s=100, label='End')
    
    # Add event horizon
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = 2 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 2 * np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v)) * t_c.max()/2 + t_c.max()/2
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, color='yellow', alpha=0.3)
    
    ax1.set_xlabel('X (geometric)', color='white')
    ax1.set_ylabel('Y (geometric)', color='white')
    ax1.set_zlabel('Time (geometric)', color='white')
    ax1.set_title('3D Trajectory Comparison', color='white', fontsize=14)
    ax1.legend()
    
    # Radial distance comparison
    ax2 = fig.add_subplot(222, facecolor='black')
    ax2.plot(classical_trajectory[:, 0], classical_trajectory[:, 1], 'cyan', 
             linewidth=2, label='Classical')
    ax2.plot(quantum_trajectory[:, 0], quantum_trajectory[:, 1], 'magenta', 
             linewidth=2, label='Quantum')
    ax2.set_xlabel('Time (geometric)', color='white')
    ax2.set_ylabel('Radius (geometric)', color='white')
    ax2.set_title('Radial Evolution', color='white')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Quantum deviation
    ax3 = fig.add_subplot(223, facecolor='black')
    # Interpolate to common time points
    common_t = np.linspace(0, min(t_c[-1], t_q[-1]), 1000)
    r_c_interp = np.interp(common_t, classical_trajectory[:, 0], classical_trajectory[:, 1])
    r_q_interp = np.interp(common_t, quantum_trajectory[:, 0], quantum_trajectory[:, 1])
    
    deviation = (r_q_interp - r_c_interp) / r_c_interp * 100  # Percentage
    ax3.plot(common_t, deviation, 'yellow', linewidth=2)
    ax3.set_xlabel('Time (geometric)', color='white')
    ax3.set_ylabel('Radial Deviation (%)', color='white')
    ax3.set_title('Quantum Correction Magnitude', color='white')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Phase space - handle different dimensions
    ax4 = fig.add_subplot(224, facecolor='black')
    if classical_trajectory.shape[1] == 4:
        # Classical is 4D: [t, r, phi, dr/dtau]
        ax4.plot(classical_trajectory[:, 1], classical_trajectory[:, 3], 'cyan', 
                 linewidth=1, alpha=0.6, label='Classical')
    else:
        # Classical is 6D: [t, r, phi, u^t, u^r, u^phi]
        ax4.plot(classical_trajectory[:, 1], classical_trajectory[:, 4], 'cyan', 
                 linewidth=1, alpha=0.6, label='Classical')
    
    # Quantum is always 6D
    ax4.plot(quantum_trajectory[:, 1], quantum_trajectory[:, 4], 'magenta', 
             linewidth=1, alpha=0.6, label='Quantum')
    ax4.set_xlabel('r (geometric)', color='white')
    ax4.set_ylabel('dr/dτ (geometric)', color='white')
    ax4.set_title('Phase Space', color='white')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output directory
    output_dir = Path('test_viz_output/quantum_pennylane')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'quantum_trajectory_comparison.png'
    plt.savefig(output_file, dpi=300, facecolor='black')
    print(f"\nVisualization saved to: {output_file}")
    
    # Print statistics
    print("\nTrajectory Statistics:")
    print(f"  Maximum radial deviation: {np.abs(deviation).max():.3f}%")
    print(f"  Average radial deviation: {np.abs(deviation).mean():.3f}%")
    print(f"  Final radius difference: {abs(r_q_interp[-1] - r_c_interp[-1]):.6f}")
    
    return output_file


if __name__ == "__main__":
    run_quantum_visualization_test()