"""
Test script to verify that quantum corrections actually change trajectories.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.theory_engine_core import TheoryEngine

# Create engine
engine = TheoryEngine(device='cpu', dtype=torch.float64)

# Create theories
schw = Schwarzschild()
qc_small = QuantumCorrected(alpha=0.01)  # Small correction
qc_large = QuantumCorrected(alpha=0.1)   # Larger correction for visibility

# Run trajectories
r0 = 6.0 * engine.rs  # Start at 6 Rs
steps = 1000
dt = 0.01 * engine.time_scale

print('Running trajectories...')
print(f'Initial radius: {r0/engine.rs:.1f} Rs')
print(f'Steps: {steps}, dt: {dt/engine.time_scale:.3f} (geometric units)')

# Run all three
hist_s, tag_s, _ = engine.run_trajectory(schw, r0, steps, dt)
hist_q1, tag_q1, _ = engine.run_trajectory(qc_small, r0, steps, dt)  
hist_q2, tag_q2, _ = engine.run_trajectory(qc_large, r0, steps, dt)

if hist_s is not None and hist_q1 is not None and hist_q2 is not None:
    # Extract coordinates
    t_s, r_s, phi_s = hist_s[:, 0], hist_s[:, 1], hist_s[:, 2]
    t_q1, r_q1, phi_q1 = hist_q1[:, 0], hist_q1[:, 1], hist_q1[:, 2]
    t_q2, r_q2, phi_q2 = hist_q2[:, 0], hist_q2[:, 1], hist_q2[:, 2]
    
    # Convert to Cartesian for visualization
    x_s, y_s = r_s * torch.cos(phi_s), r_s * torch.sin(phi_s)
    x_q1, y_q1 = r_q1 * torch.cos(phi_q1), r_q1 * torch.sin(phi_q1)
    x_q2, y_q2 = r_q2 * torch.cos(phi_q2), r_q2 * torch.sin(phi_q2)
    
    # Print differences
    print(f'\nTrajectory comparison after {steps} steps:')
    print(f'Schwarzschild:')
    print(f'  Final r: {r_s[-1]:.3f} Rs')
    print(f'  Total angle: {(phi_s[-1] - phi_s[0]) * 180/np.pi:.1f}°')
    
    print(f'\nQuantum Corrected (α=0.01):')
    print(f'  Final r: {r_q1[-1]:.3f} Rs')
    print(f'  Total angle: {(phi_q1[-1] - phi_q1[0]) * 180/np.pi:.1f}°')
    print(f'  Position difference: Δr={abs(r_q1[-1] - r_s[-1]):.6f} Rs, Δφ={abs(phi_q1[-1] - phi_s[-1]) * 180/np.pi:.3f}°')
    
    print(f'\nQuantum Corrected (α=0.1):')
    print(f'  Final r: {r_q2[-1]:.3f} Rs')
    print(f'  Total angle: {(phi_q2[-1] - phi_q2[0]) * 180/np.pi:.1f}°')
    print(f'  Position difference: Δr={abs(r_q2[-1] - r_s[-1]):.6f} Rs, Δφ={abs(phi_q2[-1] - phi_s[-1]) * 180/np.pi:.3f}°')
    
    # Plot trajectories
    plt.figure(figsize=(10, 10))
    
    # Plot trajectories
    plt.plot(x_s, y_s, 'k-', linewidth=2, label='Schwarzschild', alpha=0.8)
    plt.plot(x_q1, y_q1, 'b--', linewidth=2, label='Quantum (α=0.01)', alpha=0.8)
    plt.plot(x_q2, y_q2, 'r:', linewidth=3, label='Quantum (α=0.1)', alpha=0.8)
    
    # Mark start and end
    plt.plot(x_s[0], y_s[0], 'go', markersize=10, label='Start')
    plt.plot(x_s[-1], y_s[-1], 'ko', markersize=8)
    plt.plot(x_q1[-1], y_q1[-1], 'bo', markersize=8)
    plt.plot(x_q2[-1], y_q2[-1], 'ro', markersize=8)
    
    # Add black hole
    circle = plt.Circle((0, 0), 2.0, color='black', label='Black hole')
    plt.gca().add_patch(circle)
    
    # Add photon sphere
    photon_sphere = plt.Circle((0, 0), 3.0, color='gray', fill=False, 
                              linestyle='--', alpha=0.5, label='Photon sphere (classical)')
    plt.gca().add_patch(photon_sphere)
    
    plt.xlabel('x/Rs')
    plt.ylabel('y/Rs')
    plt.title('Quantum Corrections Change Orbital Trajectories')
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    
    plt.savefig('quantum_trajectory_comparison.png', dpi=150, bbox_inches='tight')
    print('\nPlot saved to quantum_trajectory_comparison.png')
    
    # Compute phase differences over time
    phase_diff_1 = (phi_q1 - phi_s) * 180/np.pi
    phase_diff_2 = (phi_q2 - phi_s) * 180/np.pi
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_s, phase_diff_1, 'b-', label='α=0.01')
    plt.plot(t_s, phase_diff_2, 'r-', label='α=0.1')
    plt.xlabel('Time (geometric units)')
    plt.ylabel('Phase difference from Schwarzschild (degrees)')
    plt.title('Quantum Effects Accumulate Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('quantum_phase_difference.png', dpi=150, bbox_inches='tight')
    print('Phase difference plot saved to quantum_phase_difference.png')
    
else:
    print('Failed to compute one or more trajectories') 