#!/usr/bin/env python3
"""Create a working 3D viewer with proper data format."""

import os
import sys
import torch
import json

sys.path.insert(0, '.')

from physics_agent.ui.multi_particle_trajectory_viewer_generator import generate_multi_particle_trajectory_viewer
from physics_agent.ui.extract_trajectory_data_3d import extract_trajectory_data_3d

def create_working_viewer():
    """Create a working 3D viewer."""
    
    cache_dir = 'physics_agent/cache/trajectories/1.0.0/Primordial_Mini_Black_Hole'
    
    # Find a Schwarzschild trajectory
    cache_files = os.listdir(cache_dir)
    schwarzschild_file = None
    
    for f in cache_files:
        if 'Schwarzschild' in f and f.endswith('.pt'):
            schwarzschild_file = f
            break
    
    if not schwarzschild_file:
        print("No Schwarzschild trajectory found!")
        return
    
    print(f"Using trajectory file: {schwarzschild_file}")
    
    # Load trajectory
    traj_path = os.path.join(cache_dir, schwarzschild_file)
    data = torch.load(traj_path, weights_only=True)
    
    if isinstance(data, dict) and 'trajectory' in data:
        trajectory = data['trajectory']
    else:
        trajectory = data
    
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"First few points:")
    print(f"  t: {trajectory[:3, 0].tolist()}")
    print(f"  r: {trajectory[:3, 1].tolist()}")
    print(f"  theta: {trajectory[:3, 2].tolist()}")
    print(f"  phi: {trajectory[:3, 3].tolist()}")
    
    # Extract data for all particles
    extracted = extract_trajectory_data_3d(trajectory)
    
    print(f"\nExtracted data:")
    print(f"  r length: {len(extracted['r'])}")
    print(f"  First r values: {extracted['r'][:5]}")
    print(f"  First theta values: {extracted['theta'][:5]}")
    print(f"  First phi values: {extracted['phi'][:5]}")
    
    # Create particle data for all 4 particles
    particle_data = {
        'electron': {'theory': extracted},
        'neutrino': {'theory': extracted},
        'photon': {'theory': extracted},
        'proton': {'theory': extracted}
    }
    
    # Save debug info
    debug_info = {
        'trajectory_file': schwarzschild_file,
        'trajectory_shape': list(trajectory.shape),
        'data_lengths': {
            'r': len(extracted['r']),
            'theta': len(extracted['theta']),
            'phi': len(extracted['phi'])
        },
        'sample_data': {
            'r': extracted['r'][:5],
            'theta': extracted['theta'][:5],
            'phi': extracted['phi'][:5]
        }
    }
    
    with open('3d_viewer_debug_info.json', 'w') as f:
        json.dump(debug_info, f, indent=2)
    
    print("\nSaved debug info to 3d_viewer_debug_info.json")
    
    # Generate viewer
    output_path = '3d_viewer_working.html'
    generate_multi_particle_trajectory_viewer(
        'Schwarzschild Working Test',
        particle_data,
        9.945e13,
        output_path
    )
    
    print(f"\nGenerated: {output_path}")
    print("\nNOTE: All particles will follow the same trajectory in this test.")
    print("In the real viewer, each particle has its own unique trajectory.")

if __name__ == "__main__":
    create_working_viewer()