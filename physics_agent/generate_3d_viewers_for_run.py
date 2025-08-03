#!/usr/bin/env python3
"""
Generate 3D WebGL viewers for all theories in a run directory.
"""

import os
import sys
import json
import torch
from typing import Dict, Any

sys.path.insert(0, '.')

from physics_agent.ui.multi_particle_trajectory_viewer_generator import (
    generate_multi_particle_trajectory_viewer
)
from physics_agent.ui.extract_trajectory_data_3d import extract_trajectory_data_3d

def load_trajectory_from_cache(cache_path: str) -> torch.Tensor:
    """Load trajectory from cache file."""
    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=True)
        if isinstance(data, dict) and 'trajectory' in data:
            return data['trajectory']
        elif isinstance(data, torch.Tensor):
            return data
    return None

def generate_3d_viewers_for_run(run_dir: str):
    """Generate 3D viewers for all theories in the run directory."""
    
    # First try to use particle trajectories from the run directory
    particle_traj_dir = os.path.join(run_dir, 'particle_trajectories')
    use_run_trajectories = os.path.exists(particle_traj_dir) and len(os.listdir(particle_traj_dir)) > 0
    
    if use_run_trajectories:
        # Use trajectories from run directory
        print(f"Using particle trajectories from run directory: {particle_traj_dir}")
        cache_dir = particle_traj_dir
    else:
        # Look for trajectory cache files
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache/trajectories/1.0.0/Primordial_Mini_Black_Hole')
    
    if not os.path.exists(cache_dir):
        print(f"Trajectory directory not found: {cache_dir}")
        return
    
    # Create viewers directory in the run
    viewers_dir = os.path.join(run_dir, 'trajectory_viewers')
    os.makedirs(viewers_dir, exist_ok=True)
    
    if use_run_trajectories:
        # Extract unique theories from filenames
        theories = set()
        for filename in os.listdir(cache_dir):
            if filename.endswith('_trajectory.pt'):
                # Format: TheoryName_particle_trajectory.pt
                parts = filename.replace('_trajectory.pt', '').rsplit('_', 1)
                if len(parts) == 2:
                    theory_name = parts[0]
                    theories.add(theory_name)
        theories = sorted(list(theories))
    else:
        # List of theories and particles for cache files
        theories = [
            'Schwarzschild', 'Newtonian_Limit', 'Kerr_a_0_00', 'Kerr-Newman_a_0_00_Q_0_00',
            'Yukawa', 'Einstein_Teleparallel', 'Spinor_Conformal', 'Quantum_Corrected',
            'String_Theory', 'Asymptotic_Safety', 'Loop_Quantum_Gravity',
            'Non-Commutative_Geometry', 'Twistor_Theory', 'Aalto_Gauge_Gravity',
            'Causal_Dynamical_Triangulations'
        ]
    
    particles = ['electron', 'neutrino', 'photon', 'proton']
    
    for theory_clean in theories:
        print(f"\nProcessing {theory_clean}...")
        
        # Prepare particle data
        particle_data = {}
        
        for particle in particles:
            # Look for cache files matching this theory and particle
            found_trajectory = False
            
            if use_run_trajectories:
                # Simple format: TheoryName_particle_trajectory.pt
                expected_filename = f"{theory_clean}_{particle}_trajectory.pt"
                cache_path = os.path.join(cache_dir, expected_filename)
                
                if os.path.exists(cache_path):
                    trajectory = load_trajectory_from_cache(cache_path)
                    if trajectory is not None:
                        print(f"  Found trajectory for {particle}: {expected_filename}")
                        particle_data[particle] = {
                            'theory': extract_trajectory_data_3d(trajectory)
                        }
                        found_trajectory = True
            else:
                # New cache format includes particle in filename: Theory_particle_hash_steps_N.pt
                for cache_file in os.listdir(cache_dir):
                    # Check if this file matches theory and particle
                    if (theory_clean in cache_file and 
                        f"_{particle}_" in cache_file and 
                        cache_file.endswith('.pt')):
                        cache_path = os.path.join(cache_dir, cache_file)
                        trajectory = load_trajectory_from_cache(cache_path)
                        
                        if trajectory is not None:
                            print(f"  Found trajectory for {particle}: {cache_file}")
                            particle_data[particle] = {
                                'theory': extract_trajectory_data_3d(trajectory)
                            }
                            found_trajectory = True
                            break
            
            if not found_trajectory:
                # Try generic theory cache file
                for cache_file in os.listdir(cache_dir):
                    if theory_clean in cache_file and 'steps_10000' in cache_file:
                        cache_path = os.path.join(cache_dir, cache_file)
                        trajectory = load_trajectory_from_cache(cache_path)
                        
                        if trajectory is not None:
                            print(f"  Using generic trajectory for {particle}: {cache_file}")
                            particle_data[particle] = {
                                'theory': extract_trajectory_data_3d(trajectory)
                            }
                            break
        
        if particle_data:
            # Generate viewer
            theory_display = theory_clean.replace('_', ' ')
            viewer_path = os.path.join(viewers_dir, f'{theory_clean}_multi_particle_viewer.html')
            
            generate_multi_particle_trajectory_viewer(
                theory_name=theory_display,
                particle_data=particle_data,
                black_hole_mass=9.945e13,  # Primordial mini BH
                output_path=viewer_path
            )
            
            print(f"  Generated viewer: {viewer_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', help='Run directory containing trajectory data')
    args = parser.parse_args()
    
    generate_3d_viewers_for_run(args.run_dir)