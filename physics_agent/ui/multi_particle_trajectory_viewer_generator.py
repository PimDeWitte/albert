#!/usr/bin/env python3
"""
Generates multi-particle trajectory viewer HTML files with all particle trajectories.
"""

import json
import os
import torch
from typing import Dict, Optional, List, Tuple
import numpy as np

def extract_trajectory_data(trajectory_tensor: torch.Tensor) -> Dict:
    """Extract trajectory data from tensor for visualization."""
    if trajectory_tensor is None:
        return None
        
    # Convert to numpy for easier handling
    traj = trajectory_tensor.cpu().numpy() if torch.is_tensor(trajectory_tensor) else trajectory_tensor
    
    # Trajectory format: [steps, features]
    # features typically: [t, r, theta, phi, r_dot, theta_dot, phi_dot, ...]
    
    data = {
        'positions': [],
        'velocities': [],
        'times': [],
        'losses': []
    }
    
    for i in range(traj.shape[0]):
        # Extract position (r, theta, phi)
        if traj.shape[1] >= 4:
            r = float(traj[i, 1])
            theta = float(traj[i, 2]) 
            phi = float(traj[i, 3])
            data['positions'].append([r, phi])  # Use r, phi for 2D projection
            data['times'].append(float(traj[i, 0]))
            
        # Extract velocities if available
        if traj.shape[1] >= 7:
            r_dot = float(traj[i, 4])
            phi_dot = float(traj[i, 6])
            data['velocities'].append([r_dot, phi_dot])
    
    return data

def calculate_trajectory_loss(theory_traj: torch.Tensor, kerr_traj: torch.Tensor) -> List[float]:
    """Calculate per-step loss between trajectories."""
    if theory_traj is None or kerr_traj is None:
        return []
        
    losses = []
    min_steps = min(theory_traj.shape[0], kerr_traj.shape[0])
    
    for i in range(min_steps):
        # Calculate position difference
        dr = theory_traj[i, 1] - kerr_traj[i, 1]
        dtheta = theory_traj[i, 2] - kerr_traj[i, 2]
        dphi = theory_traj[i, 3] - kerr_traj[i, 3]
        
        # Simple Euclidean loss in spherical coordinates
        loss = float(dr**2 + (dtheta * theory_traj[i, 1])**2 + 
                    (dphi * theory_traj[i, 1] * torch.sin(theory_traj[i, 2]))**2)
        losses.append(np.sqrt(loss))
        
    return losses

def load_particle_trajectories(run_dir: str, theory_name: str) -> Dict[str, Dict]:
    """Load all particle trajectories for a theory from the run directory."""
    particle_data = {}
    
    # Common particle names
    particle_names = ['electron', 'neutrino', 'photon', 'proton']
    
    # Try to find theory directory
    theory_dir = None
    for subdir in os.listdir(run_dir):
        if theory_name.replace(' ', '_') in subdir or theory_name in subdir:
            theory_dir = os.path.join(run_dir, subdir)
            break
    
    if not theory_dir or not os.path.exists(theory_dir):
        print(f"Warning: Could not find theory directory for {theory_name}")
        return particle_data
    
    # Look for particle trajectories
    particles_dir = os.path.join(theory_dir, 'particles')
    if os.path.exists(particles_dir):
        # Load from particles subdirectory
        for particle_name in particle_names:
            particle_file = os.path.join(particles_dir, f"{particle_name}_trajectory.pt")
            if os.path.exists(particle_file):
                try:
                    traj = torch.load(particle_file, map_location='cpu')
                    particle_data[particle_name] = {'theory': extract_trajectory_data(traj)}
                except Exception as e:
                    print(f"Warning: Could not load {particle_name} trajectory: {e}")
    else:
        # Try legacy format
        for particle_name in particle_names:
            particle_file = os.path.join(theory_dir, f"trajectory_{particle_name}.pt")
            if os.path.exists(particle_file):
                try:
                    traj = torch.load(particle_file, map_location='cpu')
                    particle_data[particle_name] = {'theory': extract_trajectory_data(traj)}
                except Exception as e:
                    print(f"Warning: Could not load {particle_name} trajectory: {e}")
    
    # Load Kerr baseline trajectories
    baseline_dirs = ['baseline_Kerr_a=0.00', 'Kerr']
    kerr_dir = None
    
    for baseline_name in baseline_dirs:
        potential_dir = os.path.join(run_dir, baseline_name)
        if os.path.exists(potential_dir):
            kerr_dir = potential_dir
            break
    
    if kerr_dir:
        # Load Kerr baselines for each particle
        kerr_particles_dir = os.path.join(kerr_dir, 'particles')
        if os.path.exists(kerr_particles_dir):
            for particle_name in particle_names:
                if particle_name in particle_data:
                    kerr_file = os.path.join(kerr_particles_dir, f"{particle_name}_trajectory.pt")
                    if os.path.exists(kerr_file):
                        try:
                            kerr_traj = torch.load(kerr_file, map_location='cpu')
                            particle_data[particle_name]['kerr'] = extract_trajectory_data(kerr_traj)
                            
                            # Calculate losses
                            if 'theory' in particle_data[particle_name]:
                                theory_tensor = torch.load(
                                    os.path.join(particles_dir, f"{particle_name}_trajectory.pt"),
                                    map_location='cpu'
                                )
                                particle_data[particle_name]['theory']['losses'] = calculate_trajectory_loss(
                                    theory_tensor, kerr_traj
                                )
                        except Exception as e:
                            print(f"Warning: Could not load Kerr baseline for {particle_name}: {e}")
        else:
            # Try legacy format
            for particle_name in particle_names:
                if particle_name in particle_data:
                    kerr_file = os.path.join(kerr_dir, f"trajectory_{particle_name}.pt")
                    if os.path.exists(kerr_file):
                        try:
                            kerr_traj = torch.load(kerr_file, map_location='cpu')
                            particle_data[particle_name]['kerr'] = extract_trajectory_data(kerr_traj)
                        except Exception as e:
                            print(f"Warning: Could not load Kerr baseline for {particle_name}: {e}")
    
    return particle_data

def generate_multi_particle_trajectory_viewer(
    theory_name: str,
    particle_data: Dict[str, Dict],
    black_hole_mass: float,
    output_path: str
):
    """Generate an HTML viewer for multi-particle trajectory visualization."""
    
    # Load template - use fixed 3D version
    template_path = os.path.join(os.path.dirname(__file__), 'multi_particle_trajectory_viewer_3d_fixed.html')
    if not os.path.exists(template_path):
        # Fallback to original 3D version
        template_path = os.path.join(os.path.dirname(__file__), 'multi_particle_trajectory_viewer_3d.html')
        if not os.path.exists(template_path):
            # Fallback to 2D version
            template_path = os.path.join(os.path.dirname(__file__), 'multi_particle_trajectory_viewer.html')
    
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders
    html = template.replace('{THEORY_NAME}', theory_name)
    html = html.replace('{BLACK_HOLE_MASS}', str(black_hole_mass))
    html = html.replace('{PARTICLE_DATA_JSON}', json.dumps(particle_data))
    
    # Write output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path

def generate_multi_particle_viewer_from_run(
    theory_name: str,
    run_dir: str,
    output_path: str,
    black_hole_mass: float = 9.945e13  # Primordial mini BH in kg
) -> str:
    """Generate multi-particle viewer from a run directory."""
    
    # Load all particle trajectories
    particle_data = load_particle_trajectories(run_dir, theory_name)
    
    if not particle_data:
        print(f"Warning: No particle data found for {theory_name}")
        return None
    
    print(f"Loaded trajectories for {len(particle_data)} particles: {', '.join(particle_data.keys())}")
    
    return generate_multi_particle_trajectory_viewer(
        theory_name=theory_name,
        particle_data=particle_data,
        black_hole_mass=black_hole_mass,
        output_path=output_path
    )

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python multi_particle_trajectory_viewer_generator.py <theory_name> <run_dir>")
        sys.exit(1)
    
    theory_name = sys.argv[1]
    run_dir = sys.argv[2]
    
    output_path = f"{theory_name.replace(' ', '_')}_multi_particle_viewer.html"
    
    result = generate_multi_particle_viewer_from_run(
        theory_name=theory_name,
        run_dir=run_dir,
        output_path=output_path
    )
    
    if result:
        print(f"Generated multi-particle viewer: {result}")
    else:
        print("Failed to generate viewer")