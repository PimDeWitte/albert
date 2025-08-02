#!/usr/bin/env python3
"""
Generates trajectory viewer HTML files with embedded data.
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

def generate_trajectory_viewer(
    theory_name: str,
    theory_trajectory: Optional[torch.Tensor],
    kerr_trajectory: Optional[torch.Tensor],
    black_hole_mass: float,
    particle_name: str,
    output_path: str,
    cache_dir: Optional[str] = None
):
    """Generate an HTML viewer for trajectory visualization."""
    
    # Load template
    template_path = os.path.join(os.path.dirname(__file__), 'trajectory_viewer.html')
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Extract trajectory data
    theory_data = extract_trajectory_data(theory_trajectory) if theory_trajectory is not None else None
    kerr_data = extract_trajectory_data(kerr_trajectory) if kerr_trajectory is not None else None
    
    # Calculate losses
    if theory_data and kerr_data and theory_trajectory is not None and kerr_trajectory is not None:
        theory_data['losses'] = calculate_trajectory_loss(theory_trajectory, kerr_trajectory)
    
    # Replace placeholders
    html = template.replace('{THEORY_NAME}', theory_name)
    html = html.replace('{BLACK_HOLE_MASS}', str(black_hole_mass))
    html = html.replace('{PARTICLE_NAME}', particle_name)
    html = html.replace('{TRAJECTORY_JSON}', json.dumps(theory_data) if theory_data else 'null')
    html = html.replace('{KERR_JSON}', json.dumps(kerr_data) if kerr_data else 'null')
    
    # If cache directory is provided, add a link to cached data
    if cache_dir and theory_data:
        # TODO: Add link to cached trajectory file
        pass
    
    # Write output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path

def generate_viewer_from_cache(
    theory_name: str,
    cache_path: str,
    kerr_cache_path: str,
    output_path: str,
    black_hole_mass: float = 1.0,
    particle_name: str = "electron"
) -> str:
    """Generate viewer from cached trajectory files."""
    
    # Load cached trajectories
    theory_traj = None
    kerr_traj = None
    
    if os.path.exists(cache_path):
        theory_traj = torch.load(cache_path, map_location='cpu')
        
    if os.path.exists(kerr_cache_path):
        kerr_traj = torch.load(kerr_cache_path, map_location='cpu')
    
    return generate_trajectory_viewer(
        theory_name=theory_name,
        theory_trajectory=theory_traj,
        kerr_trajectory=kerr_traj,
        black_hole_mass=black_hole_mass,
        particle_name=particle_name,
        output_path=output_path,
        cache_dir=os.path.dirname(cache_path)
    )

# Example usage
if __name__ == "__main__":
    # Test with dummy data
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Create dummy trajectory
    t = torch.linspace(0, 100, 1000)
    r = 10 - 0.001 * t  # Slowly falling in
    theta = torch.ones_like(t) * np.pi/2
    phi = 0.1 * t  # Orbiting
    
    dummy_traj = torch.stack([t, r, theta, phi, torch.zeros_like(t), torch.zeros_like(t), 0.1*torch.ones_like(t)], dim=1)
    
    generate_trajectory_viewer(
        theory_name="Test Theory",
        theory_trajectory=dummy_traj,
        kerr_trajectory=dummy_traj * 1.01,  # Slightly different
        black_hole_mass=1.0,
        particle_name="test particle",
        output_path="physics_agent/ui/test_viewer.html"
    )
    
    print("Generated test_viewer.html")