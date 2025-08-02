"""Extract trajectory data in the format expected by the 3D viewer."""

import torch
from typing import Dict, List

def extract_trajectory_data_3d(traj: torch.Tensor, length_scale: float = 1.485e-12) -> Dict[str, List[float]]:
    """Extract trajectory data for 3D visualization.
    
    Args:
        traj: Trajectory tensor [t, r, theta, phi, ...]
        length_scale: Length scale to convert from SI to geometric units (default: primordial mini BH)
    """
    data = {
        'r': [],
        'theta': [],
        'phi': [],
        'time': []
    }
    
    # Trajectory format: [t, r, theta, phi, ...]
    for i in range(traj.shape[0]):
        if traj.shape[1] >= 4:
            data['time'].append(float(traj[i, 0]))
            # Convert r from SI units to geometric units (M)
            r_geom = float(traj[i, 1]) / length_scale
            data['r'].append(r_geom)
            data['theta'].append(float(traj[i, 2]))
            data['phi'].append(float(traj[i, 3]))
    
    return data