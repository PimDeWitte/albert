#!/usr/bin/env python3
"""
Generates unified multi-particle trajectory viewer for an entire run with all theories.
"""

import json
import os
import torch
from typing import Dict, Optional, List, Tuple
import numpy as np

def extract_trajectory_data(trajectory_tensor: torch.Tensor, black_hole_mass: float = 9.945e13) -> Dict:
    """Extract trajectory data from tensor for visualization.
    
    Args:
        trajectory_tensor: Trajectory data tensor
        black_hole_mass: Black hole mass in kg for unit conversion
    """
    if trajectory_tensor is None:
        return None
        
    # Convert to numpy for easier handling
    traj = trajectory_tensor.cpu().numpy() if torch.is_tensor(trajectory_tensor) else trajectory_tensor
    
    # Calculate conversion factor from meters to geometric units (M)
    G = 6.67430e-11
    c = 299792458
    M_physical = G * black_hole_mass / (c * c)  # Schwarzschild radius / 2 in meters
    
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
            r_meters = float(traj[i, 1])
            r_geometric = r_meters / M_physical  # Convert to units of M
            theta = float(traj[i, 2]) 
            phi = float(traj[i, 3])
            data['positions'].append([r_geometric, phi])  # Use r in geometric units
            data['times'].append(float(traj[i, 0]))
            
        # Extract velocities if available
        if traj.shape[1] >= 7:
            r_dot = float(traj[i, 4])
            phi_dot = float(traj[i, 6])
            data['velocities'].append([r_dot, phi_dot])
    
    return data

def load_all_theories_data(run_dir: str) -> Dict[str, Dict]:
    """Load all theories and their particle trajectories from a run directory."""
    run_data = {}
    
    # Common particle names
    particle_names = ['electron', 'neutrino', 'photon', 'proton']
    
    # Check for particle_trajectories directory
    particle_traj_dir = os.path.join(run_dir, 'particle_trajectories')
    
    if os.path.exists(particle_traj_dir):
        # Extract theory names from trajectory files
        theory_names = set()
        for filename in os.listdir(particle_traj_dir):
            if filename.endswith('_trajectory.pt'):
                # Extract theory name from filename
                # Format: TheoryName_particle_trajectory.pt
                parts = filename.replace('_trajectory.pt', '').split('_')
                if len(parts) >= 2 and parts[-1] in particle_names:
                    theory_name = '_'.join(parts[:-1])
                    theory_names.add(theory_name)
        
        # Load data for each theory
        for theory_name in theory_names:
            theory_data = {
                'name': theory_name,
                'category': categorize_theory(theory_name),
                'particles': {}
            }
            
            # Load each particle for this theory
            for particle_name in particle_names:
                filename = f"{theory_name}_{particle_name}_trajectory.pt"
                filepath = os.path.join(particle_traj_dir, filename)
                
                if os.path.exists(filepath):
                    try:
                        data = torch.load(filepath, map_location='cpu')
                        if isinstance(data, dict) and 'trajectory' in data:
                            trajectory = data['trajectory']
                        else:
                            trajectory = data
                        
                        traj_data = extract_trajectory_data(trajectory, black_hole_mass=9.945e13)
                        if traj_data:
                            theory_data['particles'][particle_name] = traj_data
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
            
            if theory_data['particles']:
                # Restore original theory name (with spaces and special chars)
                display_name = restore_theory_name(theory_name)
                run_data[display_name] = theory_data
    
    # Also check for old directory structure
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.') and item != 'particle_trajectories':
            # This might be a theory directory
            theory_name = item.replace('_', ' ')
            
            if theory_name not in run_data:
                theory_data = {
                    'name': theory_name,
                    'category': categorize_theory(theory_name),
                    'particles': {}
                }
                
                # Check for particles subdirectory
                particles_dir = os.path.join(item_path, 'particles')
                if os.path.exists(particles_dir):
                    for particle_name in particle_names:
                        particle_file = os.path.join(particles_dir, f"{particle_name}_trajectory.pt")
                        if os.path.exists(particle_file):
                            try:
                                traj = torch.load(particle_file, map_location='cpu')
                                traj_data = extract_trajectory_data(traj)
                                if traj_data:
                                    theory_data['particles'][particle_name] = traj_data
                            except Exception as e:
                                print(f"Warning: Could not load {particle_name} trajectory: {e}")
                
                if theory_data['particles']:
                    run_data[theory_name] = theory_data
    
    return run_data

def categorize_theory(theory_name: str) -> str:
    """Categorize theory based on its name."""
    name_lower = theory_name.lower()
    
    if any(term in name_lower for term in ['kerr', 'schwarzschild', 'baseline']):
        return 'baseline'
    elif any(term in name_lower for term in ['quantum', 'qg', 'string', 'loop']):
        return 'quantum'
    else:
        return 'classical'

def restore_theory_name(safe_name: str) -> str:
    """Restore original theory name from safe filename version."""
    # Common replacements
    replacements = {
        'Regularised_Core_QG': 'Regularised Core QG',
        'Asymptotic_Safety': 'Asymptotic Safety',
        'String_Theory': 'String Theory',
        'Quantum_Corrected': 'Quantum Corrected',
        'Phase_Transition_QG': 'Phase Transition QG',
        'Aalto_Gauge_Gravity': 'Aalto Gauge Gravity',
        'Variable_G': 'Variable G',
        'Log_Corrected_QG': 'Log-Corrected QG',
        'Post_Quantum_Gravity': 'Post-Quantum Gravity',
        'Emergent_Gravity': 'Emergent Gravity',
        'Kerr_Newman': 'Kerr-Newman',
        'baseline_Kerr': 'Kerr',
        'baseline_Schwarzschild': 'Schwarzschild'
    }
    
    # Check for exact matches first
    for safe, original in replacements.items():
        if safe in safe_name:
            safe_name = safe_name.replace(safe, original)
    
    # Handle parameters in parentheses
    if '_a_' in safe_name or '_q_e_' in safe_name:
        # This is likely a parameterized theory
        parts = safe_name.split('_')
        theory_parts = []
        param_parts = []
        in_params = False
        
        for part in parts:
            if part in ['a', 'q', 'e'] or any(c.replace('.', '').replace('-', '').isdigit() for c in part):
                in_params = True
            
            if in_params:
                param_parts.append(part)
            else:
                theory_parts.append(part)
        
        base_name = ' '.join(theory_parts)
        if param_parts:
            # Reconstruct parameters
            param_str = ' '.join(param_parts)
            param_str = param_str.replace('_', '=').replace(' a ', ' (a').replace(' q e ', ', q_e')
            if '(' in param_str and ')' not in param_str:
                param_str += ')'
            base_name = f"{base_name} {param_str}"
        
        return base_name
    
    # General underscore replacement
    return safe_name.replace('_', ' ')

def generate_unified_multi_particle_viewer(
    run_dir: str,
    output_path: str,
    black_hole_mass: float = 9.945e13  # Primordial mini BH in kg
) -> str:
    """Generate unified HTML viewer for all theories in a run."""
    
    # Load all theories data
    run_data = load_all_theories_data(run_dir)
    
    if not run_data:
        print(f"Warning: No trajectory data found in {run_dir}")
        return None
    
    print(f"Loaded data for {len(run_data)} theories")
    for theory_name, data in run_data.items():
        print(f"  - {theory_name}: {len(data['particles'])} particles")
    
    # Load template
    template_path = os.path.join(os.path.dirname(__file__), 'unified_multi_particle_viewer_advanced.html')
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return None
    
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders
    html = template.replace('{RUN_DATA_JSON}', json.dumps(run_data))
    html = html.replace('{BLACK_HOLE_MASS}', str(black_hole_mass))
    
    # Write output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path

def update_comprehensive_report_generator():
    """Update the comprehensive report generator to use unified viewer."""
    report_gen_path = "physics_agent/comprehensive_test_report_generator.py"
    
    # Read the current file
    with open(report_gen_path, 'r') as f:
        content = f.read()
    
    # Replace the trajectory viewer generation section
    new_section = '''    def _generate_trajectory_viewers(self, results: List[Dict[str, Any]], output_dir: str):
        """Generate unified trajectory viewer for the entire run."""
        viewers_dir = os.path.join(output_dir, 'trajectory_viewers')
        os.makedirs(viewers_dir, exist_ok=True)
        
        try:
            from physics_agent.ui.renderer import (
                generate_unified_multi_particle_viewer
            )
        except ImportError:
            print("Warning: Could not import unified multi-particle viewer generator")
            return
        
        # Generate unified viewer for all theories
        try:
            unified_viewer_path = os.path.join(viewers_dir, 'unified_multi_particle_viewer.html')
            generate_unified_multi_particle_viewer(
                run_dir=output_dir,
                output_path=unified_viewer_path,
                black_hole_mass=9.945e13  # Primordial mini BH in kg
            )
            print(f"Generated unified multi-particle viewer: {unified_viewer_path}")
        except Exception as e:
            print(f"Error generating unified viewer: {e}")
            
        # Also generate individual viewers for backward compatibility
        try:
            from physics_agent.ui.multi_particle_trajectory_viewer_generator import (
                generate_multi_particle_viewer_from_run
            )
            
            for result in results:
                theory_name = result['theory']
                clean_name = theory_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                
                try:
                    viewer_path = os.path.join(viewers_dir, f'{clean_name}_multi_particle_viewer.html')
                    generate_multi_particle_viewer_from_run(
                        theory_name=theory_name,
                        run_dir=output_dir,
                        output_path=viewer_path,
                        black_hole_mass=9.945e13
                    )
                except Exception as e:
                    print(f"Warning: Could not generate viewer for {theory_name}: {e}")
        except:
            pass'''
    
    return new_section

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python renderer.py <run_dir> [output_path]")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "unified_trajectory_viewer.html"
    
    result = generate_unified_multi_particle_viewer(
        run_dir=run_dir,
        output_path=output_path
    )
    
    if result:
        print(f"Generated unified viewer: {result}")
    else:
        print("Failed to generate unified viewer")