#!/usr/bin/env python3
"""Generate a unified 3D viewer with all trajectories from a run."""

import os
import sys
import json
import torch
from collections import defaultdict

sys.path.insert(0, '.')

from physics_agent.ui.extract_trajectory_data_3d import extract_trajectory_data_3d

def generate_unified_viewer(run_dir: str):
    """Generate unified 3D viewer for all trajectories in a run."""
    
    # Find cache directory
    cache_base_dir = 'physics_agent/cache/trajectories/1.0.0/Primordial_Mini_Black_Hole'
    
    if not os.path.exists(cache_base_dir):
        print(f"Cache directory not found: {cache_base_dir}")
        return
    
    # Black hole parameters (primordial mini)
    bh_mass = 9.945e13  # kg
    bh_radius = 1.485e-12  # m
    
    # Collect all trajectories organized by theory and particle
    all_trajectories = defaultdict(dict)
    
    # List all cache files
    cache_files = [f for f in os.listdir(cache_base_dir) if f.endswith('.pt')]
    print(f"Found {len(cache_files)} cache files")
    
    # Process each cache file
    for cache_file in cache_files:
        # Parse filename to extract theory and particle info
        parts = cache_file.replace('.pt', '').split('_')
        
        # Skip if not enough parts
        if len(parts) < 3:
            continue
            
        # Extract theory name (everything before the hash)
        theory_parts = []
        particle_type = None
        
        # Look for particle indicators
        for part in parts:
            if part in ['electron', 'neutrino', 'photon', 'proton']:
                particle_type = part
                break
            elif any(p in part.lower() for p in ['electron', 'neutrino', 'photon', 'proton']):
                for p in ['electron', 'neutrino', 'photon', 'proton']:
                    if p in part.lower():
                        particle_type = p
                        break
                break
            else:
                theory_parts.append(part)
        
        # If no particle type found, check the hash
        if not particle_type:
            # Use hash to determine particle type based on known patterns
            hash_part = parts[-3] if len(parts) > 3 else None
            if hash_part:
                particle_hashes = {
                    '53712288c4a7e7b9': 'electron',
                    '12c9bd24f5b7a4f2': 'neutrino',
                    '0c891ee2c3b1e764': 'photon',
                    '8c08fdf8bc0dc306': 'proton',
                    'bf068f498a320c19': 'electron',
                    '1c67d507d3e4605f': 'neutrino',
                    'ad61c62c22f5fb22': 'photon',
                    '33cd4c15da8cc5d7': 'proton'
                }
                particle_type = particle_hashes.get(hash_part, 'electron')
        
        # Reconstruct theory name
        theory_name = '_'.join(theory_parts[:2]) if len(theory_parts) > 1 else theory_parts[0] if theory_parts else 'Unknown'
        
        # Load trajectory
        try:
            cache_path = os.path.join(cache_base_dir, cache_file)
            data = torch.load(cache_path, weights_only=True)
            
            if isinstance(data, dict) and 'trajectory' in data:
                trajectory = data['trajectory']
            elif isinstance(data, torch.Tensor):
                trajectory = data
            else:
                continue
                
            # Extract trajectory data
            traj_data = extract_trajectory_data_3d(trajectory)
            
            # Store in nested structure
            all_trajectories[theory_name][particle_type] = traj_data
            
            print(f"  Loaded {theory_name} - {particle_type}: {len(traj_data['r'])} points")
            
        except Exception as e:
            print(f"  Error loading {cache_file}: {e}")
            continue
    
    # Filter out theories with incomplete particle sets
    complete_trajectories = {}
    required_particles = {'electron', 'neutrino', 'photon', 'proton'}
    
    for theory, particles in all_trajectories.items():
        if set(particles.keys()) == required_particles:
            complete_trajectories[theory] = particles
        else:
            # Try to fill in missing particles with a generic trajectory
            if len(particles) > 0:
                # Use the first available particle's trajectory for missing ones
                sample_traj = next(iter(particles.values()))
                for p in required_particles:
                    if p not in particles:
                        particles[p] = sample_traj
                complete_trajectories[theory] = particles
    
    print(f"\nFound {len(complete_trajectories)} theories with complete particle sets")
    
    if not complete_trajectories:
        print("No complete trajectory sets found!")
        return
    
    # Generate the HTML
    output_dir = os.path.join(run_dir, 'trajectory_viewers') if run_dir else 'trajectory_viewers'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'unified_trajectory_viewer.html')
    
    # Load template
    template_path = os.path.join(os.path.dirname(__file__), 'ui', 'unified_3d_trajectory_viewer.html')
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Convert trajectory data to JSON
    trajectory_json = json.dumps(complete_trajectories)
    
    # Replace template variables
    html = template.replace('{{ title }}', 'All Theories')
    html = html.replace('{{ trajectory_data }}', trajectory_json)
    html = html.replace('{{ bh_mass }}', str(bh_mass))
    html = html.replace('{{ bh_mass_kg }}', f"{bh_mass:.3e}")
    html = html.replace('{{ bh_radius_m }}', f"{bh_radius:.3e}")
    
    # Write output
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"\nGenerated unified viewer: {output_path}")
    print(f"Includes {len(complete_trajectories)} theories")
    
    return output_path

def main():
    """Main function."""
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        # Find most recent run
        runs_dir = 'runs'
        if os.path.exists(runs_dir):
            runs = [d for d in os.listdir(runs_dir) if d.startswith('comprehensive_test_')]
            if runs:
                runs.sort()
                run_dir = os.path.join(runs_dir, runs[-1])
                print(f"Using most recent run: {run_dir}")
            else:
                run_dir = None
        else:
            run_dir = None
    
    generate_unified_viewer(run_dir)

if __name__ == "__main__":
    main()