#!/usr/bin/env python3
"""
Create particle configuration files for theories that are missing them.
This ensures the multi-particle trajectory viewer can find particle data.
"""

import os
import json
import shutil

# List of theories that need particle configs (from the warnings)
theories_needing_configs = [
    'yukawa',
    'spinor_conformal',
    'quantum_corrected', 
    'string',
    'asymptotic_safety',
    'non_commutative_geometry',
    'causal_dynamical_triangulations'
]

# Base directory for theories
theories_base = 'physics_agent/theories'

# Source particle files
particles_dir = 'physics_agent/particles/defaults'

def create_particle_config_for_theory(theory_name):
    """Create particle configuration directory for a theory."""
    
    # Find the theory directory
    theory_dir = None
    for item in os.listdir(theories_base):
        if item.lower().replace('_', '').replace('-', '') == theory_name.lower().replace('_', ''):
            theory_dir = os.path.join(theories_base, item)
            break
    
    if not theory_dir or not os.path.isdir(theory_dir):
        print(f"  ✗ Could not find directory for {theory_name}")
        return False
    
    # Create particles subdirectory
    particles_subdir = os.path.join(theory_dir, 'particles')
    if os.path.exists(particles_subdir):
        print(f"  ✓ {theory_name}: particles directory already exists")
        return True
    
    os.makedirs(particles_subdir, exist_ok=True)
    
    # Copy default particle configurations
    particle_files = ['electron.json', 'neutrino.json', 'photon.json', 'proton.json']
    
    for particle_file in particle_files:
        src = os.path.join(particles_dir, particle_file)
        if os.path.exists(src):
            dst = os.path.join(particles_subdir, particle_file)
            shutil.copy2(src, dst)
    
    # Create a README
    readme_content = f"""# Particle Configurations for {theory_name.replace('_', ' ').title()}

This directory contains particle configurations for trajectory simulations.

## Files

- `electron.json` - Electron parameters
- `neutrino.json` - Neutrino parameters (massless)
- `photon.json` - Photon parameters (massless)
- `proton.json` - Proton parameters

These are copies of the default particle configurations. You can modify them
if this theory predicts different particle properties or behaviors.
"""
    
    with open(os.path.join(particles_subdir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"  ✓ Created particle configs for {theory_name} in {particles_subdir}")
    return True

def main():
    print("Creating particle configurations for theories...\n")
    
    success_count = 0
    for theory_name in theories_needing_configs:
        if create_particle_config_for_theory(theory_name):
            success_count += 1
    
    print(f"\nSummary: Created configs for {success_count}/{len(theories_needing_configs)} theories")
    
    # Also check if there are any other quantum theories without configs
    print("\nChecking all quantum theories...")
    
    quantum_theories = [
        'loop_quantum_gravity',
        'twistor_theory', 
        'aalto_gauge_gravity',
        'emergent',
        'entropic_gravity',
        'fractal',
        'gauge_gravity',
        'log_corrected',
        'phase_transition',
        'post_quantum_gravity',
        'regularised_core',
        'stochastic_noise',
        'surfaceology',
        'ugm'
    ]
    
    for theory_name in quantum_theories:
        theory_dir = None
        for item in os.listdir(theories_base):
            if item.lower().replace('_', '').replace('-', '') == theory_name.lower().replace('_', ''):
                theory_dir = os.path.join(theories_base, item)
                break
        
        if theory_dir and os.path.isdir(theory_dir):
            particles_subdir = os.path.join(theory_dir, 'particles')
            if not os.path.exists(particles_subdir):
                print(f"  Creating configs for {theory_name}...")
                create_particle_config_for_theory(theory_name)

if __name__ == "__main__":
    main()