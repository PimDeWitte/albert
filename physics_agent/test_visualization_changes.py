#!/usr/bin/env python3
"""Test script to verify visualization changes"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.defaults.baselines.kerr import Kerr
from physics_agent.theories.defaults.baselines.kerr_newman import KerrNewman
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild

# Create engine
engine = TheoryEngine(device='cpu', dtype=torch.float64, verbose=True)

# Create theories
main_theory = Schwarzschild(kappa=0.0)
kerr_baseline = Kerr(a=0.0)
kn_baseline = KerrNewman(a=0.0, q_e=0.5)

baseline_theories = {
    kerr_baseline.name: kerr_baseline,
    kn_baseline.name: kn_baseline
}

# Run trajectories
print("Running main theory trajectory...")
r0 = 30.0 * engine.rs  # Starting radius
steps = 500
dtau = 0.1

# Run charged particle trajectory
hist, tag, _ = engine.run_trajectory(
    main_theory, r0, steps, dtau,
    particle_name='electron'
)

print(f"Main trajectory tag: {tag}")

# Run baselines
baseline_results = {}
for name, baseline in baseline_theories.items():
    print(f"Running baseline: {name}")
    baseline_hist, _, _ = engine.run_trajectory(
        baseline, r0, steps, dtau,
        particle_name='electron'
    )
    baseline_results[name] = baseline_hist

# Create particle info
particle_info = {
    'particle': engine.particle_loader.get_particle('electron'),
    'tag': tag,
    'particle_properties': {
        'name': 'electron',
        'type': 'massive',
        'mass': 9.11e-31,
        'charge': -1.6e-19,
        'spin': 0.5
    }
}

# Generate visualization
print("Generating visualization...")
output_file = "test_visualization.png"
engine.visualizer.generate_comparison_plot(
    main_theory, hist, baseline_results, baseline_theories,
    output_file, engine.rs,
    validations_dict=None,
    particle_info=particle_info
)

print(f"Visualization saved to: {output_file}")
print("\nKey changes to verify:")
print("1. Kerr and Kerr-Newman should NOT appear in the main legend (right side)")
print("2. They should only appear in the 'Baseline Checkpoints' legend (left side)")
print("3. Title should show solver type with clear 4D/6D designation")
print("4. All solver names should include the word 'Solver'") 