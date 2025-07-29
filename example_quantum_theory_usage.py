#!/usr/bin/env python3
"""
Example of how to properly use quantum theories with constants from constants.py

This demonstrates the correct way to:
1. Import and use physical constants
2. Convert between SI and geometric units
3. Set up quantum theories for testing
"""

import sys
sys.path.append('.')

# Import all constants properly
from physics_agent.constants import (
    # Fundamental constants
    c, G, SOLAR_MASS, ELECTRON_MASS, PROTON_MASS,
    # Schwarzschild radius function
    schwarzschild_radius,
    # Unit conversion functions
    si_to_geometric, geometric_to_si,
    # Test parameters
    STANDARD_ORBITS, STANDARD_INTEGRATION,
    # Numerical thresholds
    NUMERICAL_THRESHOLDS
)

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.base_theory import GravitationalTheory
import torch

class ExampleQuantumTheory(GravitationalTheory):
    """Example quantum gravity theory using proper constants"""
    
    def __init__(self):
        super().__init__()
        self.name = "Example Quantum Theory"
        self.category = "quantum"
        self._is_symmetric = True
        
    def get_metric(self, r, M_param, C_param, G_param, t=None, phi=None):
        """Schwarzschild metric with quantum corrections"""
        # Use provided parameters (already in SI units)
        rs = 2 * G_param * M_param / C_param**2
        
        # Basic Schwarzschild metric
        m = 1 - rs / r
        
        # Add small quantum correction (example)
        # <reason>chain: Quantum effects scale with Planck length / r</reason>
        quantum_correction = 1e-10 * rs / r**2
        m_quantum = m + quantum_correction
        
        g_tt = -m_quantum
        g_rr = 1 / m_quantum
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp

def main():
    print("QUANTUM THEORY EXAMPLE WITH PROPER CONSTANTS")
    print("=" * 60)
    
    # Create theory and engine
    theory = ExampleQuantumTheory()
    engine = TheoryEngine(verbose=True)
    
    # Use standard orbit parameters from constants.py
    orbit = STANDARD_ORBITS['circular_stable']  # r = 6.0 Rs
    
    # Calculate starting radius in SI units
    # For a 10 solar mass black hole (typical test mass)
    test_mass = 10.0 * SOLAR_MASS  # Use SOLAR_MASS constant
    rs_si = schwarzschild_radius(test_mass)  # Use utility function
    r0_si = orbit['r'] * rs_si
    
    # Calculate proper time step in SI units
    # In geometric units: dtau = 0.01 M
    # In SI units: dtau = 0.01 * GM/c³
    dtau_geom = STANDARD_INTEGRATION['dtau']  # 0.01 from constants
    dtau_si = geometric_to_si(dtau_geom, 'time', test_mass)
    
    print(f"\nTest parameters:")
    print(f"  Test mass: {test_mass/SOLAR_MASS:.1f} M☉")
    print(f"  Schwarzschild radius: {rs_si:.1f} m")
    print(f"  Starting radius: {orbit['r']:.1f} Rs = {r0_si:.1f} m")
    print(f"  Time step: {dtau_geom:.3f} M = {dtau_si:.2e} s")
    
    # Run trajectory with proper parameters
    n_steps = 100
    
    print(f"\nRunning trajectory for {n_steps} steps...")
    hist, tag, kicks = engine.run_trajectory(
        theory, 
        r0_si,      # Starting radius in SI units
        n_steps, 
        dtau_si,    # Time step in SI units
        use_quantum_trajectories=True,
        particle_name='electron',
        no_cache=True
    )
    
    if hist is not None:
        print(f"\nResults:")
        print(f"  Tag: {tag}")
        print(f"  Trajectory shape: {hist.shape}")
        print(f"  Integration successful: {'quantum' in tag}")
        
        # Extract radial coordinates and convert to geometric units
        r_values = hist[:, 1].numpy()
        r_geom = si_to_geometric(r_values, 'length', test_mass)
        
        print(f"\nTrajectory statistics:")
        print(f"  Min radius: {r_geom.min():.2f} M")
        print(f"  Max radius: {r_geom.max():.2f} M")
        print(f"  Mean radius: {r_geom.mean():.2f} M")
        
        # Check if orbit is stable (within NUMERICAL_THRESHOLDS)
        if r_geom.min() > NUMERICAL_THRESHOLDS['orbit_stability']:
            print(f"  Orbit stability: STABLE (r_min > {NUMERICAL_THRESHOLDS['orbit_stability']} M)")
        else:
            print(f"  Orbit stability: UNSTABLE (approaching singularity)")
            
    else:
        print("\nERROR: Trajectory computation failed!")
        
    # Example with different particles
    print(f"\n\nTesting different particles:")
    particles = {
        'electron': ELECTRON_MASS,
        'proton': PROTON_MASS,
    }
    
    for particle_name, mass in particles.items():
        print(f"\n{particle_name.capitalize()} (mass = {mass:.3e} kg):")
        
        # Quantum effects scale with particle mass
        de_broglie_wavelength = 6.626e-34 / (mass * 1e6)  # At 1 km/s
        print(f"  de Broglie wavelength: {de_broglie_wavelength:.3e} m")
        
        # Run short test
        hist, tag, _ = engine.run_trajectory(
            theory, r0_si, 10, dtau_si,
            use_quantum_trajectories=True,
            particle_name=particle_name,
            no_cache=True
        )
        
        if hist is not None:
            print(f"  Result: {tag}")
        else:
            print(f"  Result: FAILED")

if __name__ == "__main__":
    main() 