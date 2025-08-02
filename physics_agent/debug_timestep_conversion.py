#!/usr/bin/env python3
"""Debug timestep conversion issues."""

import torch
import sys
sys.path.insert(0, '.')

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild

def main():
    # Create engine  
    engine = TheoryEngine(black_hole_preset='primordial_mini')
    
    print("Unit conversions:")
    print(f"  length_scale (GM/c²) = {engine.length_scale:.3e} m")
    print(f"  time_scale (GM/c³) = {engine.time_scale:.3e} s")
    print(f"  c = {engine.c_si:.3e} m/s")
    print(f"  G = {engine.G_si:.3e} m³/kg/s²")
    print(f"  M = {engine.M_si:.3e} kg\n")
    
    # Initial conditions
    r0_si = 10 * engine.length_scale  # 10 Schwarzschild radii
    r0_geom = r0_si / engine.length_scale
    
    print(f"Initial radius:")
    print(f"  r0_si = {r0_si:.3e} m") 
    print(f"  r0_geom = {r0_geom:.3f} (geometric units)\n")
    
    # Timestep analysis
    dtau_si = 1e-6  # 1 microsecond
    dtau_geom = dtau_si / engine.time_scale
    
    print(f"Timestep:")
    print(f"  dtau_si = {dtau_si:.3e} s")
    print(f"  dtau_geom = {dtau_geom:.3e} (geometric units)")
    print(f"  dtau_geom is {dtau_geom:.3e} in units of GM/c³\n")
    
    # For comparison, orbital period at r=10M
    T_orbit = 2 * 3.14159 * (r0_geom**1.5)  # Keplerian period in geometric units
    T_orbit_si = T_orbit * engine.time_scale
    
    print(f"Orbital period at r=10M:")
    print(f"  T = {T_orbit:.3f} (geometric units)")
    print(f"  T = {T_orbit_si:.3e} s")
    print(f"  dtau/T = {dtau_si/T_orbit_si:.3e}\n")
    
    # Recommended timestep
    recommended_dtau_geom = 0.1  # From black hole config
    recommended_dtau_si = recommended_dtau_geom * engine.time_scale
    
    print(f"Recommended timestep from config:")
    print(f"  dtau_geom = {recommended_dtau_geom:.3f}")
    print(f"  dtau_si = {recommended_dtau_si:.3e} s")
    print(f"  This is {recommended_dtau_si/dtau_si:.3e}x smaller than our test\n")
    
    # The issue: dtau_si = 1e-6 s is HUGE compared to time_scale = 2.477e-21 s
    print("⚠️  PROBLEM IDENTIFIED:")
    print(f"  Our timestep {dtau_si:.3e} s is {dtau_si/engine.time_scale:.3e} geometric time units!")
    print(f"  This is way too large - should be ~0.1 geometric units")
    print(f"  Solution: Use dtau_si = {recommended_dtau_si:.3e} s instead")

if __name__ == "__main__":
    main()