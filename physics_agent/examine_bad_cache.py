#!/usr/bin/env python3
"""Examine the specific cache file that showed r=0."""

import torch
import numpy as np

# The file we examined earlier that showed r ≈ 0
cache_file = "physics_agent/cache/trajectories/1.0.0/Primordial_Mini_Black_Hole/Schwarzschild_a8b7beb9b5aa8c8f_steps_1000.pt"

print(f"Examining: {cache_file}\n")

data = torch.load(cache_file, map_location='cpu')

# Extract trajectory
if isinstance(data, dict) and 'trajectory' in data:
    hist = data['trajectory']
else:
    hist = data

print(f"Trajectory shape: {hist.shape}")
print(f"Data type: {hist.dtype}\n")

# Look at r values more carefully
r_vals = hist[:, 1]

print(f"r statistics:")
print(f"  min: {r_vals.min():.15e} m")
print(f"  max: {r_vals.max():.15e} m") 
print(f"  mean: {r_vals.mean():.15e} m")
print(f"  std: {r_vals.std():.15e} m\n")

# Check actual values
print(f"First 10 r values:")
for i in range(10):
    print(f"  r[{i}] = {r_vals[i]:.15e} m")

print(f"\nLast 5 r values:")
for i in range(len(r_vals)-5, len(r_vals)):
    print(f"  r[{i}] = {r_vals[i]:.15e} m")
    
# The issue from earlier
print(f"\nChecks:")
print(f"  r < 1e-10? {(r_vals.max() < 1e-10).item()}")
print(f"  r < 1e-15? {(r_vals.max() < 1e-15).item()}")

# Compute r in Schwarzschild radii
rs = 1.485e-12  # 2 * length_scale for primordial mini
r_in_rs = r_vals / rs

print(f"\nIn Schwarzschild radii:")
print(f"  r/Rs min: {r_in_rs.min():.3f}")
print(f"  r/Rs max: {r_in_rs.max():.3f}")
print(f"  Expected: ~5.0 (since r = 10M = 5Rs)")

# Check if this is actually correct
expected_r = 7.426e-12  # 10 * length_scale
print(f"\nExpected r = {expected_r:.3e} m")
print(f"Actual r ≈ {r_vals.mean():.3e} m")
print(f"Match? {np.isclose(r_vals.mean().item(), expected_r, rtol=0.01)}")