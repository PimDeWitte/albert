import torch
import numpy as np
from physics_agent.theories.yukawa.theory import Yukawa
from physics_agent import constants

# Test PPN parameters
theory = Yukawa(lambda_rs=1e6)
print(f"Theory: {theory.name}")
print(f"lambda_rs: {theory.lambda_rs}")
print(f"alpha: {theory.alpha}")

# Use SI units
M_si = 1.989e30  # Solar mass
c_si = 2.998e8   # Speed of light
G_si = 6.674e-11 # Gravitational constant

# Test at weak field distances (100-1000 AU)
AU = 1.496e11  # meters
test_r = torch.tensor([100*AU, 300*AU, 500*AU, 1000*AU])

print(f"\nTesting metric at weak field distances:")
print(f"{'r (AU)':>10} {'r (m)':>15} {'Phi/c²':>12} {'g_rr':>20} {'g_rr-1':>15} {'m':>15} {'1/m':>20} {'est gamma':>10}")
print("-" * 120)

for i, r in enumerate(test_r):
    # Get metric
    g_tt, g_rr, g_pp, g_tp = theory.get_metric(r.unsqueeze(0), M_si, c_si, G_si)
    
    # Compute potential
    Phi_over_c2 = G_si * M_si / (r.item() * c_si**2)
    
    # Extract values
    g_rr_val = g_rr.item()
    g_rr_dev = g_rr_val - 1.0
    
    # Also compute m directly
    rs = 2 * G_si * M_si / c_si**2
    lambda_m = theory.lambda_rs * rs
    yukawa_factor = torch.exp(-r / lambda_m).item()
    m = 1 - rs / r.item() * (1 + theory.alpha * yukawa_factor)
    
    # Estimate gamma: g_rr ≈ 1 + (1+gamma)*2*Phi/c²
    # So gamma ≈ (g_rr-1)/(2*Phi/c²) - 1
    if abs(Phi_over_c2) > 1e-20:
        gamma_est = (g_rr_dev / (2 * Phi_over_c2)) - 1.0
    else:
        gamma_est = float('nan')
    
    print(f"{r.item()/AU:10.0f} {r.item():15.3e} {Phi_over_c2:12.3e} {g_rr_val:20.15f} {g_rr_dev:15.3e} {m:15.10f} {1/m:20.15f} {gamma_est:10.3f}")

# Check the Yukawa correction at these distances
print(f"\nYukawa corrections:")
print(f"{'r (AU)':>10} {'r/lambda':>12} {'exp(-r/λ)':>12} {'correction':>12}")
print("-" * 50)

lambda_phys = theory.lambda_rs * 2 * G_si * M_si / c_si**2  # Convert to meters
for r in test_r:
    r_over_lambda = r.item() / lambda_phys
    yukawa_factor = np.exp(-r_over_lambda)
    correction = theory.alpha * yukawa_factor
    print(f"{r.item()/AU:10.0f} {r_over_lambda:12.3e} {yukawa_factor:12.3e} {correction:12.3e}") 