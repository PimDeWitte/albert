#!/usr/bin/env python3
"""
Test script for Unified Gauge Model (UGM) implementation.

<reason>chain: Verify that the four U(1) gauge symmetries correctly generate gravity</reason>
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from theory import UnifiedGaugeModel

def test_metric_recovery():
    """
    Test that UGM recovers Schwarzschild metric in weak-field limit.
    
    <reason>chain: In weak-field limit with equal alphas, should approach GR</reason>
    """
    print("Testing UGM metric recovery...")
    
    # Create UGM instance with equal weights
    ugm = UnifiedGaugeModel(alpha0=1.0, alpha1=1.0, alpha2=1.0, alpha3=1.0, g_coupling=0.1)
    
    # Test at various radii
    r_values = torch.linspace(3.0, 50.0, 100, dtype=torch.float64)
    
    # Standard parameters (geometric units)
    M = torch.tensor(1.0, dtype=torch.float64)
    c = 1.0
    G = 1.0
    
    # Get metric components
    g_tt, g_rr, g_pp, g_tp = ugm.get_metric(r_values, M, c, G)
    
    # Compare with Schwarzschild
    rs = 2.0  # Schwarzschild radius
    g_tt_schw = -(1 - rs / r_values)
    g_rr_schw = 1 / (1 - rs / r_values)
    
    # Calculate relative errors
    err_tt = torch.abs((g_tt - g_tt_schw) / g_tt_schw)
    err_rr = torch.abs((g_rr - g_rr_schw) / g_rr_schw)
    
    print(f"Max g_tt error: {err_tt.max().item():.2e}")
    print(f"Max g_rr error: {err_rr.max().item():.2e}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(r_values.numpy(), -g_tt.numpy(), 'b-', label='UGM')
    plt.plot(r_values.numpy(), -g_tt_schw.numpy(), 'r--', label='Schwarzschild')
    plt.xlabel('r (geometric units)')
    plt.ylabel('-g_tt')
    plt.legend()
    plt.title('Time Component Comparison')
    
    plt.subplot(1, 2, 2)
    plt.plot(r_values.numpy(), g_rr.numpy(), 'b-', label='UGM')
    plt.plot(r_values.numpy(), g_rr_schw.numpy(), 'r--', label='Schwarzschild')
    plt.xlabel('r (geometric units)')
    plt.ylabel('g_rr')
    plt.legend()
    plt.title('Radial Component Comparison')
    
    plt.tight_layout()
    plt.savefig('ugm_metric_test.png')
    print("Saved metric comparison to ugm_metric_test.png")
    

def test_parameter_variation():
    """
    Test how varying the four U(1) couplings affects the metric.
    
    <reason>chain: Different alphas should produce distinct gravitational signatures</reason>
    """
    print("\nTesting parameter variation...")
    
    r = torch.tensor([10.0], dtype=torch.float64)  # Test at r=10
    M = torch.tensor(1.0, dtype=torch.float64)
    
    # Vary alpha0 (time component weight)
    alpha0_values = np.linspace(0.5, 2.0, 10)
    g_tt_values = []
    
    for a0 in alpha0_values:
        ugm = UnifiedGaugeModel(alpha0=a0, alpha1=1.0, alpha2=1.0, alpha3=1.0, g_coupling=0.1)
        g_tt, _, _, _ = ugm.get_metric(r, M, 1.0, 1.0)
        g_tt_values.append(g_tt.item())
    
    plt.figure(figsize=(8, 6))
    plt.plot(alpha0_values, g_tt_values, 'b-')
    plt.xlabel('α₀ (time U(1) coupling)')
    plt.ylabel('g_tt at r=10')
    plt.title('Effect of Time U(1) Coupling on Metric')
    plt.grid(True)
    plt.savefig('ugm_alpha0_variation.png')
    print("Saved alpha0 variation to ugm_alpha0_variation.png")
    

def test_field_strengths():
    """
    Test computation of U(1) field strengths.
    
    <reason>chain: Field strengths should satisfy antisymmetry F_μν = -F_νμ</reason>
    """
    print("\nTesting field strength computation...")
    
    ugm = UnifiedGaugeModel()
    
    # Create test coordinates
    coords = torch.tensor([[0.0, 10.0, np.pi/2, 0.0]], dtype=torch.float64, requires_grad=True)
    
    # Get gauge fields
    H = ugm.initialize_gauge_fields(coords[:, 1], coords[:, 2], coords[:, 3])
    
    # Compute field strengths
    F = ugm.compute_field_strengths(H, coords)
    
    # Check antisymmetry
    max_asymm = 0.0
    for a in range(4):
        for mu in range(4):
            for nu in range(4):
                asymm = torch.abs(F[a, mu, nu] + F[a, nu, mu]).max().item()
                max_asymm = max(max_asymm, asymm)
    
    print(f"Max antisymmetry violation: {max_asymm:.2e}")
    
    # Compute Lagrangian density
    L = ugm.lagrangian_density(H, F)
    print(f"Lagrangian density at r=10: {L.item():.6f}")
    

def test_tetrad_metric_relation():
    """
    Test that g_μν = η_ab e^a_μ e^b_ν holds exactly.
    
    <reason>chain: Fundamental tetrad-metric relation must be satisfied</reason>
    """
    print("\nTesting tetrad-metric relation...")
    
    ugm = UnifiedGaugeModel(g_coupling=0.05)  # Small coupling for clarity
    
    r = torch.tensor([5.0, 10.0, 20.0], dtype=torch.float64)
    theta = torch.full_like(r, np.pi/2)
    phi = torch.zeros_like(r)
    
    # Get gauge fields and build tetrad
    H = ugm.initialize_gauge_fields(r, theta, phi)
    e = torch.eye(4, dtype=torch.float64).reshape(4, 4, 1) + ugm.g * H
    
    # Compute metric via tetrad formula
    eta = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    g_from_tetrad = torch.einsum('ab,aim,bjm->ijm', eta, e, e)
    
    # Get metric from theory
    g_tt, g_rr, g_pp, g_tp = ugm.get_metric(r, torch.tensor(1.0), 1.0, 1.0)
    
    # Compare components
    print(f"g_tt match: {torch.allclose(g_from_tetrad[0, 0], g_tt, rtol=1e-10)}")
    print(f"g_rr match: {torch.allclose(g_from_tetrad[1, 1], g_rr, rtol=1e-10)}")
    print(f"g_φφ match: {torch.allclose(g_from_tetrad[3, 3], g_pp, rtol=1e-10)}")
    print(f"g_tφ match: {torch.allclose(g_from_tetrad[0, 3], g_tp, rtol=1e-10)}")
    

if __name__ == "__main__":
    print("=== Unified Gauge Model (UGM) Test Suite ===")
    print("Testing Partanen & Tulkki (2025) framework implementation\n")
    
    test_metric_recovery()
    test_parameter_variation()
    test_field_strengths()
    test_tetrad_metric_relation()
    
    print("\n✓ All tests completed!") 