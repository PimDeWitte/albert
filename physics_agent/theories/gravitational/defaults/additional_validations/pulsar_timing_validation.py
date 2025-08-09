# pulsar_timing_validation.py
import torch
import numpy as np
from typing import Dict, Any
from base_validation import ObservationalValidation
from base_theory import GravitationalTheory
from geodesic_integrator import GeodesicIntegrator
import math

class PulsarTimingValidation(ObservationalValidation):
    """Validates theories against pulsar timing observations."""
    
    def __init__(self, device=None, dtype=None):
        super().__init__(device, dtype)
        
    def validate(self, theory: GravitationalTheory, **kwargs) -> Dict[str, Any]:
        """
        Validate a theory against PSR B1913+16 observations.
        """
        # Check for verbose mode
        verbose = kwargs.get('verbose', False)
        
        # System parameters (all as tensors on correct device)
        M1 = self.tensor(1.4408 * 1.989e30)  # Primary mass (kg)
        M2 = self.tensor(1.3886 * 1.989e30)  # Secondary mass (kg)
        M_total = M1 + M2
        
        # Orbital parameters
        a = self.tensor(1.95e9)  # Semi-major axis (m)
        e = self.tensor(0.6171338)  # Eccentricity
        P = self.tensor(27906.98163)  # Orbital period (s)
        
        # Observed periastron advance
        observed_advance = 4.226595  # degrees per year
        
        # Initial conditions at periastron
        r_peri = a * (1 - e)
        r0 = r_peri
        
        # Simulation parameters
        if kwargs.get('test', False):
            n_orbits = 5  # Increased for better fitting
            steps_per_orbit = 200
        else:
            n_orbits = 10
            steps_per_orbit = 5000
        N_STEPS = n_orbits * steps_per_orbit
        DTau = P / steps_per_orbit
        
        print(f"\n    Starting pulsar timing validation for {theory.name}...")
        if verbose:
            print(f"    Device: {self.device}, Dtype: {self.dtype}")
            print(f"    N_STEPS: {N_STEPS}, DTau: {DTau:.3e}")
            print(f"    Initial r0: {r0:.3e}")
            print(f"    Total mass: {M_total:.3e} kg ({(M_total/1.989e30).cpu().item():.2f} M_sun)")
        
        # If theory has custom PN gamma, use that; else simulate
        if hasattr(theory, 'get_ppn_gamma'):
            gamma = theory.get_ppn_gamma()
        else:
            gamma = 1.0  # Assume GR-like

        # PN periastron advance for binary (degrees/year)
        mu = self.G * M_total
        n = 2 * math.pi / P  # Mean motion
        advance_rad_per_orbit = (3 * (n**(2/3) * mu / self.c**2)**(1/3)) / (1 - e**2) * (1 + (1 + gamma)/4)  # Simplified PN formula
        orbits_per_year = (365.25 * 86400) / P
        advance_rad_per_orbit * orbits_per_year * (180 / math.pi)

        # Rest of calculation..."""
        
        # Get initial conditions (using same method as self_discovery.py)
        y0_full = self.get_initial_conditions(theory, r0, M_total)
        y0_state = y0_full[[0, 1, 2, 4]]  # [t, r, phi, dr/dtau]
        
        # Initialize integrator
        integrator = GeodesicIntegrator(theory, y0_full, M_total, self.c, self.G)
        
        # Run simulation
        hist = torch.empty((N_STEPS + 1, 4), device=self.device, dtype=self.dtype)
        hist[0] = y0_state
        y = y0_state.clone()
        periastron_times = []
        periastron_angles = []
        
        # Progress logging
        step_print = 100 if kwargs.get('test', False) else 1000
        if verbose:
            step_print = 100
        
        for i in range(N_STEPS):
            y = integrator.rk4_step(y, DTau)
            y = y.to(self.device)
            hist[i + 1] = y
            
            if (i + 1) % step_print == 0:
                print(f"      Step {i+1}/{N_STEPS} | r={y[1]/a:.3f} a")
            
            # Improved periastron detection with interpolation
            if i > 1:
                r_prev2 = hist[i-1, 1]
                r_prev = hist[i, 1]
                r_curr = y[1]
                if r_prev < r_prev2 and r_prev < r_curr:  # Local minimum
                    t0, t1, t2 = hist[i-1, 0], hist[i, 0], y[0]
                    r0, r1, r2 = r_prev2, r_prev, r_curr
                    denom = (t0-t1)*(t0-t2)*(t1-t2)
                    A = (t2*(r1-r0) + t1*(r0-r2) + t0*(r2-r1)) / denom if denom != 0 else 0
                    B = ((t2**2)*(r0-r1) + (t1**2)*(r2-r0) + (t0**2)*(r1-r2)) / denom if denom != 0 else 0
                    t_min = -B / (2*A) if A != 0 else t1
                    phi_min = hist[i, 2] + (t_min - t1) * (y[2] - hist[i, 2]) / DTau
                    periastron_times.append(t_min)
                    periastron_angles.append(phi_min)
                    if verbose:
                        print(f"      Periastron {len(periastron_times)} at t={t_min:.1f}s, φ={phi_min:.3f} rad")
            
            if not torch.all(torch.isfinite(y)):
                print(f"      Warning: Non-finite values at step {i} - aborting")
                hist = hist[:i+2]
                break
                
        hist = hist.to(self.device)
        
        # New: Run GR baseline for comparison
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'baselines'))
        from schwarzschild import Schwarzschild
        gr_theory = Schwarzschild()
        gr_y0_full = self.get_initial_conditions(gr_theory, r0, M_total)
        gr_y0_state = gr_y0_full[[0, 1, 2, 4]]
        gr_integrator = GeodesicIntegrator(gr_theory, gr_y0_full, M_total, self.c, self.G)
        gr_hist = self.empty((N_STEPS + 1, 4))
        gr_hist[0] = gr_y0_state
        gr_y = gr_y0_state.clone()
        for i in range(N_STEPS):
            gr_y = gr_integrator.rk4_step(gr_y, DTau)
            gr_hist[i + 1] = gr_y
            if not torch.all(torch.isfinite(gr_y)):
                break
        gr_hist = gr_hist.to(self.device)
        
        # Compute running MSE vs. GR
        mse_vs_gr = torch.cumsum((hist[:, 1] - gr_hist[:, 1])**2, dim=0) / (torch.arange(1, len(hist)+1, device=self.device, dtype=self.dtype))
        if verbose:
            print(f"      Final MSE vs. GR: {mse_vs_gr[-1]:.3e}")
        
        # Log progress every 10%
        if verbose:
            for pct in range(10, 101, 10):
                step = int(N_STEPS * pct / 100)
                current_mse = mse_vs_gr[step-1]
                prev_mse = mse_vs_gr[step//2] if step > 1 else 0
                trend = 'improving' if current_mse < prev_mse else 'degrading'
                print(f"      At {pct}% (step {step}): MSE={current_mse:.3e} ({trend})")
        
        # Generate interactive viz
        self.generate_viz(theory, hist, os.path.dirname(__file__), gr_hist=gr_hist)

        # Calculate periastron advance properly
        if len(periastron_angles) >= 3:  # Require at least 3 for reliable fit
            times_np = np.array([t.cpu().item() for t in periastron_times])
            angles_np = np.array([phi.cpu().item() for phi in periastron_angles])
            angles_unwrapped = np.unwrap(angles_np)
            coeffs = np.polyfit(times_np, angles_unwrapped, 1)
            if verbose:
                print(f"      Fitted omega_dot = {coeffs[0]:.3e} rad/s")
            
            omega_dot = coeffs[0]
            advance_deg_yr = omega_dot * (365.25 * 86400) * (180 / np.pi)
            
            # Subtract Post-Keplerian base rate for binary
            mu = self.G.item() * M_total.item()
            omega_PK = (2 * np.pi / P.item()) * (1 - e.item()**2)**(-3/2) * (mu / self.c.item()**3)**(2/3) * (1 + (3/2) * (mu / self.c.item()**3)**(2/3))
            advance_deg_yr -= omega_PK * (365.25 * 86400) * (180 / np.pi)
            
            # GR reference for binary
            gr_advance = 3 * (2 * np.pi / P.item())**(5/3) * (mu / self.c.item()**3)**(2/3) / (1 - e.item()**2)
            gr_deg_yr = gr_advance * (365.25 * 86400) * (180 / np.pi)
            if verbose:
                print(f"      GR expected: {gr_deg_yr:.3f} deg/yr")
        else:
            print("      Warning: Insufficient periastrons detected ({len(periastron_angles)}), using PN approximation fallback")
            # PN fallback (similar to new code)
            gamma = theory.get_ppn_gamma() if hasattr(theory, 'get_ppn_gamma') else 1.0
            mu = self.G * M_total
            n = 2 * math.pi / P
            advance_rad_per_orbit = (3 * (n**(2/3) * mu / self.c**2)**(1/3)) / (1 - e**2) * (1 + (1 + gamma)/4)
            orbits_per_year = (365.25 * 86400) / P
            advance_deg_yr = advance_rad_per_orbit * orbits_per_year * (180 / math.pi)
            # Subtract PK base if needed...
            # GR reference for binary
            gr_advance = 3 * (2 * np.pi / P.item())**(5/3) * (mu / self.c.item()**3)**(2/3) / (1 - e.item()**2)
            gr_deg_yr = gr_advance * (365.25 * 86400) * (180 / np.pi)
            if verbose:
                print(f"      GR expected: {gr_deg_yr:.3f} deg/yr")
            
        error = abs(advance_deg_yr - observed_advance)
        passed = error < 0.01  # Tightened for precision
        
        print(f"    Pulsar timing validation complete:")
        print(f"      Observed: {observed_advance:.3f} deg/yr")
        print(f"      Predicted: {advance_deg_yr:.3f} deg/yr")
        print(f"      Error: {error:.3f} deg/yr")
        print(f"      Result: {'PASSED' if passed else 'FAILED'}")
        
        return {
            'test_name': 'PSR B1913+16 Periastron Advance',
            'observed': observed_advance,
            'predicted': advance_deg_yr,
            'error': error,
            'units': 'degrees/year',
            'passed': bool(passed),  # Convert numpy bool to Python bool
            'trajectory': hist.cpu().numpy()
        } 