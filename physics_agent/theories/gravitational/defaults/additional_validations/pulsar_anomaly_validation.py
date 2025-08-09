# pulsar_anomaly_validation.py
import torch
import numpy as np
from typing import Dict, Any
from base_validation import ObservationalValidation
from base_theory import GravitationalTheory
from geodesic_integrator import GeodesicIntegrator
import os
import pandas as pd  # For loading NANOGrav CSV
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'baselines'))
from schwarzschild import Schwarzschild
from reissner_nordstrom import ReissnerNordstrom

class PulsarAnomalyValidation(ObservationalValidation):
    """Validates theories against pulsar timing anomalies, e.g., PSR J2043+1711 acceleration."""
    
    def __init__(self, device=None, dtype=None):
        super().__init__(device, dtype)
        self.data_path = 'path/to/nanograv_15yr_data/PSR_J2043+1711_TOAs.csv'  # Update with actual path
        # <reason>Fallback for dev: If data missing, synth TOAs with anomaly (e.g., accel deviation) to test unification—Einstein's asymmetry could cause such noise as geometric 'wobble' in residuals.</reason>
        self.use_synthetic = not os.path.exists(self.data_path)
        if self.use_synthetic:
            print('Warning: Real data not found—using synthetic TOAs for testing. Download NANOGrav data for real validation.')
    
    def validate(self, theory: GravitationalTheory, **kwargs) -> Dict[str, Any]:
        # <reason>Einstein's deathbed pursuit: Test if geometric info-loss (gamma) explains GR-unexplained pulsar anomalies like J2043+1711's acceleration, treating it as torsion-like deviation in timing residuals without extra fields.</reason>
        kwargs.get('verbose', False)
        
        if self.use_synthetic:
            # <reason>Synth data: Mock MJDs/residuals with GR baseline + injected anomaly (3.5e-3 m/s/yr quadratic) + Gaussian noise; tests if gamma fits better, probing Einstein-inspired info-loss for anomalies.</reason>
            N = 1000  # Mock points
            mjds = torch.linspace(50000, 60000, N, device=self.device, dtype=self.dtype)
            residuals_gr = 0.01 * (mjds - mjds.mean())**2  # Quadratic ~ accel
            residuals_obs = residuals_gr + 3.5e-3 * (mjds - mjds.mean()) + torch.randn(N, device=self.device, dtype=self.dtype) * 1e-3  # Injected + noise
            errors = torch.ones(N, device=self.device, dtype=self.dtype) * 1e-3
        else:
            # Load public NANOGrav TOAs (MJD, residual, error)
            data = pd.read_csv(self.data_path)
            mjds = torch.tensor(data['MJD'].values, device=self.device, dtype=self.dtype)
            residuals_obs = torch.tensor(data['residual'].values, device=self.device, dtype=self.dtype)  # arcsec or similar
            errors = torch.tensor(data['error'].values, device=self.device, dtype=self.dtype)
        
        # Pulsar params (from arXiv:2407.06482)
        M_pulsar = self.tensor(1.4 * 1.989e30)  # Typical NS mass (kg)
        acceleration_obs = 3.5e-3  # m/s/yr, convert to consistent units
        
        # Simulate timing residuals under theory
        # <reason>Simulate geodesic with theory's metric; compute residual deviations vs. GR, checking symmetry with RN for unification (per feedback: predict charged-system anomalies).</reason>
        r0 = self.tensor(1e10)  # Convert to tensor for metric calculations
        y0_full = self.get_initial_conditions(theory, r0=r0, M=M_pulsar)  # Placeholder r0
        integrator = GeodesicIntegrator(theory, y0_full, M_pulsar, self.c, self.G)
        # Integrate over times corresponding to MJDs (simplified; add full timing model)
        N_STEPS = len(mjds)
        hist = self.simulate_trajectory(integrator, N_STEPS, DTau=1e3)  # Placeholder
        residuals_theory = self.compute_residuals(hist, mjds)  # Implement residual calc
        
        # Baselines
        gr_theory = Schwarzschild()
        y0_gr = self.get_initial_conditions(gr_theory, r0=r0, M=M_pulsar)
        integrator_gr = GeodesicIntegrator(gr_theory, y0_gr, M_pulsar, self.c, self.G)
        hist_gr = self.simulate_trajectory(integrator_gr, N_STEPS, DTau=1e3)
        residuals_gr = self.compute_residuals(hist_gr, mjds)
        
        rn_theory = ReissnerNordstrom(Q=1e19)  # Charged proxy
        y0_rn = self.get_initial_conditions(rn_theory, r0=r0, M=M_pulsar)
        integrator_rn = GeodesicIntegrator(rn_theory, y0_rn, M_pulsar, self.c, self.G)
        hist_rn = self.simulate_trajectory(integrator_rn, N_STEPS, DTau=1e3)
        residuals_rn = self.compute_residuals(hist_rn, mjds)
        
        # Compute acceleration from residuals (fit quadratic term)
        accel_theory = self.fit_acceleration(residuals_theory, mjds)
        accel_gr = self.fit_acceleration(residuals_gr, mjds)
        accel_rn = self.fit_acceleration(residuals_rn, mjds)
        
        # Losses (Fourier MSE on residuals, per draft)
        loss_gr = self.fft_mse(residuals_theory, residuals_gr)
        loss_rn = self.fft_mse(residuals_theory, residuals_rn)
        balance = abs(loss_gr - loss_rn)
        
        error = abs(accel_theory - acceleration_obs)
        passed = error < 0.5e-3  # Arbitrary threshold
        
        print(f"    Pulsar anomaly validation complete for {theory.name}:")
        print(f"      Observed accel: {acceleration_obs:.2e} m/s/yr")
        print(f"      Predicted: {accel_theory:.2e} m/s/yr")
        print(f"      GR: {accel_gr:.2e}, RN: {accel_rn:.2e}")
        print(f"      Loss balance: {balance:.3f}")
        print(f"      Result: {'PASSED' if passed else 'FAILED'}")
        
        return {
            'test_name': 'PSR J2043+1711 Acceleration Anomaly',
            'observed_accel': acceleration_obs,
            'predicted_accel': accel_theory,
            'loss_gr': loss_gr,
            'loss_rn': loss_rn,
            'balance': balance,
            'passed': passed
        }
    
    # <reason>Helper: Simulate trajectory and residuals; Einstein link: Torsion/asymmetry could cause timing noise as geometric 'wobble'—test if gamma replicates this.</reason>
    def simulate_trajectory(self, integrator, N_STEPS, DTau):
        # Similar to mercury script's integration loop
        hist = []
        # Get initial state from integrator's initialization
        y = self.get_initial_conditions(integrator.model, self.tensor(1e10), integrator.M)[[0, 1, 2, 4]]  # [t, r, phi, dr/dtau]
        
        for i in range(N_STEPS):
            hist.append(y)
            y = integrator.rk4_step(y, DTau)
        return torch.stack(hist)
    
    def compute_residuals(self, hist, mjds):
        # <reason>Fix residuals: Use proper time dilation from metric to get gamma-dependent timing</reason>
        # Extract time and radial components
        t_coords = hist[:, 0]  # coordinate time
        r_coords = hist[:, 1]  # radial position
        
        # For a pulsar in orbit, timing residuals come from:
        # 1. Orbital motion (Roemer delay)
        # 2. Gravitational time dilation (Einstein delay)
        # 3. Anomalous acceleration
        
        # Simplified: Use coordinate time evolution which includes metric effects
        # The rate dt/dτ varies with the metric, giving gamma-dependent residuals
        
        # Expected linear time evolution
        t0 = t_coords[0]
        t_expected = t0 + torch.arange(len(t_coords), device=t_coords.device) * (t_coords[-1] - t0) / (len(t_coords) - 1)
        
        # Residuals are deviation from linear time
        residuals = t_coords - t_expected
        
        # Scale to realistic units (convert from geometric to physical)
        # Typical pulsar timing precision is microseconds
        residuals = residuals * 1e6  # Convert to microseconds
        
        return residuals
    
    def fit_acceleration(self, residuals, mjds):
        # Polyfit quadratic term ~ acceleration
        # Handle both tensor and numpy/pandas inputs
        if torch.is_tensor(mjds):
            mjds_np = mjds.cpu().numpy()
        else:
            mjds_np = np.array(mjds)
        
        if torch.is_tensor(residuals):
            residuals_np = residuals.cpu().numpy()
        else:
            residuals_np = np.array(residuals)
            
        coeffs = np.polyfit(mjds_np, residuals_np, 2)
        return coeffs[0] * 2  # accel = 2 * quadratic coeff (units adjusted)
    
    def fft_mse(self, a, b):
        # Fourier MSE as in draft
        fft_a = torch.fft.fft(a)
        fft_b = torch.fft.fft(b)
        return torch.mean(torch.abs(fft_a - fft_b)**2).item() 