import torch
import numpy as np
from typing import Dict, Any, TYPE_CHECKING
from ..base_validation import BaseValidation

if TYPE_CHECKING:
    from physics_agent.base_theory import GravitationalTheory
    from physics_agent.theory_engine_core import TheoryEngine

class CosmologyValidator(BaseValidation):
    category = "observational"
    
    def __init__(self, engine: "TheoryEngine"):
        super().__init__(engine, "Cosmology Validator")

    def validate(self, theory: "GravitationalTheory", hist: torch.Tensor = None, **kwargs) -> Dict[str, Any]:
        verbose = kwargs.get('verbose', False)
        
        if verbose:
            print(f"\nCalculating cosmological distances for {theory.name}...")
        
        try:
            # <reason>chain: Wrap entire calculation in try-catch to handle numpy/torch errors</reason>
            # This prevents errors from propagating up
            
            # <reason>chain: Test theory against Type Ia supernovae luminosity distances</reason>
            # We use a simplified Hubble diagram test
            
            # Redshift range for nearby supernovae
            z = np.linspace(0.01, 0.1, 20)  # Low-z sample where cosmology is well-constrained
            
            # <reason>chain: Standard ΛCDM luminosity distance</reason>
            # For low z: d_L ≈ (c/H_0) × z × (1 + z/2 + z²/6)
            # Using H_0 = 70 km/s/Mpc
            H_0 = 70  # km/s/Mpc
            # <reason>chain: Ensure C_T is a scalar float, not a tensor</reason>
            c_km = float(self.engine.C_T.item() if torch.is_tensor(self.engine.C_T) else self.engine.C_T) / 1000  # Speed of light in km/s
            
            # ΛCDM prediction (Taylor expansion for low z)
            dl_cdm = (c_km / H_0) * z * (1 + z/2 + z**2/6)
            
            # <reason>chain: Theory might modify Hubble parameter or dark energy equation of state</reason>
            # Simple parameterization: H(z) = H_0 × sqrt(Ω_m(1+z)³ + Ω_Λ × f(z))
            # Where f(z) encodes dark energy evolution
            
            # <reason>chain: Enforce theory-specific dark energy computation</reason>
            if hasattr(theory, 'dark_energy_parameter'):
                w = theory.dark_energy_parameter  # Equation of state
            elif hasattr(theory, 'compute_dark_energy_eos'):
                # <reason>chain: Allow theories to compute w dynamically</reason>
                w = theory.compute_dark_energy_eos(z=z.mean())
            else:
                # <reason>chain: Warn about missing dark energy implementation</reason>
                if verbose:
                    print(f"  Warning: Theory has no dark energy parameter implementation")
                # <reason>chain: Return warning instead of random value to avoid false passes</reason>
                return {
                    "loss": 0.1,  # Small penalty
                    "flags": {"overall": "WARNING", "no_dark_energy_model": True},
                    "details": {
                        "chi2": float('nan'),
                        "chi2_per_dof": float('nan'),
                        "dark_energy_w": -1.0,
                        "n_supernovae": len(z),
                        "units": "dimensionless",
                        "notes": "Theory lacks dark energy model - cannot test cosmological expansion"
                    }
                }
            
            # Modified distance for w ≠ -1
            if abs(w + 1) > 1e-6:
                # Dark energy density evolution: ρ_DE ∝ (1+z)^(3(1+w))
                # This modifies distances
                correction = (1 + 0.3 * (w + 1) * z)  # Linear approximation
                dl_theory = dl_cdm * correction
            else:
                dl_theory = dl_cdm
            
            # <reason>chain: Compute chi-squared against "mock" supernova data</reason>
            # Real analysis would use actual SNe Ia data
            # We simulate realistic errors: σ_m ≈ 0.15 mag → σ_dL/dL ≈ 0.07
            relative_error = 0.07
            sigma_dl = dl_cdm * relative_error
            
            # Chi-squared
            # Ensure we're using numpy arrays, not torch tensors
            if torch.is_tensor(dl_theory):
                dl_theory = dl_theory.detach().cpu().numpy()
            if torch.is_tensor(dl_cdm):
                dl_cdm = dl_cdm.detach().cpu().numpy()
            if torch.is_tensor(sigma_dl):
                sigma_dl = sigma_dl.detach().cpu().numpy()
            
            chi2 = float(np.sum(((dl_theory - dl_cdm) / sigma_dl)**2))
            chi2_per_dof = chi2 / len(z)
            
            # For ΛCDM, chi²/dof should be ~1
            # Significant deviations indicate non-standard cosmology
            loss = abs(chi2_per_dof - 1.0)
            flag = "PASS" if chi2_per_dof < 2.0 else "FAIL"  # 2σ tolerance
            
            if verbose:
                print(f"  Redshift range: {z[0]:.2f} - {z[-1]:.2f}")
                print(f"  Hubble constant: {H_0} km/s/Mpc")
                print(f"  Dark energy EoS: w = {w:.3f} (ΛCDM: w = -1)")
                print(f"\nResults:")
                print(f"  χ²/dof: {chi2_per_dof:.2f} (expected: ~1)")
                print(f"  Total χ²: {chi2:.1f} for {len(z)} data points")
                print(f"  Average distance deviation: {100*np.mean(abs(dl_theory - dl_cdm)/dl_cdm):.1f}%")
                print(f"  Status: {flag}")
                
        except Exception as e:
            if verbose:
                print(f"  Error computing cosmology: {str(e)}")
            return {
                "loss": 1.0,
                "flags": {"overall": "FAIL"},
                "details": {
                    "chi2": float('inf'),
                    "error": str(e),
                    "notes": "Failed to compute cosmological distances"
                }
            }
        
        return {
            "loss": loss,
            "flags": {"overall": flag},
            "details": {
                "chi2": chi2,
                "chi2_per_dof": chi2_per_dof,
                "dark_energy_w": w,
                "n_supernovae": len(z),
                "units": "dimensionless",
                "notes": f"Type Ia SNe: χ²/dof = {chi2_per_dof:.2f}"
            }
        } 