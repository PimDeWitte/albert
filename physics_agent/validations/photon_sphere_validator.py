import torch
from typing import Dict, Any, TYPE_CHECKING
from .base_validation import BaseValidation

if TYPE_CHECKING:
    from physics_agent.base_theory import GravitationalTheory
    from physics_agent.theory_engine_core import TheoryEngine

class PhotonSphereValidator(BaseValidation):
    category = "observational"
    
    def __init__(self, engine: "TheoryEngine", tolerance: float = 1e-3):
        super().__init__(engine, "Photon Sphere Validator")
        self.tolerance = tolerance

    def validate(self, theory: "GravitationalTheory", hist: torch.Tensor = None, **kwargs) -> Dict[str, Any]:
        verbose = kwargs.get('verbose', False)
        
        if verbose:
            print(f"\nCalculating photon sphere for {theory.name}...")
        
        try:
            # <reason>chain: Search for photon sphere radius where effective potential has extremum</reason>
            # For photons, the effective potential extremum determines circular orbits
            # In Schwarzschild: r_ph = 3GM/c² = 1.5 r_s
            
            rs = 2 * self.engine.G_T * self.engine.M / self.engine.C_T**2  # Schwarzschild radius
            
            # Search range: 1.1 to 5 Schwarzschild radii
            r_test = torch.linspace(1.1, 5.0, 200, device=self.engine.device, dtype=self.engine.dtype) * rs
            
            g_tt, g_rr, g_pp, g_tp = theory.get_metric(r_test, self.engine.M, self.engine.C_T, self.engine.G_T)
            
            # <reason>chain: For circular photon orbits, we need d(L²/E²)/dr = 0</reason>
            # Where L²/E² = r²g_pp/g_tt for equatorial orbits
            # This gives the photon sphere radius
            
            # Compute effective potential ratio
            potential_ratio = r_test**2 * g_pp / g_tt
            
            # Find extremum by numerical differentiation
            dr = r_test[1] - r_test[0]
            d_potential = torch.gradient(potential_ratio, spacing=dr)[0]
            
            # Find where derivative is closest to zero
            min_idx = torch.argmin(torch.abs(d_potential))
            r_photon = r_test[min_idx].item()
            
            # <reason>chain: Check stability using second derivative for unstable orbit verification</reason>
            d2_potential = torch.gradient(d_potential, spacing=dr)[0]
            stability_check = d2_potential[min_idx].item()
            is_unstable = stability_check < 0  # Photon sphere should be unstable (maximum)
            
            # Convert to units of Schwarzschild radii
            r_photon_rs = r_photon / rs.item()
            
            # <reason>chain: Shadow diameter relates to photon sphere via d = 2√27 * r_ph for Schwarzschild</reason>
            # This gives approximately 5.196 r_s for Schwarzschild black hole
            shadow_diameter_rs = 2 * torch.sqrt(torch.tensor(27.0)) * r_photon / rs.item()
            
            # Expected values for Schwarzschild
            expected_r_ph = 1.5  # in units of r_s
            expected_shadow = 5.196  # in units of r_s
            
            # Compute errors
            abs(r_photon_rs - expected_r_ph) / expected_r_ph
            shadow_error = abs(shadow_diameter_rs - expected_shadow) / expected_shadow
            
            # <reason>chain: Include stability check in validation</reason>
            # Photon sphere should be unstable for physical black holes
            if not is_unstable and shadow_error < self.tolerance:
                if verbose:
                    print(f"  Warning: Photon sphere appears stable (d²V/dr² = {stability_check:.3e} > 0)")
                shadow_error += 0.1  # Add penalty for non-physical stability
            
            # Use shadow diameter error as primary metric
            loss = shadow_error
            flag = "PASS" if loss < self.tolerance else "FAIL"
            
            if verbose:
                print(f"  Schwarzschild radius: {rs.item():.3e} m")
                print(f"  Search range: {1.1:.1f} - {5.0:.1f} r_s")
                print(f"\nResults:")
                print(f"  Photon sphere radius: {r_photon_rs:.3f} r_s (expected: {expected_r_ph:.3f} r_s)")
                print(f"  Shadow diameter: {shadow_diameter_rs:.3f} r_s (expected: {expected_shadow:.3f} r_s)")
                print(f"  Orbit stability: {'Unstable (correct)' if is_unstable else 'Stable (non-physical)'}")
                print(f"  Error: {100*shadow_error:.2f}%")
                print(f"  Status: {flag}")
                
        except Exception as e:
            if verbose:
                print(f"  Error computing photon sphere: {str(e)}")
            return {
                "loss": 1.0,
                "flags": {"overall": "FAIL"},
                "details": {
                    "shadow_diameter": float('nan'),
                    "photon_radius": float('nan'),
                    "error": str(e),
                    "notes": "Failed to compute photon sphere"
                }
            }
        
        return {
            "loss": loss,
            "flags": {"overall": flag, "unstable_orbit": is_unstable},
            "details": {
                "shadow_diameter": shadow_diameter_rs,
                "photon_radius": r_photon_rs,
                "expected_shadow": expected_shadow,
                "expected_r_ph": expected_r_ph,
                "error_percent": 100 * shadow_error,
                "is_unstable": is_unstable,
                "stability_parameter": stability_check,
                "units": "Schwarzschild radii",
                "notes": f"Black hole shadow: {100*shadow_error:.1f}% error from GR" + (", orbit stable (non-physical)" if not is_unstable else "")
            }
        } 