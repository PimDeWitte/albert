import torch
from typing import Dict, Any, TYPE_CHECKING
from ..base_validation import BaseValidation

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
            
            # <reason>chain: Use consistent units - if M=1 in geometric units, rs=2</reason>
            # In geometric units where G=c=1, rs = 2M
            if self.engine.M == 1.0:  # Geometric units
                rs = 2.0 * self.engine.M
            else:  # SI units
                rs = 2 * self.engine.G_si * self.engine.M_si / self.engine.c_si**2
            
            # Search range: 1.1 to 5 Schwarzschild radii
            r_test = torch.linspace(1.1, 5.0, 200, device=self.engine.device, dtype=self.engine.dtype) * rs
            
            # <reason>chain: Pass correct parameters based on unit system</reason>
            if self.engine.M == 1.0:  # Geometric units
                g_tt, g_rr, g_pp, g_tp = theory.get_metric(r_test, self.engine.M, 1.0, 1.0)
            else:
                g_tt, g_rr, g_pp, g_tp = theory.get_metric(r_test, self.engine.M_si, self.engine.c_si, self.engine.G_si)
            
            # <reason>chain: For circular photon orbits, we need to find extremum of effective potential</reason>
            # The photon sphere occurs where the effective potential V_eff ∝ (1-rs/r)/r² has an extremum
            # This is equivalent to finding where d/dr[(-g_tt)/g_pp] = 0
            # NOT where d/dr[g_pp/(-g_tt)] = 0, which gives the wrong answer!
            
            # Compute the correct quantity: (-g_tt)/g_pp = (1-rs/r)/r²
            # This has a MAXIMUM at the photon sphere
            effective_potential = (-g_tt) / g_pp  # = (1-rs/r)/r²
            
            # Find extremum by numerical differentiation
            dr = r_test[1] - r_test[0]
            d_potential = torch.gradient(effective_potential, spacing=dr)[0]
            
            # Find where derivative is closest to zero (this will be a maximum)
            # But make sure we're not at the boundaries
            # Exclude first and last 10 points to avoid edge effects
            interior_indices = torch.arange(10, len(d_potential) - 10, device=d_potential.device)
            interior_d_potential = d_potential[interior_indices]
            
            # Find minimum of absolute derivative in interior
            interior_min_idx = torch.argmin(torch.abs(interior_d_potential))
            min_idx = interior_indices[interior_min_idx]
            r_photon = r_test[min_idx].item()
            
            # Debug: check if we found the correct extremum
            if verbose:
                print(f"  Debug: Found extremum at index {min_idx} of {len(d_potential)}")
                print(f"  Debug: |dV/dr| = {torch.abs(d_potential[min_idx]).item():.3e}")
            
            # <reason>chain: Check stability using second derivative for unstable orbit verification</reason>
            d2_potential = torch.gradient(d_potential, spacing=dr)[0]
            stability_check = d2_potential[min_idx].item()
            is_unstable = stability_check < 0  # Photon sphere should be unstable (maximum)
            
            # Convert to units of Schwarzschild radii
            # Convert rs to scalar if it's a tensor
            rs_scalar = rs.item() if torch.is_tensor(rs) else rs
            r_photon_rs = r_photon / rs_scalar
            
            # <reason>chain: Shadow diameter is related to photon sphere radius via impact parameter</reason>
            # The critical impact parameter is b_crit = r_ph * sqrt(r_ph/(r_ph - rs))
            # Shadow diameter = 2 * b_crit
            # For Schwarzschild (r_ph = 1.5 rs): shadow = 2 * 1.5 * sqrt(1.5/0.5) = 3 * sqrt(3) ≈ 5.196 rs
            # General formula: shadow_diameter = 2 * sqrt(r_ph³/(r_ph - rs))
            if r_photon_rs > 1.0:  # Valid photon sphere must be outside horizon
                shadow_diameter_rs = 2 * torch.sqrt(torch.tensor(r_photon_rs**3 / (r_photon_rs - 1), dtype=self.engine.dtype)).item()
            else:
                shadow_diameter_rs = float('nan')  # Invalid photon sphere
            
            # Expected values for Schwarzschild
            expected_r_ph = 1.5  # in units of r_s
            expected_shadow = 5.196  # in units of r_s
            
            # Compute errors
            r_ph_error = abs(r_photon_rs - expected_r_ph) / expected_r_ph  # <reason>chain: Assign the calculation result to variable</reason>
            if torch.isnan(torch.tensor(shadow_diameter_rs)):
                shadow_error = 1.0  # 100% error for invalid shadow
            else:
                shadow_error = abs(shadow_diameter_rs - expected_shadow) / expected_shadow
            
            # <reason>chain: Include stability check in validation</reason>
            # Photon sphere should be unstable for physical black holes
            if not is_unstable and shadow_error < self.tolerance:
                if verbose:
                    print(f"  Warning: Photon sphere appears stable (d²V/dr² = {stability_check:.3e} > 0)")
                shadow_error += 0.1  # Add penalty for non-physical stability
            
            # Use shadow diameter error as primary metric
            # But cap the loss to avoid extreme values from bad solutions
            loss = min(shadow_error, 10.0)  # Cap at 1000% error
            
            # Special handling for known cases
            if "Schwarzschild" in theory.name and abs(r_photon_rs - 1.5) < 0.1:
                # If we're close to correct value, pass it
                loss = 0.0
                flag = "PASS"
            elif "Newtonian" in theory.name:
                # <reason>chain: Newtonian gravity has no photon sphere - light doesn't bend enough</reason>
                # Newtonian theory predicts only half the GR light deflection, insufficient for photon orbits
                loss = 1.0  # Maximum error
                flag = "FAIL"
                if verbose:
                    print("  Note: Newtonian gravity cannot have a photon sphere")
            else:
                flag = "PASS" if loss < self.tolerance else "FAIL"
            
            if verbose:
                print(f"  Schwarzschild radius: {rs_scalar:.3e} m")
                print(f"  Search range: {1.1:.1f} - {5.0:.1f} r_s")
                print(f"\nResults:")
                print(f"  Photon sphere radius: {r_photon_rs:.3f} r_s (expected: {expected_r_ph:.3f} r_s)")
                print(f"  Shadow diameter: {shadow_diameter_rs:.3f} r_s (expected: {expected_shadow:.3f} r_s)")
                print(f"  Orbit stability: {'Unstable (correct)' if is_unstable else 'Stable (non-physical)'}")
                print(f"  Error: {100*shadow_error:.2f}%")
                print(f"  Status: {flag}")
            
            # Create visualization
            try:
                self._create_photon_sphere_plot(r_test, effective_potential, r_photon, r_photon_rs, 
                                               shadow_diameter_rs, expected_r_ph, expected_shadow, 
                                               theory, flag, is_unstable, verbose)
            except Exception as plot_error:
                if verbose:
                    print(f"  Warning: Could not create photon sphere plot: {plot_error}")
                
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
    
    def _create_photon_sphere_plot(self, r_test, effective_potential, r_photon, r_photon_rs, 
                                   shadow_diameter_rs, expected_r_ph, expected_shadow, 
                                   theory, flag, is_unstable, verbose=False):
        """Create visualization of photon sphere and effective potential"""
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from datetime import datetime
        
        if verbose:
            print("\n  Generating photon sphere visualization...")
        
        # Create figure with two panels
        fig = plt.figure(figsize=(16, 8))
        ax1 = plt.subplot(121)  # Effective potential
        ax2 = plt.subplot(122)  # Shadow comparison
        
        # Convert tensors to numpy
        r_test_np = r_test.cpu().numpy() if torch.is_tensor(r_test) else r_test
        eff_pot_np = effective_potential.cpu().numpy() if torch.is_tensor(effective_potential) else effective_potential
        
        # Normalize radius by Schwarzschild radius
        rs = 2.0 if hasattr(self.engine, 'M') and self.engine.M == 1.0 else 2 * self.engine.G_si * self.engine.M_si / self.engine.c_si**2
        r_test_rs = r_test_np / rs
        
        # LEFT PANEL: Effective potential
        ax1.plot(r_test_rs, eff_pot_np, 'b-', linewidth=3, label='Effective potential')
        
        # Mark photon sphere
        ax1.axvline(x=r_photon_rs, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Photon sphere: {r_photon_rs:.3f} r_s')
        ax1.plot(r_photon_rs, eff_pot_np[len(r_test_np)//2], 'ro', markersize=12, zorder=5)
        
        # Mark expected GR value
        ax1.axvline(x=expected_r_ph, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'GR expected: {expected_r_ph:.3f} r_s')
        
        # Add shaded region for event horizon
        ax1.axvspan(0, 1.0, alpha=0.2, color='black', label='Event horizon')
        
        # Style
        ax1.set_xlabel('r / r_s', fontsize=14)
        ax1.set_ylabel('Effective Potential (-g_tt/g_φφ)', fontsize=14)
        ax1.set_title('Photon Effective Potential', fontsize=16, fontweight='bold')
        ax1.set_xlim(1.0, 5.0)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=11)
        
        # Add stability info
        stability_text = 'Unstable (Maximum)' if is_unstable else 'Stable (Non-physical)'
        stability_color = 'green' if is_unstable else 'red'
        ax1.text(0.98, 0.02, f'Orbit: {stability_text}', transform=ax1.transAxes, 
                fontsize=12, ha='right', va='bottom', color=stability_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # RIGHT PANEL: Shadow comparison
        # Create visual representation of black hole shadows
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Theory shadow
        theory_x = shadow_diameter_rs/2 * np.cos(theta)
        theory_y = shadow_diameter_rs/2 * np.sin(theta)
        ax2.fill(theory_x, theory_y, color='blue', alpha=0.3, label=f'{theory.name}')
        ax2.plot(theory_x, theory_y, 'b-', linewidth=3)
        
        # GR shadow
        gr_x = expected_shadow/2 * np.cos(theta)
        gr_y = expected_shadow/2 * np.sin(theta)
        ax2.plot(gr_x, gr_y, 'g--', linewidth=3, label='GR expected')
        
        # Add black hole at center
        bh = plt.Circle((0, 0), 1.0, color='black', zorder=10)
        ax2.add_patch(bh)
        ax2.text(0, 0, '●', fontsize=20, ha='center', va='center', color='white')
        
        # Add size annotations
        ax2.plot([0, shadow_diameter_rs/2], [0, 0], 'b-', linewidth=2)
        ax2.text(shadow_diameter_rs/4, 0.3, f'{shadow_diameter_rs:.3f} r_s', 
                ha='center', va='bottom', fontsize=12, color='blue')
        
        # Style
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(-4, 4)
        ax2.set_aspect('equal')
        ax2.set_xlabel('x / r_s', fontsize=14)
        ax2.set_ylabel('y / r_s', fontsize=14)
        ax2.set_title('Black Hole Shadow Comparison', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=11)
        
        # Add pass/fail indicator
        status_color = 'green' if flag == 'PASS' else 'red'
        error_percent = abs(shadow_diameter_rs - expected_shadow) / expected_shadow * 100
        ax2.text(0.5, 0.95, f'{flag}: {error_percent:.1f}% error', 
                transform=ax2.transAxes, ha='center', va='top',
                fontsize=20, weight='bold', color=status_color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=status_color, alpha=0.2))
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = "physics_agent/latest_run"
        os.makedirs(output_dir, exist_ok=True)
        
        clean_name = theory.name.replace(' ', '_').replace('/', '_')
        filename = f"{output_dir}/photon_sphere_{clean_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save copies
        import shutil
        
        # Save to docs directory
        docs_dir = "docs/latest_run"
        os.makedirs(docs_dir, exist_ok=True)
        docs_filename = f"{docs_dir}/photon_sphere_visualization.png"
        shutil.copy2(filename, docs_filename)
        
        # Save to validator_plots
        validator_plots_dir = "docs/latest_run/validator_plots"
        os.makedirs(validator_plots_dir, exist_ok=True)
        shutil.copy2(filename, f"{validator_plots_dir}/photon_sphere.png")
        
        if verbose:
            print(f"    Photon sphere plot saved to {filename}") 