import torch
import numpy as np
from typing import Dict, Any, TYPE_CHECKING
from ..base_validation import BaseValidation
from physics_agent.constants import EULER_GAMMA  # <reason>chain: Import Euler's constant from centralized constants module</reason>

if TYPE_CHECKING:
    from physics_agent.base_theory import GravitationalTheory
    from physics_agent.theory_engine_core import TheoryEngine

class GwValidator(BaseValidation):
    category = "observational"
    
    def __init__(self, engine: "TheoryEngine", tolerance: float = 0.05):
        super().__init__(engine, "Gravitational Wave Validator")
        self.tolerance = tolerance

    def validate(self, theory: "GravitationalTheory", hist: torch.Tensor = None, **kwargs) -> Dict[str, Any]:
        verbose = kwargs.get('verbose', False)
        
        if verbose:
            print(f"\nGenerating gravitational waveforms for {theory.name}...")
        
        try:
            # <reason>chain: Generate full PN waveform including spin-orbit effects</reason>
            # Use 3.5PN phase evolution for accurate LIGO/Virgo comparison
            
            # Import required constants from centralized module
            from physics_agent.constants import (
                GRAVITATIONAL_CONSTANT as G, SPEED_OF_LIGHT as c,
                SOLAR_MASS as M_sun
            )
            
            # Binary parameters (GW150914-like)
            m1 = 36 * M_sun  # Primary mass
            m2 = 29 * M_sun  # Secondary mass
            M_total = m1 + m2
            mu = m1 * m2 / M_total  # Reduced mass
            eta = mu / M_total  # Symmetric mass ratio
            
            # Spin parameters (dimensionless)
            chi1 = 0.3  # Primary spin
            chi2 = -0.1  # Secondary spin
            chi_eff = (m1 * chi1 + m2 * chi2) / M_total  # Effective spin
            
            # Time array (last 0.2 seconds before merger)
            fs = 4096  # Sampling frequency (Hz)
            t = torch.linspace(-0.2, 0, int(0.2 * fs), device=self.engine.device, dtype=self.engine.dtype)
            
            # <reason>chain: Compute PN orbital phase evolution</reason>
            # Φ(f) = Φ_N + Σ Φ_k (πMf)^(k/3) where k goes from 0 to 7 (3.5PN)
            
            # Frequency evolution using PN approximation
            # f(t) = f_0 / (1 - 256/5 * η * (πMf_0)^(8/3) * t)^(3/8)
            f_0 = 30.0  # Initial frequency (Hz)
            M_sec = G * M_total / c**3  # Total mass in seconds
            
            # PN frequency evolution
            tau = -t  # Time to coalescence
            f_gw = f_0 * (1 + (256/5) * eta * (np.pi * M_sec * f_0)**(8/3) * tau)**(3/8)
            
            # <reason>chain: Include PN amplitude corrections</reason>
            # h(t) = A(f) * exp(iΦ(f)) with PN amplitude corrections
            
            # Leading order amplitude
            D_L = 410e6 * 3.086e16  # Luminosity distance (410 Mpc in meters)
            A_0 = (4 * mu * (G * M_total)**(5/3)) / (D_L * c**4)
            
            # PN phase accumulation
            v = (np.pi * M_sec * f_gw)**(1/3)  # PN expansion parameter
            
            # Phase corrections up to 3.5PN
            phi_0 = 0  # Initial phase
            phi_1 = 0  # 0.5PN - vanishes
            phi_2 = 3715/756 + 55*eta/9  # 1PN
            phi_3 = -16*np.pi + 113*chi_eff/3  # 1.5PN with spin
            phi_4 = 15293365/508032 + 27145*eta/504 + 3085*eta**2/72  # 2PN
            phi_5 = np.pi*(38645/756 - 65*eta/9) * (1 + 3*torch.log(v))  # 2.5PN, <reason>chain: Use torch.log for tensor operations</reason>
            phi_6 = 11583231236531/4694215680 - 640*np.pi**2/3 - 6848*EULER_GAMMA/21  # 3PN
            phi_7 = np.pi*(77096675/254016 + 378515*eta/1512 - 74045*eta**2/756)  # 3.5PN
            
            # Total phase
            phase = 2 * np.pi * torch.cumsum(f_gw * (1/fs), dim=0)
            for k, phi_k in enumerate([phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7]):
                # <reason>chain: Handle both scalar and tensor phi_k values</reason>
                if torch.is_tensor(phi_k):
                    if torch.any(phi_k != 0):
                        phase += phi_k * v**(k)
                elif phi_k != 0:
                    phase += phi_k * v**(k)
                    
            # <reason>chain: Apply theory-specific modifications</reason>
            # Modified dispersion relation or propagation effects
            theory_has_modifications = False
            
            if hasattr(theory, 'gw_speed'):
                v_gw = theory.gw_speed(f_gw)  # Frequency-dependent speed
                phase *= v_gw / c
                theory_has_modifications = True
            
            if hasattr(theory, 'gw_damping'):
                A_0 *= torch.exp(-theory.gw_damping * D_L / c)
                theory_has_modifications = True
                
            # <reason>chain: Check for gw_modifications method for comprehensive modifications</reason>
            if hasattr(theory, 'gw_modifications'):
                mods = theory.gw_modifications(f_gw, eta, chi_eff)
                if 'phase_correction' in mods:
                    phase += mods['phase_correction']
                    theory_has_modifications = True
                if 'amplitude_correction' in mods:
                    A_0 *= mods['amplitude_correction']
                    theory_has_modifications = True
                    
            # <reason>chain: Continue with GR waveform if no modifications</reason>
            if not theory_has_modifications:
                if verbose:
                    print(f"  Note: Theory has no GW modifications - computing standard GR waveform")
            
            # Generate plus and cross polarizations
            # <reason>chain: Ensure A_0 is a tensor for operations</reason>
            if not torch.is_tensor(A_0):
                A_0 = torch.tensor(A_0, device=self.engine.device, dtype=self.engine.dtype)
            h_plus = A_0 * (f_gw / f_0)**(-7/6) * torch.cos(phase)
            h_cross = A_0 * (f_gw / f_0)**(-7/6) * torch.sin(phase)
            
            # Detector response (simplified - assumes optimal orientation)
            h_theory = 0.5 * h_plus + 0.5 * h_cross
            
            # <reason>chain: Generate GR reference waveform for comparison</reason>
            # Pure GR waveform with same parameters but no theory modifications
            phase_gr = 2 * np.pi * torch.cumsum(f_gw * (1/fs), dim=0)
            # Apply GR phase corrections
            # <reason>chain: Use same phi coefficients including spin for proper comparison</reason>
            for k, phi_k in enumerate([phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7]):
                # <reason>chain: Handle both scalar and tensor phi_k values</reason>
                if torch.is_tensor(phi_k):
                    if torch.any(phi_k != 0):
                        phase_gr += phi_k * v**(k)
                elif phi_k != 0:
                    phase_gr += phi_k * v**(k)
            
            # <reason>chain: Match the detector response for proper comparison</reason>
            h_plus_gr = A_0 * (f_gw / f_0)**(-7/6) * torch.cos(phase_gr)
            h_cross_gr = A_0 * (f_gw / f_0)**(-7/6) * torch.sin(phase_gr)
            h_gr = 0.5 * h_plus_gr + 0.5 * h_cross_gr
            
            # <reason>chain: Add realistic detector noise to GR baseline for better calibration</reason>
            # LIGO design sensitivity at 100 Hz: ~1e-23 strain/rtHz
            noise_amplitude = 1e-23 * torch.sqrt(torch.tensor(fs / 2, device=self.engine.device, dtype=self.engine.dtype))
            noise = noise_amplitude * torch.randn_like(h_gr)
            h_gr_noisy = h_gr + noise
            h_theory_noisy = h_theory + noise
            
            # <reason>chain: Compute match via normalized inner product</reason>
            # Match = max <h1|h2> / sqrt(<h1|h1><h2|h2>)
            inner_gr_gr = torch.sum(h_gr_noisy * h_gr_noisy)
            inner_theory_theory = torch.sum(h_theory_noisy * h_theory_noisy)
            inner_gr_theory = torch.sum(h_gr_noisy * h_theory_noisy)
            
            match = (inner_gr_theory / torch.sqrt(inner_gr_gr * inner_theory_theory)).item()
            
            # Phase difference at ISCO (f = 1/(6^1.5 × 2π × GM/c³))
            f_isco = c**3 / (6**1.5 * 2 * np.pi * G * M_total)
            # Handle both tensor and float types
            if torch.is_tensor(f_gw):
                isco_idx = torch.argmin(torch.abs(f_gw - f_isco))
                phase_diff_tensor = (phase[isco_idx] - phase_gr[isco_idx]) % (2 * np.pi)
                phase_diff = abs(phase_diff_tensor.item() if torch.is_tensor(phase_diff_tensor) else phase_diff_tensor)
                f_gw_initial = f_gw[0].item() if torch.is_tensor(f_gw[0]) else f_gw[0]
            else:
                isco_idx = np.argmin(np.abs(f_gw - f_isco))
                phase_diff = abs((phase[isco_idx] - phase_gr[isco_idx]) % (2 * np.pi))
                f_gw_initial = f_gw[0]
            
            # Default delta value (frequency deviation)
            delta = 0.0
            
            loss = 1 - match  # Loss is 1 - correlation
            
            # <reason>chain: Set appropriate status based on match and modifications</reason>
            if match > 0.97:
                flag = "PASS"
            elif match > 0.95:
                flag = "WARNING"
            else:
                flag = "FAIL"
                
            # <reason>chain: For theories without modifications, perfect match is expected</reason>
            if not theory_has_modifications and match > 0.99:
                flag = "PASS"
                notes_suffix = " (matches GR as expected)"
            else:
                notes_suffix = ""
            
            if verbose:
                print(f"  Binary masses: {m1/M_sun:.1f} + {m2/M_sun:.1f} M_sun")
                print(f"  Effective spin χ_eff: {chi_eff:.2f}")
                print(f"  Initial frequency: {f_gw_initial:.1f} Hz")
                # Handle f_isco being either float or numpy scalar
                f_isco_val = f_isco.item() if hasattr(f_isco, 'item') else float(f_isco)
                print(f"  ISCO frequency: {f_isco_val:.1f} Hz")
                print(f"\nResults:")
                print(f"  Waveform match: {match:.3f}")
                print(f"  Phase difference at ISCO: {phase_diff:.3f} rad")
                print(f"  PN order: 3.5PN")
                print(f"  Status: {flag}")
            
            # Create waveform visualization
            try:
                self._create_waveform_plot(t, h_theory_noisy, h_gr_noisy, match, phase_diff, 
                                         theory, m1, m2, chi_eff, flag, verbose)
            except Exception as plot_error:
                if verbose:
                    print(f"  Warning: Could not create waveform plot: {plot_error}")
                
        except Exception as e:
            if verbose:
                print(f"  Error generating waveforms: {str(e)}")
            return {
                "loss": 1.0,
                "flags": {"overall": "FAIL"},
                "details": {
                    "correlation": 0.0,
                    "error": str(e),
                    "notes": "Failed to generate waveforms"
                }
            }
        
        return {
            "loss": loss,
            "flags": {"overall": flag},
            "details": {
                "correlation": match,
                "phase_difference": phase_diff,
                "frequency_deviation": delta,
                "units": "dimensionless",
                "notes": f"Waveform match: {match:.3f} (>0.95 required){notes_suffix}"
            }
        }
    
    def _create_waveform_plot(self, t, h_theory, h_gr, match, phase_diff, theory, m1, m2, chi_eff, flag, verbose=False):
        """Create a visualization of gravitational waveforms"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import os
        from datetime import datetime
        
        if verbose:
            print("\n  Generating gravitational waveform visualization...")
        
        # Create figure with two panels
        fig = plt.figure(figsize=(16, 10))
        
        # Convert tensors to numpy for plotting
        t_np = t.cpu().numpy() if torch.is_tensor(t) else t
        h_theory_np = h_theory.cpu().numpy() if torch.is_tensor(h_theory) else h_theory
        h_gr_np = h_gr.cpu().numpy() if torch.is_tensor(h_gr) else h_gr
        
        # Top panel: Full waveform comparison
        ax1 = plt.subplot(211)
        
        # Plot waveforms
        ax1.plot(t_np * 1000, h_gr_np * 1e21, 'g-', linewidth=2, alpha=0.8, label='GR (reference)')
        ax1.plot(t_np * 1000, h_theory_np * 1e21, 'b--', linewidth=2, alpha=0.8, label=f'{theory.name}')
        
        # Add merger time marker
        ax1.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Merger')
        
        # Styling
        ax1.set_xlabel('Time from merger (ms)', fontsize=14)
        ax1.set_ylabel('Strain (×10⁻²¹)', fontsize=14)
        ax1.set_title('Gravitational Waveform Comparison', fontsize=16, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-200, 5)
        
        # Add binary info box
        from physics_agent.constants import SOLAR_MASS as M_sun
        binary_info = (f'Binary masses: {m1/M_sun:.1f} + {m2/M_sun:.1f} M☉\n'
                      f'Effective spin: χ_eff = {chi_eff:.2f}\n'
                      f'Distance: 410 Mpc\n'
                      f'PN order: 3.5PN')
        ax1.text(0.02, 0.98, binary_info, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        # Bottom panel: Zoomed-in comparison and match statistics
        ax2 = plt.subplot(223)
        
        # Zoom to last 10 ms
        zoom_mask = t_np > -0.01
        t_zoom = t_np[zoom_mask] * 1000
        h_gr_zoom = h_gr_np[zoom_mask] * 1e21
        h_theory_zoom = h_theory_np[zoom_mask] * 1e21
        
        ax2.plot(t_zoom, h_gr_zoom, 'g-', linewidth=2, alpha=0.8, label='GR')
        ax2.plot(t_zoom, h_theory_zoom, 'b--', linewidth=2, alpha=0.8, label=theory.name)
        
        # Highlight differences
        diff = h_theory_zoom - h_gr_zoom
        ax2.fill_between(t_zoom, h_gr_zoom, h_theory_zoom, 
                        where=abs(diff) > 0.1 * abs(h_gr_zoom).max(),
                        color='red', alpha=0.2, label='Significant difference')
        
        ax2.set_xlabel('Time from merger (ms)', fontsize=14)
        ax2.set_ylabel('Strain (×10⁻²¹)', fontsize=14)
        ax2.set_title('Last 10ms (Zoomed)', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-10, 1)
        
        # Match statistics panel
        ax3 = plt.subplot(224)
        
        # Create match visualization
        match_percent = match * 100
        
        # Draw match meter
        meter_bg = patches.Rectangle((0, 0), 100, 20, facecolor='lightgray', edgecolor='black', linewidth=2)
        ax3.add_patch(meter_bg)
        
        # Color based on match value
        if match > 0.97:
            meter_color = 'green'
            status_text = '✓ PASS'
            status_color = 'green'
        elif match > 0.95:
            meter_color = 'orange'
            status_text = '⚠ WARNING'
            status_color = 'orange'
        else:
            meter_color = 'red'
            status_text = '✗ FAIL'
            status_color = 'red'
        
        meter_fill = patches.Rectangle((0, 0), match_percent, 20, facecolor=meter_color, alpha=0.7)
        ax3.add_patch(meter_fill)
        
        # Add percentage text
        ax3.text(50, 10, f'{match_percent:.1f}%', ha='center', va='center', 
                fontsize=16, weight='bold', color='black', zorder=10)
        
        # Status text
        ax3.text(50, 35, status_text, ha='center', va='center',
                fontsize=24, weight='bold', color=status_color)
        
        # Requirements
        ax3.text(50, -10, 'Requirement: >95% match', ha='center', va='center',
                fontsize=12, style='italic')
        
        # Phase difference info
        phase_diff_deg = phase_diff * 180 / np.pi
        ax3.text(50, -20, f'Phase diff at ISCO: {phase_diff_deg:.1f}°', 
                ha='center', va='center', fontsize=11)
        
        # Remove axes
        ax3.set_xlim(-10, 110)
        ax3.set_ylim(-30, 50)
        ax3.axis('off')
        ax3.set_title('Waveform Match', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = "physics_agent/latest_run"
        os.makedirs(output_dir, exist_ok=True)
        
        clean_name = theory.name.replace(' ', '_').replace('/', '_')
        filename = f"{output_dir}/gw_waveform_{clean_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save copies to various directories
        import shutil
        
        # Save to docs directory
        docs_dir = "docs/latest_run"
        os.makedirs(docs_dir, exist_ok=True)
        docs_filename = f"{docs_dir}/gw_waveform_visualization.png"
        shutil.copy2(filename, docs_filename)
        
        # Save to validator_plots
        validator_plots_dir = "docs/latest_run/validator_plots"
        os.makedirs(validator_plots_dir, exist_ok=True)
        shutil.copy2(filename, f"{validator_plots_dir}/gw_waveform.png")
        
        if verbose:
            print(f"    Gravitational waveform plot saved to {filename}") 