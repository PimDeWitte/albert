from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Dict, Any

from .base_validation import BaseValidation

if TYPE_CHECKING:
    from physics_agent.base_theory import GravitationalTheory
    from physics_agent.theory_engine_core import TheoryEngine

class MetricPropertiesValidator(BaseValidation):
    """
    Validates fundamental properties of a theory's metric tensor, such as
    Lorentzian signature and asymptotic flatness.
    """
    category = "constraint"
    
    def __init__(self, engine: "TheoryEngine", tolerance: float = 1e-3):
        super().__init__(engine, "Metric Properties Validator")
        self.tolerance = tolerance

    def validate(self, theory: "GravitationalTheory", hist: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Computes losses based on metric properties at sampled points from the history.
        """
        if hist is None or hist.shape[0] < 1:
            return {"loss": 1.0, "flags": {"overall": "FAIL", "details": "History is too short or None."}}

        # Sample radii from the history
        num_samples = min(10, hist.shape[0])
        sample_indices = torch.linspace(0, hist.shape[0] - 1, num_samples, dtype=torch.long)
        sampled_radii = hist[sample_indices, 1]

        # --- Lorentzian Signature Check ---
        g_tt, g_rr, g_pp, _ = theory.get_metric(sampled_radii, self.engine.M_si, self.engine.c_si, self.engine.G_si)
        
        signature_loss = torch.mean(
            torch.relu(g_tt) +      # Should be < 0
            torch.relu(-g_rr) +     # Should be > 0
            torch.relu(-g_pp)       # Should be > 0
        ).item()
        
        sig_flag = "PASS" if signature_loss < self.tolerance else "FAIL"

        # --- Asymptotic Flatness Check ---
        large_r = self.engine.RS * 1e6
        g_tt_flat, g_rr_flat, _, g_tp_flat = theory.get_metric(large_r, self.engine.M_si, self.engine.c_si, self.engine.G_si)
        
        flatness_loss = (
            torch.abs(g_tt_flat + 1.0) +
            torch.abs(g_rr_flat - 1.0) +
            torch.abs(g_tp_flat)
        ).item()
        
        flat_flag = "PASS" if flatness_loss < self.tolerance else "FAIL"

        # --- Total Loss and Flags ---
        total_loss = signature_loss + flatness_loss
        overall_flag = "PASS" if (sig_flag == "PASS" and flat_flag == "PASS") else "FAIL"

        return {
            "loss": total_loss,
            "flags": {
                "overall": overall_flag,
                "lorentzian_signature": sig_flag,
                "asymptotic_flatness": flat_flag
            },
            "details": {
                "signature_loss": signature_loss,
                "flatness_loss": flatness_loss
            }
        } 