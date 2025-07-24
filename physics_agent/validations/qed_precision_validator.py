#!/usr/bin/env python3
"""
QED Precision Tests Validator

Tests quantum theories against ultra-precise QED measurements:
- Electron anomalous magnetic moment (g-2)
- Lamb shift in hydrogen
- Vacuum polarization effects

<reason>chain: QED provides the most precise tests of quantum field theory</reason>
<reason>chain: These tests can reveal tiny deviations from standard physics</reason>
"""

import torch
import numpy as np
from typing import Dict, Any, TYPE_CHECKING

from physics_agent.validations.base_validation import PredictionValidator
from physics_agent.quantum_path_integrator import QuantumPathIntegrator, QuantumLossCalculator

if TYPE_CHECKING:
    from physics_agent.base_theory import GravitationalTheory
    from physics_agent.theory_engine_core import TheoryEngine


class QEDPrecisionValidator(PredictionValidator):
    """
    <reason>chain: Validates theories against precision QED measurements</reason>
    
    Tests include:
    - Electron g-2 to 13 significant figures
    - Lamb shift in hydrogen
    - Gravitational corrections to QED
    """
    
    validator_name = "qed_precision"
    
    def __init__(self, engine: "TheoryEngine"):
        super().__init__(engine)
        self.tolerance_g2 = 1e-10  # Relative tolerance for g-2
        self.tolerance_lamb = 1e-6  # Relative tolerance for Lamb shift
        
    def validate(self, theory: "GravitationalTheory", hist: torch.Tensor = None, 
                experimental: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Run QED precision tests.
        """
        # Skip non-quantum theories
        if not hasattr(theory, 'category') or theory.category != 'quantum':
            return {
                "loss": 0.0,
                "flags": {"overall": "SKIP", "details": "Not a quantum theory"}
            }
            
        # Ensure theory has quantum integrator
        if not hasattr(theory, 'quantum_integrator') or theory.quantum_integrator is None:
            theory.quantum_integrator = QuantumPathIntegrator(theory, enable_quantum=True)
            
        # Create loss calculator
        loss_calc = QuantumLossCalculator(theory.quantum_integrator)
        
        # Theory parameters
        theory_params = {
            'alpha': 1.0/137.035999084,  # Fine structure constant
            'm_e': 9.109e-31,  # Electron mass
            'e': 1.602e-19,    # Elementary charge
            'hbar': 1.055e-34, # Reduced Planck constant
            'M': self.engine.M.item() if torch.is_tensor(self.engine.M) else self.engine.M,
            'c': self.engine.c_si,
            'G': self.engine.G_si,
        }
        
        # Add theory-specific parameters
        for param in ['alpha', 'beta', 'gamma', 'kappa', 'omega', 'sigma']:
            if hasattr(theory, param):
                value = getattr(theory, param)
                # Convert tensor to float
                if torch.is_tensor(value):
                    theory_params[param] = value.item()
                # Check if it's a SymPy symbol/expression
                elif hasattr(value, 'is_symbol') or hasattr(value, 'is_Symbol'):
                    # Skip symbolic parameters - they shouldn't affect QED
                    continue
                else:
                    theory_params[param] = value
                
        # Test 1: g-2 in flat spacetime
        print("  Testing electron g-2...")
        g2_loss_flat = loss_calc.compute_qed_g2_loss(theory_params, gravitational_field_strength=0.0)
        
        # Test 2: g-2 in weak gravitational field (Earth surface)
        # g = GM/r² = 9.8 m/s², so GM/(rc²) ≈ 1e-9
        g2_loss_earth = loss_calc.compute_qed_g2_loss(theory_params, gravitational_field_strength=1e-9)
        
        # Test 3: g-2 near neutron star (strong field)
        # For r = 3rs, GM/(rc²) ≈ 0.17
        g2_loss_strong = loss_calc.compute_qed_g2_loss(theory_params, gravitational_field_strength=0.17)
        
        # Test 4: Lamb shift in hydrogen
        print("  Testing Lamb shift...")
        lamb_loss_flat = loss_calc.compute_qed_lamb_shift_loss(theory_params, near_horizon=False)
        
        # Test 5: Lamb shift near black hole horizon
        r_test = 3.0 * (2 * theory_params['G'] * theory_params['M'] / theory_params['c']**2)  # r = 3rs
        lamb_loss_horizon = loss_calc.compute_qed_lamb_shift_loss(
            theory_params, near_horizon=True, r_distance=r_test
        )
        
        # Combine losses
        total_loss = (
            0.4 * g2_loss_flat +      # Most weight on precision flat space test
            0.2 * g2_loss_earth +     # Some weight on Earth gravity
            0.1 * g2_loss_strong +    # Less weight on extreme gravity
            0.2 * lamb_loss_flat +    # Good weight on Lamb shift
            0.1 * lamb_loss_horizon   # Some weight on extreme Lamb shift
        )
        
        # Determine flags
        g2_flag = "PASS" if g2_loss_flat < 10.0 else "FAIL"  # Chi² < 10 is good
        lamb_flag = "PASS" if lamb_loss_flat < 10.0 else "FAIL"
        gravity_flag = "PASS" if g2_loss_strong < 100.0 else "FAIL"  # More tolerance for strong field
        
        overall_flag = "PASS" if all(f == "PASS" for f in [g2_flag, lamb_flag]) else "FAIL"
        
        # <reason>chain: Add quantum path visualization if requested</reason>
        quantum_paths = None
        if experimental and hist is not None and len(hist) > 1:
            # Visualize quantum paths around classical trajectory
            start = (hist[0, 0].item(), hist[0, 1].item(), np.pi/2, hist[0, 2].item())
            end = (hist[-1, 0].item(), hist[-1, 1].item(), np.pi/2, hist[-1, 2].item())
            
            from physics_agent.quantum_path_integrator import visualize_quantum_paths
            quantum_paths = visualize_quantum_paths(
                theory.quantum_integrator, start, end, num_paths=5
            )
            
        results = {
            "loss": float(total_loss),
            "flags": {
                "overall": overall_flag,
                "g2_flat": g2_flag,
                "g2_earth": f"χ²={g2_loss_earth:.2f}",
                "g2_strong": gravity_flag,
                "lamb_shift": lamb_flag,
                "lamb_horizon": f"χ²={lamb_loss_horizon:.2f}"
            },
            "details": {
                "g2_chi2_flat": float(g2_loss_flat),
                "g2_chi2_earth": float(g2_loss_earth),
                "g2_chi2_strong": float(g2_loss_strong),
                "lamb_chi2_flat": float(lamb_loss_flat),
                "lamb_chi2_horizon": float(lamb_loss_horizon),
                "total_loss": float(total_loss)
            }
        }
        
        if quantum_paths is not None:
            results["quantum_paths"] = quantum_paths
            
        return results
        
    def compute_loss(self, theory: "GravitationalTheory", observations: Dict = None) -> float:
        """Compute loss for optimization"""
        result = self.validate(theory)
        return result.get("loss", float('inf'))
        
    def fetch_dataset(self) -> Dict:
        """Return QED precision measurements"""
        return {
            'g2_electron': {
                'value': 0.00115965218059,
                'error': 0.00000000000013,
                'source': 'Harvard 2023'
            },
            'lamb_shift': {
                'value': 1057.845,  # MHz
                'error': 0.009,
                'source': 'CODATA 2018'
            }
        }
        
    def get_observational_data(self) -> Dict:
        """Return observational data for QED tests"""
        return self.fetch_dataset()
        
    def get_sota_benchmark(self) -> Dict:
        """Return state-of-the-art QED predictions"""
        return {
            'g2_theory': 0.00115965218178,  # 5-loop QED + hadronic
            'lamb_theory': 1057.839,  # MHz, full QED calculation
            'precision': '12 significant figures'
        } 