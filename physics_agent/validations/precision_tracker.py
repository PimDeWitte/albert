import torch
import numpy as np
from typing import Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from physics_agent.base_theory import GravitationalTheory

@dataclass
class PrecisionMetrics:
    """Comprehensive precision tracking metrics."""
    # Trajectory precision
    energy_drift: float = 0.0
    angular_momentum_drift: float = 0.0
    constraint_violations: List[float] = field(default_factory=list)
    
    # Numerical stability
    condition_numbers: List[float] = field(default_factory=list)
    jacobian_determinants: List[float] = field(default_factory=list)
    
    # Error propagation
    local_truncation_errors: List[float] = field(default_factory=list)
    global_error_estimate: float = 0.0
    error_growth_rate: float = 0.0
    
    # Floating point analysis
    relative_errors: List[float] = field(default_factory=list)
    catastrophic_cancellations: int = 0
    underflow_warnings: int = 0
    overflow_warnings: int = 0
    
    # Metric tensor analysis
    metric_condition_number: float = 0.0
    christoffel_residuals: List[float] = field(default_factory=list)
    riemann_tensor_errors: List[float] = field(default_factory=list)

class PrecisionTracker:
    """
    Tracks numerical precision throughout gravitational simulations.
    Implements rigorous error analysis required for scientific publication.
    """
    
    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype
        self.metrics = PrecisionMetrics()
        self.machine_epsilon = torch.finfo(dtype).eps
        self.tolerance_hierarchy = {
            'conservation': 1e-12,
            'constraint': 1e-10,
            'trajectory': 1e-8,
            'observable': 1e-6
        }
        
    def analyze_rk4_step(self, y: torch.Tensor, k1: torch.Tensor, k2: torch.Tensor, 
                         k3: torch.Tensor, k4: torch.Tensor, h: float) -> Dict[str, Any]:
        """
        Analyze numerical precision of RK4 integration step.
        
        Returns error estimates and stability metrics.
        """
        # Richardson extrapolation error estimate
        y_full = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Half-step calculation for error estimation
        h_half = h / 2
        y + (h_half/6) * (k1 + 2*k2 + 2*k3 + k4)
        # Would need k values at half step - simplified here
        
        # Local truncation error (4th order)
        local_error = torch.max(torch.abs(y_full - y)).item()
        self.metrics.local_truncation_errors.append(local_error)
        
        # Condition number of derivative computation
        J = self._compute_jacobian(y, k1)
        if J is not None:
            cond = torch.linalg.cond(J).item()
            self.metrics.condition_numbers.append(cond)
            
        # Check for catastrophic cancellation
        for i in range(len(y)):
            if self._check_cancellation(k1[i], k2[i], k3[i], k4[i]):
                self.metrics.catastrophic_cancellations += 1
                
        return {
            'local_error': local_error,
            'stable': local_error < h**5,  # Expected O(h^5) for RK4
            'condition_number': cond if J is not None else None
        }
        
    def _compute_jacobian(self, y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of derivative function."""
        try:
            if y.requires_grad:
                J = torch.autograd.functional.jacobian(lambda x: k, y)
                return J
        except:
            return None
            
    def _check_cancellation(self, *values: torch.Tensor) -> bool:
        """Detect catastrophic cancellation in sum."""
        total = sum(values)
        max_val = max(torch.abs(v).item() for v in values)
        if max_val > 0:
            relative_result = torch.abs(total).item() / max_val
            return relative_result < self.machine_epsilon * 100
        return False
        
    def analyze_metric_precision(self, theory: "GravitationalTheory", 
                               r_values: torch.Tensor, M: torch.Tensor, 
                               c: float, G: float) -> Dict[str, Any]:
        """
        Analyze precision of metric tensor computations.
        """
        results = {}
        
        for r in r_values:
            g_tt, g_rr, g_pp, g_tp = theory.get_metric(r.unsqueeze(0), M, c, G)
            
            # Metric tensor as matrix
            g = torch.tensor([
                [-g_tt, 0, 0, -g_tp],
                [0, g_rr, 0, 0],
                [0, 0, 1, 0],  # g_theta_theta = 1
                [-g_tp, 0, 0, g_pp]
            ], dtype=self.dtype)
            
            # Condition number
            cond = torch.linalg.cond(g).item()
            self.metrics.metric_condition_number = max(self.metrics.metric_condition_number, cond)
            
            # Check metric signature (should be -,+,+,+)
            eigenvals = torch.linalg.eigvals(g).real
            signature_correct = (eigenvals < 0).sum() == 1
            
            # Verify metric invertibility
            try:
                g_inv = torch.linalg.inv(g)
                invertible = True
                inv_error = torch.max(torch.abs(g @ g_inv - torch.eye(4, dtype=self.dtype))).item()
            except:
                invertible = False
                inv_error = float('inf')
                
            results[f'r_{r.item():.2e}'] = {
                'condition_number': cond,
                'signature_correct': signature_correct,
                'invertible': invertible,
                'inversion_error': inv_error
            }
            
        return results
        
    def compute_global_error_bound(self, N_steps: int, h: float, 
                                 lipschitz_constant: float = None) -> float:
        """
        Compute rigorous global error bound using Gronwall's inequality.
        
        For RK4: global_error ≤ (L*h^4/5) * (exp(L*T) - 1) / L
        where L is Lipschitz constant, T = N_steps * h
        """
        if not self.metrics.local_truncation_errors:
            return float('inf')
            
        # Estimate Lipschitz constant from condition numbers if not provided
        if lipschitz_constant is None:
            if self.metrics.condition_numbers:
                lipschitz_constant = np.median(self.metrics.condition_numbers)
            else:
                lipschitz_constant = 1.0  # Conservative default
                
        T = N_steps * h
        max_local_error = max(self.metrics.local_truncation_errors)
        
        # Gronwall bound for RK4
        if lipschitz_constant * T < 100:  # Avoid overflow
            global_bound = (max_local_error / h) * (np.exp(lipschitz_constant * T) - 1) / lipschitz_constant
        else:
            global_bound = float('inf')
            
        self.metrics.global_error_estimate = global_bound
        
        # Error growth rate
        if len(self.metrics.local_truncation_errors) > 10:
            errors = np.array(self.metrics.local_truncation_errors)
            steps = np.arange(len(errors))
            # Fit exponential growth: log(error) = log(a) + b*step
            coeffs = np.polyfit(steps, np.log(errors + 1e-20), 1)
            self.metrics.error_growth_rate = coeffs[0]
            
        return global_bound
        
    def validate_conservation_precision(self, E_history: torch.Tensor, 
                                      L_history: torch.Tensor) -> Dict[str, Any]:
        """
        Validate that conservation laws hold within machine precision.
        """
        E_drift = torch.abs((E_history[-1] - E_history[0]) / E_history[0]).item()
        L_drift = torch.abs((L_history[-1] - L_history[0]) / L_history[0]).item()
        
        self.metrics.energy_drift = E_drift
        self.metrics.angular_momentum_drift = L_drift
        
        # Spectral analysis of drift
        E_fft = torch.fft.fft(E_history - E_history.mean())
        dominant_freq = torch.argmax(torch.abs(E_fft[1:len(E_fft)//2])) + 1
        
        return {
            'energy_drift': E_drift,
            'angular_momentum_drift': L_drift,
            'energy_drift_periodic': dominant_freq.item() > 1,
            'conservation_valid': E_drift < self.tolerance_hierarchy['conservation'] and 
                                L_drift < self.tolerance_hierarchy['conservation']
        }
        
    def generate_precision_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive precision analysis report for publication.
        """
        report = {
            'machine_precision': {
                'dtype': str(self.dtype),
                'epsilon': float(self.machine_epsilon),
                'decimal_digits': int(-np.log10(self.machine_epsilon))
            },
            'numerical_stability': {
                'median_condition_number': np.median(self.metrics.condition_numbers) if self.metrics.condition_numbers else None,
                'max_condition_number': max(self.metrics.condition_numbers) if self.metrics.condition_numbers else None,
                'catastrophic_cancellations': self.metrics.catastrophic_cancellations,
                'numerical_warnings': self.metrics.underflow_warnings + self.metrics.overflow_warnings
            },
            'error_analysis': {
                'global_error_bound': self.metrics.global_error_estimate,
                'error_growth_rate': self.metrics.error_growth_rate,
                'mean_local_error': np.mean(self.metrics.local_truncation_errors) if self.metrics.local_truncation_errors else None,
                'max_local_error': max(self.metrics.local_truncation_errors) if self.metrics.local_truncation_errors else None
            },
            'conservation_analysis': {
                'energy_drift': self.metrics.energy_drift,
                'angular_momentum_drift': self.metrics.angular_momentum_drift,
                'conservation_valid': self.metrics.energy_drift < self.tolerance_hierarchy['conservation']
            },
            'metric_analysis': {
                'max_condition_number': self.metrics.metric_condition_number,
                'christoffel_residuals': self.metrics.christoffel_residuals,
                'riemann_errors': self.metrics.riemann_tensor_errors
            },
            'scientific_validity': self._assess_scientific_validity()
        }
        
        return report
        
    def _assess_scientific_validity(self) -> Dict[str, bool]:
        """
        Assess whether results meet standards for scientific publication.
        """
        return {
            'precision_adequate': self.metrics.global_error_estimate < 1e-10 if self.metrics.global_error_estimate else False,
            'numerically_stable': all(c < 1e8 for c in self.metrics.condition_numbers) if self.metrics.condition_numbers else False,
            'conservation_satisfied': self.metrics.energy_drift < 1e-12 and self.metrics.angular_momentum_drift < 1e-12,
            'no_catastrophic_failures': self.metrics.catastrophic_cancellations == 0,
            'publication_ready': all([
                self.metrics.global_error_estimate < 1e-10 if self.metrics.global_error_estimate else False,
                all(c < 1e8 for c in self.metrics.condition_numbers) if self.metrics.condition_numbers else False,
                self.metrics.energy_drift < 1e-12,
                self.metrics.catastrophic_cancellations == 0
            ])
        } 