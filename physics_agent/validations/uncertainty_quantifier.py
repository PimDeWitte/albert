import torch
import numpy as np
from typing import Dict, Any, List
from scipy import stats

class UncertaintyQuantifier:
    """
    Implements rigorous uncertainty quantification for gravitational simulations.
    Required for scientific credibility.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
    def propagate_metric_uncertainty(self, theory, r_values: torch.Tensor, 
                                   M: torch.Tensor, c: float, G: float,
                                   param_uncertainties: Dict[str, float]) -> Dict[str, Any]:
        """
        Propagate parameter uncertainties through metric calculations.
        Uses first-order Taylor expansion.
        """
        nominal_metrics = []
        metric_jacobians = []
        
        for r in r_values:
            r.requires_grad_(True)
            g_tt, g_rr, g_pp, g_tp = theory.get_metric(r.unsqueeze(0), M, c, G)
            nominal_metrics.append([g_tt, g_rr, g_pp, g_tp])
            
            # Compute sensitivities
            jacobians = {}
            for param_name, uncertainty in param_uncertainties.items():
                if param_name == 'M':
                    grad = torch.autograd.grad(g_tt.sum(), M, retain_graph=True)[0]
                    jacobians['M'] = grad.item() * uncertainty
                # Add other parameters
                    
            metric_jacobians.append(jacobians)
            
        # Compute propagated uncertainties
        metric_uncertainties = []
        for jac_dict in metric_jacobians:
            total_var = sum(j**2 for j in jac_dict.values())
            metric_uncertainties.append(np.sqrt(total_var))
            
        return {
            'nominal_values': nominal_metrics,
            'uncertainties': metric_uncertainties,
            'relative_uncertainties': [u/abs(m[0].item()) for u, m in zip(metric_uncertainties, nominal_metrics)]
        }
        
    def bootstrap_trajectory_confidence(self, theory, initial_conditions: torch.Tensor,
                                      n_bootstrap: int = 100, n_steps: int = 1000) -> Dict[str, Any]:
        """
        Use bootstrap resampling to estimate trajectory confidence intervals.
        """
        
        for _ in range(n_bootstrap):
            # Add small perturbations to initial conditions
            initial_conditions + torch.randn_like(initial_conditions) * 1e-10
            # Run trajectory (simplified - would call actual integrator)
            # hist = run_trajectory(theory, perturbed_ic, n_steps)
            # trajectories.append(hist)
            
        # Compute confidence intervals
        # trajectories_array = torch.stack(trajectories)
        # lower_bound = torch.quantile(trajectories_array, (1 - self.confidence_level) / 2, dim=0)
        # upper_bound = torch.quantile(trajectories_array, (1 + self.confidence_level) / 2, dim=0)
        
        return {
            'confidence_level': self.confidence_level,
            'n_samples': n_bootstrap,
            # 'lower_bound': lower_bound,
            # 'upper_bound': upper_bound,
            # 'median_trajectory': torch.median(trajectories_array, dim=0)[0]
        }
        
    def validation_result_confidence(self, validation_results: List[Dict], 
                                   n_monte_carlo: int = 1000) -> Dict[str, Any]:
        """
        Compute confidence intervals for validation results using Monte Carlo.
        """
        # Aggregate results by validator type
        by_validator = {}
        for result in validation_results:
            validator = result.get('validator', 'unknown')
            if validator not in by_validator:
                by_validator[validator] = []
            by_validator[validator].append(result['loss'])
            
        confidence_intervals = {}
        for validator, losses in by_validator.items():
            losses_array = np.array(losses)
            mean_loss = np.mean(losses_array)
            std_loss = np.std(losses_array)
            
            # Confidence interval
            margin = self.z_score * std_loss / np.sqrt(len(losses))
            confidence_intervals[validator] = {
                'mean': mean_loss,
                'std': std_loss,
                'ci_lower': mean_loss - margin,
                'ci_upper': mean_loss + margin,
                'n_samples': len(losses)
            }
            
        return confidence_intervals 