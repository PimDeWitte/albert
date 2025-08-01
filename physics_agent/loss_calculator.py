#!/usr/bin/env python3
"""
Loss Calculator Module - Handles all loss computation for trajectory comparison
<reason>chain: Separating loss calculation logic improves modularity and enables easy addition of new loss types</reason>
"""
import torch
import numpy as np
from typing import Dict, Optional, Union
from torch import Tensor


class LossCalculator:
    """
    Calculates various loss metrics between trajectories and theories.
    <reason>chain: Encapsulating loss logic enables better testing and extension</reason>
    """
    
    def __init__(self, device: str = 'cpu', dtype: torch.dtype = torch.float64):
        """
        Initialize loss calculator.
        
        Args:
            device: PyTorch device
            dtype: PyTorch data type
        """
        self.device = device
        self.dtype = dtype
        # <reason>chain: Event horizon at r = 2M in geometric units</reason>
        self.event_horizon_geometric = 2.0
        
    def _truncate_at_horizon(self, hist: Tensor, length_scale: float) -> Tensor:
        """
        Truncate trajectory at event horizon for observable region only.
        <reason>chain: Only the portion outside the event horizon is physically observable</reason>
        
        Args:
            hist: Trajectory tensor with columns [t, r, phi, ...]
            length_scale: Conversion factor from SI to geometric units
            
        Returns:
            Truncated trajectory up to event horizon crossing
        """
        # Convert radial coordinate to geometric units
        r_geometric = hist[:, 1] / length_scale
        
        # Find where trajectory crosses event horizon
        outside_horizon = r_geometric > self.event_horizon_geometric
        
        if torch.all(outside_horizon):
            # Entire trajectory is outside horizon
            return hist
        elif torch.all(~outside_horizon):
            # Entire trajectory is inside horizon (shouldn't happen)
            return hist[:1]  # Return just initial point
        else:
            # Find last point outside horizon
            last_outside_idx = torch.where(outside_horizon)[0][-1].item()
            # Include one point past horizon for interpolation if needed
            return hist[:last_outside_idx + 2]
        
    def compute_ricci_loss(self, 
                          model1, 
                          model2, 
                          r_samples: Tensor, 
                          M: Tensor, 
                          c: float, 
                          G: float,
                          epsilon: float = 1e-8) -> float:
        """
        Compute Ricci tensor loss between two models.
        <reason>chain: Ricci loss measures fundamental differences in spacetime curvature</reason>
        
        Args:
            model1: First gravitational theory
            model2: Second gravitational theory  
            r_samples: Radial points to sample
            M: Mass parameter
            c: Speed of light
            G: Gravitational constant
            epsilon: Small value for numerical stability
            
        Returns:
            Average Ricci tensor difference across samples
        """
        total_loss = 0.0
        valid_samples = 0
        
        for r in r_samples:
            try:
                # <reason>chain: Get Ricci tensors from both models</reason>
                ricci1 = model1.compute_ricci_tensor(r.unsqueeze(0), M, c, G)
                ricci2 = model2.compute_ricci_tensor(r.unsqueeze(0), M, c, G)
                
                if ricci1 is None or ricci2 is None:
                    continue
                    
                # <reason>chain: Extract non-zero components (symmetric tensor)</reason>
                components1 = []
                components2 = []
                indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
                
                for i, j in indices:
                    if hasattr(ricci1[i, j], 'item'):
                        components1.append(ricci1[i, j].squeeze())
                        components2.append(ricci2[i, j].squeeze()) 
                    else:
                        components1.append(ricci1[i, j])
                        components2.append(ricci2[i, j])
                        
                # <reason>chain: Compute normalized difference</reason>
                ricci1_vec = torch.stack(components1)
                ricci2_vec = torch.stack(components2)
                
                diff = torch.abs(ricci1_vec - ricci2_vec)
                norm = torch.abs(ricci1_vec) + torch.abs(ricci2_vec) + epsilon
                normalized_diff = diff / norm
                
                loss = torch.mean(normalized_diff)
                total_loss += loss.item()
                valid_samples += 1
                
            except Exception:
                # <reason>chain: Handle numerical instabilities gracefully</reason>
                continue
                
        if valid_samples == 0:
            return float('inf')
            
        return total_loss / valid_samples
        
    def compute_trajectory_loss(self, 
                               hist1: Tensor, 
                               hist2: Tensor, 
                               loss_type: str,
                               length_scale: float = None) -> float:
        """
        Compute loss between two trajectories.
        <reason>chain: Multiple loss types provide different perspectives on trajectory differences</reason>
        <reason>chain: Only compare observable portions outside event horizon</reason>
        
        Args:
            hist1: First trajectory tensor
            hist2: Second trajectory tensor
            loss_type: Type of loss to compute
            length_scale: Length scale for geometric unit conversion (optional)
            
        Returns:
            Computed loss value
        """
        # <reason>chain: Truncate trajectories at event horizon if length scale provided</reason>
        if length_scale is not None:
            hist1 = self._truncate_at_horizon(hist1, length_scale)
            hist2 = self._truncate_at_horizon(hist2, length_scale)
            
        if loss_type == 'fft':
            # <reason>chain: FFT loss captures frequency domain differences</reason>
            r_pred = hist1[:, 1]
            r_base = hist2[:len(r_pred), 1]  # Match length
            fft_pred = torch.fft.fft(r_pred)
            fft_base = torch.fft.fft(r_base)
            return torch.mean(torch.abs(fft_pred - fft_base)**2).item()
            
        elif loss_type == 'endpoint_mse':
            # <reason>chain: Endpoint loss focuses on final position accuracy</reason>
            end_pred = hist1[-1, :3]
            end_base = hist2[-1, :3]
            return torch.mean((end_pred - end_base)**2).item()
            
        elif loss_type == 'cosine':
            # <reason>chain: Cosine similarity measures trajectory shape similarity</reason>
            min_len = min(len(hist1), len(hist2))
            return torch.mean(
                1 - torch.nn.functional.cosine_similarity(
                    hist1[:min_len, :3], 
                    hist2[:min_len, :3], 
                    dim=1
                )
            ).item()
            
        elif loss_type == 'trajectory_mse':
            # <reason>chain: MSE provides overall trajectory difference</reason>
            min_len = min(len(hist1), len(hist2))
            return torch.mean((hist1[:min_len, :3] - hist2[:min_len, :3])**2).item()
            
        elif loss_type == 'hausdorff':
            # <reason>chain: Hausdorff distance captures worst-case deviation</reason>
            from scipy.spatial.distance import directed_hausdorff
            points_pred = hist1[:, :3].cpu().numpy()
            points_base = hist2[:, :3].cpu().numpy()
            return max(
                directed_hausdorff(points_pred, points_base)[0],
                directed_hausdorff(points_base, points_pred)[0]
            )
            
        elif loss_type == 'frechet':
            # <reason>chain: Frechet distance considers trajectory ordering</reason>
            return self._compute_frechet_distance(hist1[:, :3], hist2[:, :3])
            
        elif loss_type == 'trajectory_dot':
            # <reason>chain: Dot product loss for trajectory alignment</reason>
            min_len = min(len(hist1), len(hist2))
            dots = torch.sum(hist1[:min_len, :3] * hist2[:min_len, :3], dim=1)
            return -torch.mean(dots).item()
            
        elif loss_type == 'raw_dot':
            # <reason>chain: Raw dot product without normalization</reason>
            min_len = min(len(hist1), len(hist2))
            flat1 = hist1[:min_len, :3].flatten()
            flat2 = hist2[:min_len, :3].flatten()
            return -torch.dot(flat1, flat2).item()
            
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
    def compute_partial_loss(self, 
                           partial_hist: Tensor,
                           baseline_results: Union[Dict[str, Tensor], Tensor],
                           loss_type: str = 'fft',
                           model = None,
                           baseline_theories: Optional[Dict] = None,
                           **kwargs) -> float:
        """
        Compute partial loss for early stopping.
        <reason>chain: Partial loss enables efficient early termination of poor trajectories</reason>
        
        Args:
            partial_hist: Partial trajectory computed so far
            baseline_results: Baseline trajectories or single trajectory
            loss_type: Type of loss to compute
            model: Current model (for Ricci loss)
            baseline_theories: Baseline theory objects (for Ricci loss)
            **kwargs: Additional parameters
            
        Returns:
            Computed partial loss
        """
        # <reason>chain: Handle both dict and single tensor inputs</reason>
        if isinstance(baseline_results, dict) and len(baseline_results) == 1 and 'single' in baseline_results:
            baseline_results = list(baseline_results.values())[0]
            
        if not isinstance(baseline_results, dict):
            baseline_results = {'baseline': baseline_results}
            
        if loss_type == 'ricci' and model and baseline_theories:
            # <reason>chain: Ricci loss requires model comparison, not trajectory</reason>
            RS = kwargs.get('RS', 2.95e3)  # Schwarzschild radius
            r_samples = torch.linspace(RS * 1.5, RS * 100, 100, device=self.device, dtype=self.dtype)
            loss = 0.0
            
            for baseline_name, baseline_hist in baseline_results.items():
                loss += self.compute_ricci_loss(
                    model, 
                    baseline_theories[baseline_name], 
                    r_samples,
                    kwargs.get('M'),
                    kwargs.get('c'), 
                    kwargs.get('G')
                )
            return loss / len(baseline_results)
            
        elif loss_type == 'fft':
            # <reason>chain: Simplified FFT for partial trajectories</reason>
            n = min(1024, len(partial_hist))
            down_hist = partial_hist[::len(partial_hist)//n] if len(partial_hist) > n else partial_hist
            r_fft = torch.fft.fft(down_hist[:, 1])
            loss = 0.0
            
            for baseline_hist in baseline_results.values():
                down_base = baseline_hist[:len(down_hist), 1]
                b_fft = torch.fft.fft(down_base)
                loss += torch.mean(torch.abs(r_fft - b_fft)**2)
                
            return loss / len(baseline_results)
            
        else:
            # <reason>chain: Fallback for unimplemented partial loss types</reason>
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
    def _compute_frechet_distance(self, curve1: Tensor, curve2: Tensor) -> float:
        """
        Compute discrete Frechet distance between two curves.
        <reason>chain: Frechet distance implementation for trajectory comparison</reason>
        """
        
        p = curve1.cpu().numpy()
        q = curve2.cpu().numpy()
        len_p, len_q = len(p), len(q)
        
        if len_p == 0 or len_q == 0:
            return float('inf')
            
        # <reason>chain: Dynamic programming table for Frechet computation</reason>
        ca = np.full((len_p, len_q), -1.0)
        
        def c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
                
            if i == 0 and j == 0:
                ca[i, j] = np.linalg.norm(p[0] - q[0])
            elif i > 0 and j == 0:
                ca[i, j] = max(c(i-1, 0), np.linalg.norm(p[i] - q[0]))
            elif i == 0 and j > 0:
                ca[i, j] = max(c(0, j-1), np.linalg.norm(p[0] - q[j]))
            elif i > 0 and j > 0:
                ca[i, j] = max(
                    min(c(i-1, j), c(i-1, j-1), c(i, j-1)),
                    np.linalg.norm(p[i] - q[j])
                )
            else:
                ca[i, j] = float('inf')
                
            return ca[i, j]
            
        return c(len_p - 1, len_q - 1) 

def compute_quantum_trajectory_loss(self, quantum_hist: Tensor, classical_hist: Tensor) -> float:
    """
    Compute loss between quantum and classical trajectories.
    <reason>chain: Specific loss for quantum vs classical comparison to address feedback on quantum properties.</reason>
    
    This could use differences in path amplitudes or uncertainties.
    """
    # Simplified: use MSE plus a term for quantum spread
    mse = self.compute_trajectory_loss(quantum_hist, classical_hist, 'trajectory_mse')
    # Add quantum uncertainty term (placeholder)
    quantum_spread = torch.std(quantum_hist[:, 1])  # Std of radial position
    return mse + 0.1 * quantum_spread.item() 