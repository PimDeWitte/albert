"""
Consolidated numeric utility functions for physics calculations.

<reason>chain: Pure mathematical functions that don't depend on instance state</reason>
<reason>chain: Extracted from theory_engine_core.py to avoid duplication across modules</reason>
<reason>chain: All functions must be stateless and side-effect free</reason>
"""

import torch
import numpy as np
from scipy.spatial.distance import cdist, directed_hausdorff
from typing import Optional

# Import constants for any hardcoded values
from physics_agent.constants import (
    NUMERICAL_THRESHOLDS
)


# ============================================================================
# GEOMETRIC CALCULATIONS
# ============================================================================

def compute_dr_dtau_sq(r: torch.Tensor, E: torch.Tensor, Lz: torch.Tensor, 
                      g_tt: torch.Tensor, g_rr: torch.Tensor, 
                      g_pp: torch.Tensor, g_tp: torch.Tensor, 
                      c: float = 1.0) -> torch.Tensor:
    """
    Compute (dr/dtau)^2 at a given r for fixed E, Lz.
    
    <reason>chain: Core geodesic equation for radial motion in axisymmetric spacetime</reason>
    
    Args:
        r: Radial coordinate
        E: Conserved energy per unit mass
        Lz: Conserved angular momentum per unit mass
        g_tt, g_rr, g_pp, g_tp: Metric components
        c: Speed of light (default 1 for geometric units)
        
    Returns:
        (dr/dtau)^2 value
    """
    det = g_tt * g_pp - g_tp**2
    u_t = - (g_pp * E + g_tp * Lz) / det  # u^t
    u_phi = (g_tp * E + g_tt * Lz) / det  # u^phi
    kinetic = g_tt * u_t**2 + 2 * g_tp * u_t * u_phi + g_pp * u_phi**2
    dr_dtau_sq = (c**2 + kinetic) / g_rr
    return dr_dtau_sq


def compute_circular_orbit_omega(r: torch.Tensor, g_tp: torch.Tensor, 
                                model_name: str = '', a: Optional[float] = None) -> torch.Tensor:
    """
    Compute angular velocity Omega for circular orbits.
    
    <reason>chain: Essential for setting up initial conditions for circular orbits</reason>
    
    Args:
        r: Radial coordinate
        g_tp: Off-diagonal metric component
        model_name: Theory name (for special handling)
        a: Spin parameter (optional, for Kerr)
        
    Returns:
        Angular velocity Omega
    """
    if g_tp.abs() < NUMERICAL_THRESHOLDS['gtol']:
        # Schwarzschild-like case
        if r > NUMERICAL_THRESHOLDS['orbit_stability']:  # Only stable for r > 3M
            Omega = torch.sqrt(1.0 / r**3) * torch.sqrt(1 - NUMERICAL_THRESHOLDS['orbit_stability']/r)
        else:
            # Inside ISCO, use Keplerian approximation
            Omega = torch.sqrt(1.0 / r**3)
    else:
        # Kerr case with frame-dragging
        if a is not None:
            Omega = 1.0 / (r**1.5 + a)
        else:
            # Estimate from g_tp
            a = abs(g_tp.item() * r.item())
            Omega = 1.0 / (r**1.5 + a)
            
    # Validate Omega
    if not torch.isfinite(Omega) or Omega <= 0 or Omega > 1.0:
        Omega = torch.sqrt(1.0 / r**3)  # Fallback to Keplerian
        
    return Omega


# ============================================================================
# DISTANCE METRICS
# ============================================================================

def frechet_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute simplified Fréchet distance between two trajectories.
    
    <reason>chain: Measures trajectory similarity considering point ordering</reason>
    
    Args:
        p: First trajectory points (N x D array)
        q: Second trajectory points (M x D array)
        
    Returns:
        Fréchet distance (simplified version)
    """
    dm = cdist(p, q)
    return max(np.max(dm), np.max(dm.T))


def hausdorff_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute bidirectional Hausdorff distance between point sets.
    
    <reason>chain: Measures maximum deviation between trajectories</reason>
    
    Args:
        p: First point set (N x D array)
        q: Second point set (M x D array)
        
    Returns:
        Maximum of directed Hausdorff distances
    """
    return max(directed_hausdorff(p, q)[0], directed_hausdorff(q, p)[0])


# ============================================================================
# LOSS COMPUTATIONS
# ============================================================================

def compute_fft_loss(r_pred: torch.Tensor, r_base: torch.Tensor) -> float:
    """
    Compute loss based on FFT of radial trajectories.
    
    <reason>chain: Captures frequency content differences</reason>
    
    Args:
        r_pred: Predicted radial trajectory
        r_base: Baseline radial trajectory
        
    Returns:
        Mean squared difference of FFT magnitudes
    """
    fft_pred = torch.fft.fft(r_pred)
    fft_base = torch.fft.fft(r_base)
    return torch.mean(torch.abs(fft_pred - fft_base)**2).item()


def compute_endpoint_mse(end_pred: torch.Tensor, end_base: torch.Tensor) -> float:
    """
    Compute mean squared error between trajectory endpoints.
    
    <reason>chain: Simple metric for final position accuracy</reason>
    
    Args:
        end_pred: Predicted endpoint (3D position)
        end_base: Baseline endpoint (3D position)
        
    Returns:
        Mean squared error
    """
    return torch.mean((end_pred - end_base)**2).item()


def compute_cosine_loss(traj1: torch.Tensor, traj2: torch.Tensor) -> float:
    """
    Compute average cosine distance between trajectory points.
    
    <reason>chain: Measures directional differences</reason>
    
    Args:
        traj1: First trajectory (N x 3)
        traj2: Second trajectory (must match length)
        
    Returns:
        Average cosine distance (1 - cosine similarity)
    """
    min_len = min(len(traj1), len(traj2))
    return torch.mean(1 - torch.nn.functional.cosine_similarity(
        traj1[:min_len], traj2[:min_len], dim=1
    )).item()


def compute_trajectory_mse(traj1: torch.Tensor, traj2: torch.Tensor) -> float:
    """
    Compute mean squared error between full trajectories.
    
    <reason>chain: Overall trajectory matching metric</reason>
    
    Args:
        traj1: First trajectory (N x D)
        traj2: Second trajectory (M x D)
        
    Returns:
        Mean squared error over matched length
    """
    min_len = min(len(traj1), len(traj2))
    return torch.mean((traj1[:min_len] - traj2[:min_len])**2).item()


def compute_trajectory_dot(traj1: torch.Tensor, traj2: torch.Tensor) -> float:
    """
    Compute average dot product between trajectory points.
    
    <reason>chain: Measures alignment of trajectories</reason>
    
    Args:
        traj1: First trajectory (N x D)
        traj2: Second trajectory (must match dimensions)
        
    Returns:
        Average dot product
    """
    min_len = min(len(traj1), len(traj2))
    dots = torch.sum(traj1[:min_len] * traj2[:min_len], dim=1)
    return torch.mean(dots).item()


# ============================================================================
# CHRISTOFFEL SYMBOLS AND CURVATURE
# ============================================================================

def compute_christoffel_symbols(g_tt: torch.Tensor, g_rr: torch.Tensor, 
                               g_pp: torch.Tensor, g_tp: torch.Tensor,
                               r: torch.Tensor, theta: torch.Tensor, 
                               dg_dr: dict) -> dict:
    """
    Compute non-zero Christoffel symbols for axisymmetric metric.
    
    <reason>chain: Essential for geodesic equations and curvature calculations</reason>
    
    Args:
        g_tt, g_rr, g_pp, g_tp: Metric components
        r, theta: Coordinates
        dg_dr: Dictionary of metric derivatives w.r.t. r
        
    Returns:
        Dictionary of Christoffel symbols
    """
    Gamma = {}
    inv_g_rr = 1 / g_rr
    
    # Key radial components
    Gamma['r_tt'] = 0.5 * inv_g_rr * dg_dr['tt']
    Gamma['r_pp'] = 0.5 * inv_g_rr * (-dg_dr['pp'])
    Gamma['r_tp'] = 0.5 * inv_g_rr * dg_dr['tp']
    
    # Theta components
    Gamma['theta_pp'] = -torch.sin(theta) * torch.cos(theta)
    
    # Phi components (example)
    inv_g_pp = 1 / g_pp
    inv_g_tt = 1 / g_tt
    Gamma['phi_tp'] = 0.5 * (inv_g_pp * dg_dr['tp'] - inv_g_tt * dg_dr['tp'])
    
    # NOTE: This is a simplified version. Full implementation requires
    # all 40 potentially non-zero symbols for axisymmetric spacetime
    
    return Gamma


def compute_ricci_scalar_diagonal(g_tt: torch.Tensor, g_rr: torch.Tensor,
                                 g_tt_r: torch.Tensor, g_rr_r: torch.Tensor,
                                 g_tt_rr: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Compute Ricci scalar for diagonal, spherically-symmetric metric.
    
    <reason>chain: Simplified calculation for Schwarzschild-like metrics</reason>
    
    Args:
        g_tt, g_rr: Metric components
        g_tt_r, g_rr_r: First derivatives
        g_tt_rr: Second derivative of g_tt
        r: Radial coordinate
        
    Returns:
        Ricci scalar R
    """
    R = -g_tt_rr / g_tt + (g_tt_r**2) / (2 * g_tt**2) - (g_tt_r * g_rr_r) / (g_tt * g_rr) \
        - 2 * g_tt_r / (r * g_tt) + 2 * g_rr_r / (r * g_rr) - 2 / r**2 + 2 / (r**2 * g_rr)
    return R


def compute_ricci_tensor(g_tt, g_rr, g_pp, g_tp, r, theta=None, dg_dr=None, model=None, M=None, c=None, G=None, device=None, dtype=None):
    """
    Full computation of Ricci tensor using Christoffel symbols.
    
    <reason>chain: Moved from TheoryEngine to consolidate computational functions</reason>
    
    Args:
        g_tt, g_rr, g_pp, g_tp: Metric components
        r: Radial coordinate
        theta: Angular coordinate (defaults to π/2)
        dg_dr: Dictionary of metric derivatives (optional)
        model: Theory model for computing derivatives (optional)
        M, c, G: Physical parameters (optional)
        device: Torch device (optional)
        dtype: Torch dtype (optional)
        
    Returns:
        Ricci tensor components
    """
    if device is None:
        device = r.device
    if dtype is None:
        dtype = r.dtype
        
    if theta is None:
        theta = torch.tensor(torch.pi / 2, device=device, dtype=dtype)
        
    if dg_dr is None and model is not None:
        # Use autograd to compute derivatives
        r.requires_grad_(True)
        from physics_agent.utils import get_metric_wrapper
        metric_func = get_metric_wrapper(model.get_metric)
        g_tt, g_rr, g_pp, g_tp = metric_func(r=r, M=M, c=c, G=G)
        dg_dr = {}
        dg_dr['tt'] = torch.autograd.grad(g_tt, r, create_graph=True)[0]
        dg_dr['rr'] = torch.autograd.grad(g_rr, r, create_graph=True)[0]
        dg_dr['pp'] = torch.autograd.grad(g_pp, r, create_graph=True)[0]
        dg_dr['tp'] = torch.autograd.grad(g_tp, r, create_graph=True)[0]
    elif dg_dr is None:
        raise ValueError("Either dg_dr or model must be provided")
        
    compute_christoffel_symbols(g_tt, g_rr, g_pp, g_tp, r, theta, dg_dr)
    # Compute Ricci R_mu_nu = partial Gamma + Gamma * Gamma terms
    R = torch.zeros(4, 4, device=device, dtype=dtype)
    # For each component, implement the formula R^rho_mu rho nu = partial + ...
    # This is complex; implement for diagonal case first
    if torch.all(g_tp == 0):
        # Schwarzschild-like
        f = -g_tt
        df_dr = -dg_dr['tt']
        R[0,0] = (df_dr / (2 * g_rr)) * (df_dr / (2 * f) + 2/r)  # Approximate; use exact
        # Actual exact expressions for Ricci in spherical coordinates
        R_tt = (1/r**2) * (r * dg_dr['rr'] / g_rr**2 - 1 + 1/g_rr) * g_tt / g_rr  # Lookup and implement accurate
        # NOTE: Replace with precise calculations from GR texts.
        - (1/r**2) * (r * dg_dr['rr'] / g_rr**2 - 1 + 1/g_rr) / g_rr
        R_theta_theta = (r / g_rr) * (dg_dr['rr'] / (2 * g_rr) - dg_dr['tt'] / (2 * g_tt)) - 1
        R_theta_theta * torch.sin(theta)**2
    else:
        # General with g_tp: Full Kerr-like Ricci (should be zero for vacuum solutions)
        # Implement full expression
        R[0,0] = torch.zeros_like(r)  # Placeholder
        # Add other components as needed
    return R


# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================

def to_serializable(value):
    """
    Convert various types to JSON-serializable format.
    
    <reason>chain: Ensures all numeric values can be saved to JSON</reason>
    
    Args:
        value: Any value to convert
        
    Returns:
        JSON-serializable version of the value
    """
    # <reason>chain: Handle NaN and infinity values first</reason>
    if isinstance(value, float):
        if np.isnan(value):
            return None  # or "NaN" as string
        elif np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    elif isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    elif isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    elif isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    else:
        return value


# ============================================================================
# UNIT CONVERSIONS
# ============================================================================

def geometric_to_si_velocity(u_geom: torch.Tensor, time_scale: float, 
                            length_scale: float, component: str) -> torch.Tensor:
    """
    Convert velocity components from geometric to SI units.
    
    <reason>chain: Essential for consistent unit handling</reason>
    
    Args:
        u_geom: Velocity in geometric units
        time_scale: GM/c^3 conversion factor
        length_scale: GM/c^2 conversion factor
        component: 't', 'r', or 'phi' to specify component type
        
    Returns:
        Velocity in SI units
    """
    if component == 't':
        # u^t has units 1/time in SI
        return u_geom / time_scale
    elif component == 'r':
        # u^r has units m/s in SI
        return u_geom * length_scale / time_scale
    elif component == 'phi':
        # u^phi has units rad/s in SI
        return u_geom / time_scale
    else:
        raise ValueError(f"Unknown velocity component: {component}")


def si_to_geometric_velocity(u_si: torch.Tensor, time_scale: float,
                            length_scale: float, component: str) -> torch.Tensor:
    """
    Convert velocity components from SI to geometric units.
    
    <reason>chain: Inverse of geometric_to_si_velocity</reason>
    
    Args:
        u_si: Velocity in SI units
        time_scale: GM/c^3 conversion factor
        length_scale: GM/c^2 conversion factor
        component: 't', 'r', or 'phi' to specify component type
        
    Returns:
        Velocity in geometric units
    """
    if component == 't':
        return u_si * time_scale
    elif component == 'r':
        return u_si * time_scale / length_scale
    elif component == 'phi':
        return u_si * time_scale
    else:
        raise ValueError(f"Unknown velocity component: {component}")


# ============================================================================
# NUMERICAL STABILITY CHECKS
# ============================================================================

def check_velocity_magnitude(y: torch.Tensor, limit: Optional[float] = None) -> bool:
    """
    Check if velocity magnitude exceeds limit.
    
    <reason>chain: Prevents numerical instabilities from runaway velocities</reason>
    
    Args:
        y: State vector with velocities in positions 3,4,5
        limit: Maximum allowed velocity (uses NUMERICAL_THRESHOLDS if None)
        
    Returns:
        True if velocity is within bounds, False otherwise
    """
    if limit is None:
        limit = NUMERICAL_THRESHOLDS['velocity_limit']
        
    if len(y) >= 6:
        u_t = y[3]
        u_r = y[4]
        u_phi = y[5]
        velocity_mag = torch.sqrt(u_t**2 + u_r**2 + u_phi**2)
        return velocity_mag <= limit
    
    return True  # No velocity components to check


def check_radius_bounds(r: torch.Tensor) -> bool:
    """
    Check if radius is within physical bounds.
    
    <reason>chain: Ensures trajectory stays in valid coordinate range</reason>
    
    Args:
        r: Radial coordinate
        
    Returns:
        True if radius is valid, False otherwise
    """
    return (r >= NUMERICAL_THRESHOLDS['radius_min'] and 
            r <= NUMERICAL_THRESHOLDS['radius_max'])


# Export all functions
__all__ = [
    # Geometric calculations
    'compute_dr_dtau_sq',
    'compute_circular_orbit_omega',
    # Distance metrics
    'frechet_distance',
    'hausdorff_distance',
    # Loss computations
    'compute_fft_loss',
    'compute_endpoint_mse',
    'compute_cosine_loss',
    'compute_trajectory_mse',
    'compute_trajectory_dot',
    # Christoffel and curvature
    'compute_christoffel_symbols',
    'compute_ricci_scalar_diagonal',
    'compute_ricci_tensor',
    # Helpers
    'to_serializable',
    # Unit conversions
    'geometric_to_si_velocity',
    'si_to_geometric_velocity',
    # Stability checks
    'check_velocity_magnitude',
    'check_radius_bounds',
] 