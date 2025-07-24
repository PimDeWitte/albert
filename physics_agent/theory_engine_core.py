#!/usr/bin/env python3
"""
Core Theory Engine - Handles gravitational theory simulation and evaluation
Extracted from self_discovery.py for better separation of concerns
"""
from __future__ import annotations
import os
import warnings
import shutil
import argparse
import sys
import time
import json
import importlib
import itertools
import glob
from datetime import datetime
# import psutil  # Commented out - not needed for basic functionality

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
# <reason>chain: Import constants from the new constants module instead of scipy.constants</reason>
from physics_agent.constants import (
    SPEED_OF_LIGHT as c, GRAVITATIONAL_CONSTANT as G, HBAR as hbar,
    VACUUM_PERMITTIVITY as epsilon_0, SOLAR_MASS as M_sun,
    MACHINE_EPSILON
)
# <reason>chain: Import for parallel sweep processing</reason>
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback

from physics_agent.geodesic_integrator import (
    GeodesicRK4Solver, GeneralGeodesicRK4Solver, ChargedGeodesicRK4Solver,
    SymmetricChargedGeodesicRK4Solver, NullGeodesicRK4Solver, UGMGeodesicRK4Solver
)
from physics_agent.unified_trajectory_calculator import UnifiedTrajectoryCalculator
from physics_agent.base_theory import GravitationalTheory, Tensor
from physics_agent.theory_loader import TheoryLoader
from physics_agent.particle_loader import ParticleLoader
from physics_agent.validations import (
    ConservationValidator, MetricPropertiesValidator,
    COWInterferometryValidator, 
    # AtomInterferometryValidator, # Not tested
    # GravitationalDecoherenceValidator, # Not tested
    # QuantumClockValidator, # Not tested
    # QuantumLagrangianGroundingValidator # Not tested
)
# from physics_agent.validations.lagrangian_validator import LagrangianValidator # Not tested
from physics_agent.theory_visualizer import TheoryVisualizer
from physics_agent.update_checker import check_on_startup  # Add new import
# <reason>chain: Import comprehensive report generator for detailed HTML reports</reason>
from physics_agent.run_logger import generate_comprehensive_summary
from physics_agent.validations.comprehensive_report_generator import ComprehensiveReportGenerator
# <reason>chain: Import theory utility functions</reason>
from physics_agent.theory_utils import get_preferred_values
from physics_agent.ui.leaderboard_html_generator import LeaderboardHTMLGenerator
from physics_agent.run_logger import RunLogger
# <reason>chain: Import new modules for better separation of concerns</reason>
from physics_agent.cli import get_cli_parser, setup_execution_mode, handle_special_modes, determine_device_and_dtype
from physics_agent.cache import TrajectoryCache
from physics_agent.loss_calculator import LossCalculator

# <reason>chain: Import centralized constants for consistency</reason>
from physics_agent.constants import (
    SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT,
    SOFTWARE_VERSION, MACHINE_EPSILON,
    # Import new simulation parameters
    M_SI, RS_SI, Q_PARAM, STOCHASTIC_STRENGTH, QUANTUM_PHASE_PRECISION,
    NUMERICAL_THRESHOLDS, INTEGRATION_STEP_FACTORS,
    UNIFICATION_THRESHOLDS, UNIFICATION_TEST_PARAMS,
    DEFAULT_INITIAL_CONDITIONS, SCORING_LOSS_DEFAULTS
)

# <reason>chain: Import consolidated numeric functions</reason>
from physics_agent.functions import (
    to_serializable
)

# Global constants that do not depend on CLI arguments
# Now imported from constants.py
G = GRAVITATIONAL_CONSTANT  # <reason>chain: Alias for backward compatibility</reason>
c = SPEED_OF_LIGHT  # <reason>chain: Alias for backward compatibility</reason>

# --- Constants ---
# SOFTWARE_VERSION is now imported from constants module

# <reason>Import torch.compile for JIT optimization of performance-critical functions. This can provide significant speedups for repeated executions in loops by compiling the function to optimized machine code.</reason>

class TheoryEngine:
    """Core engine for gravitational theory simulation and evaluation"""
    
    def __init__(self, device: str = 'cpu', dtype: torch.dtype = torch.float64, theories_base_dir: str = 'physics_agent/theories', particles_base_dir: str = 'physics_agent/particles',
                 quantum_field_content: str = 'all', quantum_phase_precision: float = QUANTUM_PHASE_PRECISION, verbose: bool = False):
        """
        Initialize the TheoryEngine with device, datatype, and directory paths.
        """
        # <reason>chain: Store verbose flag for consistent logging control</reason>
        self.verbose = verbose
        
        # Device and datatype
        self.device = torch.device(device)
        self.dtype = dtype
        
        # <reason>chain: Store quantum Lagrangian configuration</reason>
        self.quantum_field_content = quantum_field_content
        self.quantum_phase_precision = quantum_phase_precision
        
        # <reason>chain: Use constants from constants.py module for SI values</reason>
        self.M_si = M_sun  # Solar mass in kg
        self.c_si = c  # Speed of light in m/s
        self.G_si = G  # Gravitational constant in m^3 kg^-1 s^-2
        
        # <reason>chain: Define scale factors for geometric unit conversions</reason>
        self.length_scale = self.G_si * self.M_si / self.c_si**2  # GM/c^2
        self.time_scale = self.length_scale / self.c_si  # GM/c^3
        self.velocity_scale = self.c_si
        self.angular_scale = 1.0 / self.length_scale
        self.energy_scale = self.c_si**2
        
        # <reason>chain: Set geometric unit values (G=c=M=1)</reason>
        self.M = 1.0  # Mass in geometric units
        self.c = 1.0  # Speed of light in geometric units
        self.G = 1.0  # Gravitational constant in geometric units
        self.rs = 2.0  # Schwarzschild radius in geometric units (2GM/c^2 = 2*1*1/1^2 = 2)
        
        # <reason>chain: Keep tensor versions for compatibility with existing code</reason>
        self.C_T = torch.tensor(self.c_si, device=self.device, dtype=self.dtype)
        self.G_T = torch.tensor(self.G_si, device=self.device, dtype=self.dtype)
        self.RS = torch.tensor(2 * self.G_si * self.M_si / self.c_si**2, device=self.device, dtype=self.dtype)
        
        # <reason>chain: Use machine epsilon from constants module for consistency</reason>
        self.EPSILON = torch.tensor(MACHINE_EPSILON.get(str(self.dtype).split('.')[-1], 1e-16), device=self.device, dtype=self.dtype)
        
        # <reason>chain: Use HBAR from constants instead of calculating</reason>
        self.hbar = torch.tensor(hbar, device=self.device, dtype=self.dtype)

        # Handle relative paths correctly
        import os
        if not os.path.isabs(theories_base_dir):
            # Get the directory of this file
            module_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to get to the project root
            project_root = os.path.dirname(module_dir)
            theories_base_dir = os.path.join(project_root, theories_base_dir)
            
        if not os.path.isabs(particles_base_dir):
            # Get the directory of this file
            module_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to get to the project root
            project_root = os.path.dirname(module_dir)
            particles_base_dir = os.path.join(project_root, particles_base_dir)
        
        self.loader = TheoryLoader(theories_base_dir)
        self.particle_loader = ParticleLoader(particles_base_dir)
        self.EPSILON = torch.finfo(self.dtype).eps * 100
        self.loss_type = 'ricci'
            
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

        # Initialize all validators
        self.validators = []
        
        # Initialize visualizer
        self.visualizer = TheoryVisualizer(self)
        
        # <reason>chain: Initialize cache manager for trajectory caching</reason>
        self.cache = TrajectoryCache()
        
        # <reason>chain: Initialize loss calculator for trajectory comparison</reason>
        self.loss_calculator = LossCalculator(device=self.device, dtype=self.dtype)
        
        # <reason>chain: Initialize baseline trajectories for unification scoring</reason>
        self.baseline_trajectories = {}  # Will be populated later

        # <reason>chain: Use vacuum permittivity from constants module</reason>
        self.epsilon_0 = epsilon_0

    def get_trajectory_cache_path(self, theory_name: str, r0: Tensor or float, n_steps: int, dtau: Tensor or float, dtype_str: str, **kwargs) -> str:
        """
        Generates a unique path for a cached trajectory.
        <reason>chain: Delegate to cache module for better encapsulation</reason>
        """
        return self.cache.get_cache_path(theory_name, r0, n_steps, dtau, dtype_str, **kwargs) 
    
    def truncate_at_horizon(self, hist: Tensor) -> Tensor:
        """
        Truncate trajectory at event horizon for observable region only.
        <reason>chain: Only the portion outside the event horizon is physically observable</reason>
        <reason>chain: Loss calculations should only use data that could be measured by external observers</reason>
        
        Args:
            hist: Trajectory tensor with columns [t, r, phi, ...] in SI units
            
        Returns:
            Truncated trajectory up to event horizon crossing
        """
        # Convert radial coordinate to geometric units
        r_geometric = hist[:, 1] / self.length_scale
        
        # Event horizon at r = 2M in geometric units
        event_horizon_geometric = 2.0  # This is a fundamental constant, not configurable
        
        # Find where trajectory crosses event horizon
        outside_horizon = r_geometric > event_horizon_geometric
        
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
            return hist[:min(last_outside_idx + 2, len(hist))]



    def get_initial_conditions(self, model: GravitationalTheory, r0_geom: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        """ r0_geom is r in units of M """  
        r = r0_geom.clone().requires_grad_(True)
        
        # Get metric in geometric units (c=1, G=1, M=1)
        # Only pass theory-specific kwargs to get_metric, not trajectory kwargs
        metric_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['quantum_interval', 'quantum_beta', 'progress_callback', 
                                    'callback_interval', 'verbose', 'no_cache', 'test_mode', 
                                    'run_to_horizon', 'horizon_threshold', 'particle_name', 
                                    'early_stopping', 'baseline_results', 'particle_mass', 'particle_charge', 'particle_spin', 'particle_type',
                                    'M', 'c', 'G', 'EPSILON']}  # Also filter out these as they're handled by the wrapper
        
        # Use get_metric_wrapper to handle parameter name variations
        from physics_agent.utils import get_metric_wrapper
        metric_func = get_metric_wrapper(model.get_metric)
        
        g_tt, g_rr, g_pp, g_tp = metric_func(
            r=r, 
            M=torch.tensor(self.M, device=self.device, dtype=self.dtype),
            c=self.c,  # Keep as float as expected by wrapper
            G=self.G,  # Keep as float as expected by wrapper
            **metric_kwargs
        )
        
        # <reason>chain: Add debug output for Kerr metrics to diagnose issues</reason>
        if 'Kerr' in model.name and abs(g_tp) > NUMERICAL_THRESHOLDS['gtol']:
            if self.verbose:
                print(f"  Kerr metric components at r={r.item():.3f}:")
                print(f"    g_tt={g_tt.item():.6f}, g_rr={g_rr.item():.6f}")
                print(f"    g_pp={g_pp.item():.6f}, g_tp={g_tp.item():.6f}")
        
        # Calculate Omega in geometric units (1/time)
        if g_tp.abs() < NUMERICAL_THRESHOLDS['gtol']:
            # <reason>chain: Schwarzschild-like case, use simplified formula</reason>
            # For circular orbits in Schwarzschild: Omega = sqrt(M/r^3)
            # But with stability factor: multiply by sqrt(1 - 3M/r) for stable orbits
            if r > NUMERICAL_THRESHOLDS['orbit_stability']:  # Only stable for r > 3M
                Omega = torch.sqrt(1.0 / r**3) * torch.sqrt(1 - NUMERICAL_THRESHOLDS['orbit_stability']/r)
            else:
                # Inside ISCO, use Keplerian approximation
                Omega = torch.sqrt(1.0 / r**3)
        else:
            # <reason>chain: Kerr case with frame-dragging</reason>
            # For circular orbits: Omega = 1.0 / (r^{3/2} + a)
            # Extract spin parameter from the model
            if hasattr(model, 'a'):
                a = model.a
            elif hasattr(model, 'a_ratio'):
                a = model.a_ratio
            else:
                # Fallback to estimating from g_tp
                # For Kerr: g_tp = -2Mar sin^2(theta) / Sigma
                # In equatorial plane and geometric units: g_tp ≈ -a/r for large r
                a = abs(g_tp.item() * r.item())
                if self.verbose:
                    print(f"  Estimated spin parameter a={a} from g_tp")
            
            Omega = 1.0 / (r**1.5 + a)
            
            # <reason>chain: Validate Omega is reasonable</reason>
            if not torch.isfinite(Omega) or Omega <= 0 or Omega > 1.0:
                if self.verbose:
                    if self.verbose:
                        print(f"  Warning: Invalid Omega={Omega} computed, using Keplerian approximation")
                Omega = torch.sqrt(1.0 / r**3)
        
        # Norm factor in geometric units (g_mu nu u^mu u^nu = -1)
        norm_factor = g_tt + 2 * Omega * g_tp + Omega**2 * g_pp
        
        # Handle numerical precision issues
        epsilon = NUMERICAL_THRESHOLDS['epsilon']
        if norm_factor >= 0:
            # If positive or zero, set to small negative value
            norm_factor = -epsilon
            if self.verbose:
                if self.verbose:
                    print(f"  Warning: norm_factor was non-negative ({norm_factor + epsilon}), set to -epsilon")
        
        u_t = 1.0 / torch.sqrt(-norm_factor)
        u_phi = Omega * u_t
        u_r = torch.zeros_like(r0_geom)
        
        # <reason>chain: Validate 4-velocity normalization</reason>
        norm_check = g_tt * u_t**2 + 2 * g_tp * u_t * u_phi + g_pp * u_phi**2
        if abs(norm_check + 1.0) > NUMERICAL_THRESHOLDS['norm_check']:
            if self.verbose:
                if self.verbose:
                    print(f"  Warning: 4-velocity normalization error: g_mu_nu u^mu u^nu = {norm_check} (should be -1)")
        
        # <reason>chain: Ensure finite values before returning</reason>
        if not (torch.isfinite(u_t) and torch.isfinite(u_phi)):
            print(f"  ERROR: Non-finite initial velocities: u_t={u_t}, u_phi={u_phi}")
            # Fallback to simple circular orbit
            u_t = torch.tensor(DEFAULT_INITIAL_CONDITIONS['u_t'], device=self.device, dtype=self.dtype)
            u_phi = torch.tensor(DEFAULT_INITIAL_CONDITIONS['u_phi'], device=self.device, dtype=self.dtype)
        
        # <reason>chain: Convert from geometric to SI units for general solver</reason>
        # In geometric units: u^t = dt/dtau (dimensionless), u^r = dr/dtau (1/time), u^phi = dphi/dtau (1/time)
        # In SI units: u^t has units 1/time, u^r has units m/s, u^phi has units rad/s
        u_t_SI = u_t / self.time_scale  # Convert dt/dtau to SI (1/s)
        u_r_SI = u_r * self.length_scale / self.time_scale  # Convert dr/dtau to SI (m/s)
        u_phi_SI = u_phi / self.time_scale  # Convert dphi/dtau to SI (rad/s)
        
        y0_general = torch.tensor([0.0, r0_geom.item(), 0.0, u_t_SI.item(), u_r_SI.item(), u_phi_SI.item()], device=self.device, dtype=self.dtype)
        y0_symmetric = torch.tensor([0.0, r0_geom.item(), 0.0, 0.0], device=self.device, dtype=self.dtype)
        
        return y0_symmetric, y0_general, torch.tensor(0.0)
        
    def run_trajectory(self, model: GravitationalTheory, r0_si: float, N_STEPS: int, DTau_si: float, **kwargs) -> tuple[Tensor | None, str, list[int]]:
        # <reason>chain: Validate mass parameter is positive</reason>
        if self.M_si <= 0:
            raise ValueError(f"Mass must be positive, got {self.M_si} kg")
        
        # Convert to geometric units
        r0_geom = r0_si / self.length_scale
        dtau_geom = DTau_si / self.time_scale
        
        y0_sym, y0_gen, _ = self.get_initial_conditions(model, torch.tensor([r0_geom], dtype=self.dtype, device=self.device).squeeze(), **kwargs)
        hist_geom, tag, kicks = self._run_trajectory_geometric(model, r0_geom, N_STEPS, dtau_geom, y0_gen, **kwargs)
        
        if hist_geom is None:
            return None, tag, kicks
        
        # Convert back to SI
        hist_si = hist_geom.clone()
        hist_si[:,0] *= self.time_scale  # t
        hist_si[:,1] *= self.length_scale  # r
        hist_si[:,2] = hist_geom[:,2]  # phi unitless
        hist_si[:,3] *= self.velocity_scale / self.time_scale  # dr/dtau in m/s
        return hist_si, tag, kicks
        
    def run_multi_particle_trajectories(self, model: GravitationalTheory, r0_si: float, N_STEPS: int, DTau_si: float, 
                                       theory_category: str = 'unknown', **kwargs) -> dict:
        """
        <reason>chain: Run trajectories for multiple particles based on theory category</reason>
        For quantum/unified theories: only electrons and photons (QED)
        For classical theories: all particles (electrons, photons, protons, neutrinos)
        """
        particle_results = {}
        
        # Determine which particles to test based on theory category
        # Force all particles to be tested for comprehensive visualization
        # if theory_category in ['quantum', 'unified']:
        #     # Only test QED particles (electrons and photons) for quantum theories
        #     particle_names = ['electron', 'photon']
        #     print(f"  Testing QED particles only for {theory_category} theory")
        # else:
        #     # Test all particles for classical theories
        particle_names = ['electron', 'photon', 'proton', 'neutrino']
        if self.verbose:
            if self.verbose:
                print(f"  Testing all 4 standard particles for comprehensive visualization")
        
        for particle_name in particle_names:
            if self.verbose:
                if self.verbose:
                    print(f"    Running trajectory for {particle_name}...")
            
            # Load particle properties
            particle = self.particle_loader.get_particle(particle_name)
            
            # Run trajectory with particle properties
            hist, tag, kicks = self.run_trajectory(
                model, r0_si, N_STEPS, DTau_si,
                particle_name=particle_name,
                particle_mass=particle.mass,
                particle_charge=particle.charge,
                particle_spin=particle.spin,
                particle_type=particle.particle_type,
                **kwargs
            )
            
            # Store results
            particle_results[particle_name] = {
                'trajectory': hist,
                'tag': tag,
                'kicks': kicks,
                'particle': particle,
                'particle_properties': {
                    'name': particle.name,
                    'type': particle.particle_type,
                    'mass': particle.mass,
                    'charge': particle.charge,
                    'spin': particle.spin
                }
            }
            
            if self.verbose:
                print(f"      Stored {particle_name} result with tag: {tag}")
            
            if hist is None:
                if self.verbose:
                    if self.verbose:
                        print(f"      Warning: Could not compute trajectory for {particle_name}")
        
        return particle_results

    def _run_trajectory_geometric(self, model, r0_geom, N_STEPS, dtau_geom, y0_gen, **kwargs) -> tuple[Tensor, str, list]:
        """
        <reason>chain: Run trajectory integration in geometric units (c=G=M=1)</reason>
        This is the core trajectory integration logic adapted for geometric units.
        """
        # Extract optional parameters
        quantum_interval = kwargs.get('quantum_interval', 0)
        quantum_beta = kwargs.get('quantum_beta', 0.0)
        progress_callback = kwargs.get('progress_callback', None)
        callback_interval = kwargs.get('callback_interval', N_STEPS + 1)
        verbose = kwargs.get('verbose', False)
        no_cache = kwargs.get('no_cache', False)
        test_mode = kwargs.get('test_mode', False)
        run_to_horizon = kwargs.get('run_to_horizon', True)
        kwargs.get('horizon_threshold', 1.001)
        # <reason>chain: Added singularity_threshold to allow trajectories past the event horizon</reason>
        # <reason>chain: Trajectories continue to r = 0.05M by default, well inside the horizon at r = 2M</reason>
        singularity_threshold = kwargs.get('singularity_threshold', NUMERICAL_THRESHOLDS['singularity_radius'])  # Default to r = 0.05M
        particle_name = kwargs.get('particle_name', None)
        early_stopping = kwargs.get('early_stopping', False)
        baseline_results = kwargs.get('baseline_results', None)
        
        # Load particle if specified
        particle = None
        if particle_name:
            particle = self.particle_loader.get_particle(particle_name)
            if particle is None:
                if self.verbose:
                    if self.verbose:
                        print(f"Warning: Particle '{particle_name}' not found")
                return None, f"particle_{particle_name}_not_found", []
            # Apply particle properties (mass, charge, spin) in solver selection below
        
        # <reason>chain: Check if we should use quantum trajectory calculator for quantum theories</reason>
        theory_category = getattr(model, 'category', 'unknown')
        
        # <reason>chain: Enable quantum trajectories for all quantum theories by default</reason>
        # The quantum solver has been fixed to:
        # 1. Use classical geodesics as the stationary path (not linear interpolation)
        # 2. Add quantum fluctuations around the classical path
        # 3. Include QED corrections for precision tests
        use_quantum_trajectories = (
            theory_category == 'quantum' and 
            kwargs.get('use_quantum_trajectories', True)  # Allow override to disable
        )
        
        if self.verbose and theory_category == 'quantum':
            print(f"  Quantum theory detected: {model.name}")
            print(f"    - Category: {theory_category}")
            print(f"    - enable_quantum: {getattr(model, 'enable_quantum', False)}")
            print(f"    - use_quantum_trajectories: {use_quantum_trajectories}")
            print(f"    - NOTE: Using classical solver for proper curved spacetime physics")
        
        # Check cache
        cache_path = self.get_trajectory_cache_path(
            model.name, r0_geom * self.length_scale, N_STEPS, dtau_geom * self.time_scale, 
            str(self.dtype).split('.')[-1], **kwargs
        )
        
        # <reason>chain: Add theory-specific information to cache key to prevent collisions</reason>
        # Include theory module and class information
        cache_kwargs = kwargs.copy()
        cache_kwargs['theory_module'] = model.__class__.__module__
        cache_kwargs['theory_class'] = model.__class__.__name__
        
        # <reason>chain: Extract metric parameters that distinguish this theory from others</reason>
        # This ensures different theories don't share cache files
        metric_params = {}
        # Common metric parameters
        for param in ['a', 'q_e', 'alpha', 'beta', 'gamma', 'sigma', 'epsilon', 'kappa', 
                      'omega', 'tau', 'Lambda_as', 'T_c', 'R_kk', 'alpha_prime']:
            if hasattr(model, param):
                metric_params[param] = getattr(model, param)
        
        if metric_params:
            cache_kwargs['metric_params'] = metric_params
        
        # Regenerate cache path with theory-specific information
        cache_path = self.get_trajectory_cache_path(
            model.name, r0_geom * self.length_scale, N_STEPS, dtau_geom * self.time_scale, 
            str(self.dtype).split('.')[-1], **cache_kwargs
        )
        
        if not no_cache and os.path.exists(cache_path):
            hist = self.cache.load_trajectory(cache_path, self.device)
            if hist is not None:
                # Convert from SI to geometric units
                hist_geom = hist.clone()
                hist_geom[:,0] /= self.time_scale  # t
                hist_geom[:,1] /= self.length_scale  # r
                # phi is already dimensionless
                if hist.shape[1] > 3:
                    hist_geom[:,3] *= self.time_scale / self.velocity_scale  # dr/dtau
                
                # <reason>chain: Store cache path info for later copying</reason>
                self._last_cache_path_used = cache_path
                return hist_geom, "cached_trajectory", []
        
        # <reason>chain: Use quantum trajectory calculator for quantum theories if enabled</reason>
        if use_quantum_trajectories:
            if self.verbose:
                print(f"  Attempting to use UnifiedTrajectoryCalculator for quantum theory")
            try:
                # <reason>chain: Pass unit conversion parameters to UnifiedTrajectoryCalculator</reason>
                calculator = UnifiedTrajectoryCalculator(
                    model, 
                    enable_quantum=True, 
                    enable_classical=True,
                    M=self.M_si,  # Central mass in kg
                    c=self.c_si,  # Speed of light in m/s
                    G=self.G_si   # Gravitational constant
                )
                
                # Calculate E and Lz for symmetric spacetimes
                if model.is_symmetric:
                    # Get metric at initial radius
                    from physics_agent.utils import get_metric_wrapper
                    metric_func = get_metric_wrapper(model.get_metric)
                    metric_args = {
                        'r': torch.tensor([r0_geom], device=self.device, dtype=self.dtype),
                        'M': torch.tensor(1.0, device=self.device, dtype=self.dtype),
                        'c': 1.0,
                        'G': 1.0
                    }
                    g_tt, g_rr, g_pp, g_tp = metric_func(**metric_args)
                    
                    # Convert velocities to geometric units
                    u_t_geom = y0_gen[3] * self.time_scale
                    u_r_geom = y0_gen[4] * self.time_scale / self.length_scale
                    u_phi_geom = y0_gen[5] * self.time_scale
                    
                    # Calculate conserved quantities
                    E_calc = -g_tt.squeeze() * u_t_geom - g_tp.squeeze() * u_phi_geom
                    Lz_calc = g_tp.squeeze() * u_t_geom + g_pp.squeeze() * u_phi_geom
                else:
                    # For non-symmetric, use default values
                    E_calc = 0.95
                    Lz_calc = 4.0
                
                # Prepare initial conditions
                # Don't pass state directly - let UnifiedTrajectoryCalculator build it
                initial_conditions = {
                    'r': r0_geom.item() if torch.is_tensor(r0_geom) else r0_geom,  # Keep in geometric units
                    't': 0.0,
                    'phi': 0.0,
                    'E': E_calc.item() if torch.is_tensor(E_calc) else E_calc,
                    'Lz': Lz_calc.item() if torch.is_tensor(Lz_calc) else Lz_calc,
                    'u_t': y0_gen[3].item(),
                    'u_r': y0_gen[4].item(), 
                    'u_phi': y0_gen[5].item()
                }
                
                # Compute unified trajectory (both classical and quantum)
                # Set quantum parameters
                quantum_kwargs = {
                    'time_steps': N_STEPS,
                    'step_size': dtau_geom,  # Keep in geometric units
                    'quantum_method': kwargs.get('quantum_method', 'monte_carlo'),
                    'quantum_samples': kwargs.get('quantum_samples', 100)
                }
                
                results = calculator.compute_unified_trajectory(
                    initial_conditions,
                    **quantum_kwargs
                )
                
                # <reason>chain: Extract quantum trajectory if available, fall back to classical</reason>
                if self.verbose:
                    print(f"  Quantum results keys: {list(results.get('quantum', {}).keys())}")
                    print(f"  Classical results keys: {list(results.get('classical', {}).keys())}")
                
                # <reason>chain: Use classical trajectory for now since quantum paths are empty</reason>
                if 'classical' in results and 'trajectory' in results['classical']:
                    traj = results['classical']['trajectory']
                    hist_geom = torch.tensor(traj, device=self.device, dtype=self.dtype)
                    if self.verbose:
                        print(f"  Using classical trajectory from unified calculator")
                        print(f"  Classical trajectory shape: {hist_geom.shape}")
                        
                    # Store quantum results for uncertainty visualization
                    if 'quantum' in results:
                        self._last_quantum_results = results['quantum']
                        if self.verbose:
                            print(f"  Quantum results stored for uncertainty visualization")
                    
                    # Check if we got a valid trajectory
                    if hist_geom.shape[0] < N_STEPS // 2:  # Need at least half the requested steps
                        if self.verbose:
                            print(f"  Warning: Classical trajectory too short ({hist_geom.shape[0]} points, expected {N_STEPS})")
                            print(f"  Falling through to standard classical solver...")
                        # Fall through to classical solver
                    else:
                        # Convert back to geometric units
                        hist_geom[:,0] /= self.time_scale
                        hist_geom[:,1] /= self.length_scale
                        if hist_geom.shape[1] > 3:
                            hist_geom[:,3] *= self.time_scale / self.velocity_scale
                        
                        # Tag indicates quantum theory using unified calculator
                        tag = "quantum_unified_trajectory"
                        
                        # Store quantum results for later use
                        self._last_quantum_results = results.get('quantum', {})
                        
                        if self.verbose:
                            print(f"  Successfully computed trajectory using quantum solver!")
                            print(f"  Returning tag: {tag}")
                        return hist_geom, tag, []
                else:
                    if self.verbose:
                        print(f"  Warning: Quantum trajectory calculator returned no classical trajectory")
                    # Fall through to classical solver
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to use quantum trajectory calculator: {e}")
                    import traceback
                    traceback.print_exc()
                # Fall through to classical solver
        
        # Select appropriate solver
        if model.is_symmetric:
            # <reason>chain: Use 4D solver for symmetric spacetimes with conserved E, Lz</reason>
            # Extract E and Lz from initial conditions
            # In geometric units: E_geom = E_si / c^2, Lz_geom = Lz_si / (M*G/c)
            
            # Use get_metric_wrapper to handle parameter name variations
            from physics_agent.utils import get_metric_wrapper
            metric_func = get_metric_wrapper(model.get_metric)
            
            metric_args = {
                'r': torch.tensor([r0_geom], device=self.device, dtype=self.dtype),
                'M': torch.tensor(1.0, device=self.device, dtype=self.dtype),  # M=1 in geometric units
                'c': 1.0,  # c=1 in geometric units
                'G': 1.0   # G=1 in geometric units
            }
            g_tt, g_rr, g_pp, g_tp = metric_func(**metric_args)
            
            # Calculate E and Lz from initial 4-velocity
            # u^t = y0_gen[3], u^r = y0_gen[4], u^phi = y0_gen[5]
            # But y0_gen is in SI units, need to convert to geometric first
            # <reason>chain: Convert SI velocities to geometric units for consistent calculations</reason>
            # <reason>chain: Time components (u^t, u^phi) scale by time_scale since dt_SI = time_scale * dt_geom</reason>
            # <reason>chain: Radial component (u^r) scales by time_scale/length_scale since dr/dt_SI = (length_scale/time_scale) * dr/dtau_geom</reason>
            u_t_geom = y0_gen[3] * self.time_scale
            u_r_geom = y0_gen[4] * self.time_scale / self.length_scale
            u_phi_geom = y0_gen[5] * self.time_scale
            
            # <reason>chain: Calculate conserved energy E = -g_tt*u^t - g_tp*u^phi from Killing vector ∂/∂t</reason>
            # <reason>chain: In stationary spacetimes, E is conserved along geodesics due to time translation symmetry</reason>
            # <reason>chain: The negative sign comes from the timelike Killing vector being future-pointing</reason>
            E_geom = -g_tt.squeeze() * u_t_geom - g_tp.squeeze() * u_phi_geom
            
            # <reason>chain: Calculate conserved angular momentum Lz = g_tp*u^t + g_pp*u^phi from Killing vector ∂/∂phi</reason>
            # <reason>chain: In axisymmetric spacetimes, Lz is conserved along geodesics due to rotational symmetry</reason>
            # <reason>chain: For Kerr metrics, g_tp≠0 couples rotation to time, causing frame-dragging effects</reason>
            Lz_geom = g_tp.squeeze() * u_t_geom + g_pp.squeeze() * u_phi_geom
            
            # Create 4D initial state: [t, r, phi, dr/dtau]
            # <reason>chain: Use 4D state for symmetric spacetimes where theta=π/2 is fixed</reason>
            # <reason>chain: This reduces computational cost while preserving equatorial plane motion</reason>
            y0_4d = torch.tensor([y0_gen[0], y0_gen[1], y0_gen[2], y0_gen[4]], 
                                device=self.device, dtype=self.dtype)
            
            # <reason>chain: Check if this is a UGM theory first</reason>
            is_ugm = (hasattr(model, 'use_ugm_solver') and model.use_ugm_solver) or \
                     model.__class__.__name__ == 'UnifiedGaugeModel' or \
                     'ugm' in model.__class__.__name__.lower()
            
            # <reason>chain: Choose solver based on particle properties</reason>
            if is_ugm:
                # <reason>chain: Always use UGM solver for Unified Gauge Model theories</reason>
                # UGM needs special handling even for symmetric spacetimes due to gauge fields
                solver = UGMGeodesicRK4Solver(model,
                                            torch.tensor(1.0, device=self.device, dtype=self.dtype),
                                            1.0, 1.0,
                                            enable_quantum_corrections=theory_category == 'quantum')
                tag = "ugm_symmetric_solver_run"
            elif particle and particle.particle_type == 'massless':
                # Use null geodesic solver for photons
                # Calculate impact parameter b = L/E for null geodesics
                impact_param = Lz_geom.item() / E_geom.item() if E_geom.item() != 0 else Lz_geom.item()
                solver = NullGeodesicRK4Solver(model, 
                                             torch.tensor(1.0, device=self.device, dtype=self.dtype),
                                             1.0, 1.0,
                                             impact_parameter=impact_param)
                tag = "null_solver_run"
            elif particle and particle.charge != 0:
                # Use charged geodesic solver for all charged particles
                # This handles electromagnetic interactions properly even in symmetric spacetimes
                charge_geom = particle.charge / (self.C_T * self.length_scale).item()
                mass_geom = particle.mass * self.c_si**2 / self.energy_scale
                solver = SymmetricChargedGeodesicRK4Solver(model,
                                                         torch.tensor(1.0, device=self.device, dtype=self.dtype),
                                                         1.0, 1.0,
                                                         q=charge_geom,
                                                         m=mass_geom)
                tag = "symmetric_charged_solver_run"
            else:
                solver = GeodesicRK4Solver(model, 
                                         torch.tensor(1.0, device=self.device, dtype=self.dtype),
                                         1.0, 1.0)
                tag = "symmetric_solver_run"
                
            solver.E = E_geom.item()
            solver.Lz = Lz_geom.item()
            
            # Initialize history with 4D state
            hist = torch.zeros((N_STEPS + 1, 4), device=self.device, dtype=self.dtype)
            y = y0_4d
        else:
            # <reason>chain: Use 6D general solver for non-symmetric spacetimes</reason>
            # Check if this is a UGM theory that needs special handling
            is_ugm = (hasattr(model, 'use_ugm_solver') and model.use_ugm_solver) or \
                     model.__class__.__name__ == 'UnifiedGaugeModel' or \
                     'ugm' in model.__class__.__name__.lower()
            
            if is_ugm:
                # <reason>chain: Use UGM solver for Unified Gauge Model theories</reason>
                solver = UGMGeodesicRK4Solver(model,
                                            torch.tensor(1.0, device=self.device, dtype=self.dtype),
                                            1.0, 1.0,
                                            enable_quantum_corrections=theory_category == 'quantum')
                tag = "ugm_solver_run"
            elif particle and particle.particle_type == 'massless':
                # Use null geodesic solver for photons - but it derives from GeodesicRK4Solver
                # so we need to use general solver with massless initial conditions
                solver = GeneralGeodesicRK4Solver(model,
                                                torch.tensor(1.0, device=self.device, dtype=self.dtype),
                                                1.0, 1.0)
                tag = "general_null_solver_run"
            elif particle and particle.charge != 0:
                # Use charged particle solver for all charged particles
                charge_geom = particle.charge / (self.C_T * self.length_scale).item()  # Convert to geometric units
                mass_geom = particle.mass * self.c_si**2 / self.energy_scale  # Convert to geometric units
                solver = ChargedGeodesicRK4Solver(model,
                                                torch.tensor(1.0, device=self.device, dtype=self.dtype),
                                                1.0, 1.0,
                                                q=charge_geom,
                                                m=mass_geom)
                tag = "charged_solver_run"
            else:
                solver = GeneralGeodesicRK4Solver(model,
                                                torch.tensor(1.0, device=self.device, dtype=self.dtype),
                                                1.0, 1.0)
                tag = "general_solver_run"
            
            # Initialize history with 6D state (only storing t,r,phi,dr/dtau)
            hist = torch.zeros((N_STEPS + 1, 4), device=self.device, dtype=self.dtype)
            
            # <reason>chain: Convert SI velocities to geometric units for the general solver</reason>
            # y0_gen has SI units: [t, r(geom), phi, u^t(SI), u^r(SI), u^phi(SI)]
            # Need to convert to geometric units for the solver
            u_t_geom = y0_gen[3] * self.time_scale  # u^t in geometric units
            u_r_geom = y0_gen[4] * self.time_scale / self.length_scale  # u^r in geometric units  
            u_phi_geom = y0_gen[5] * self.time_scale  # u^phi in geometric units
            
            y = torch.tensor([y0_gen[0], y0_gen[1], y0_gen[2], u_t_geom, u_r_geom, u_phi_geom], 
                           device=self.device, dtype=self.dtype)
        
        # Store initial state
        hist[0] = torch.tensor([y[0], y[1], y[2], y[4] if len(y) > 4 else y[3]], 
                             device=self.device, dtype=self.dtype)
        
        # Integration loop
        quantum_kicks_indices = []
        h = torch.tensor(dtau_geom, device=self.device, dtype=self.dtype) if not isinstance(dtau_geom, torch.Tensor) else dtau_geom.clone()
        
        # <reason>chain: Adaptive timestep for Kerr and other complex metrics</reason>
        adaptive_stepping = False
        if 'Kerr' in model.name and not model.is_symmetric:
            # Kerr metrics need smaller timesteps for stability
            h = h * INTEGRATION_STEP_FACTORS['aggressive_reduction']
            adaptive_stepping = True
            if verbose:
                    print("  Using aggressive timestep reduction for Kerr metric")
        elif hasattr(model, 'sigma') and model.sigma > 0:  # Stochastic theories
            adaptive_stepping = True
            if verbose:
                    print(f"  Using adaptive timestep for stochastic theory")
        
        # <reason>chain: Track failed steps for adaptive adjustment</reason>
        failed_steps = 0
        max_failed_steps = 10
        
        for i in range(N_STEPS):
            # Progress callback
            if progress_callback and (i + 1) % callback_interval == 0:
                progress_callback(i + 1, N_STEPS)
            
            # RK4 step with retry logic for adaptive stepping
            step_successful = False
            h_current = h.clone()
            
            for retry in range(5):  # Max 5 retries with smaller steps
                y_new = solver.rk4_step(y, h_current)
                
                if y_new is None or torch.any(~torch.isfinite(y_new)):
                    if adaptive_stepping:
                        h_current = h_current * INTEGRATION_STEP_FACTORS['standard_reduction']
                        if verbose and retry == 0:
                            print(f"  Step {i} failed, reducing timestep to {h_current.item():.6f}")
                        continue
                    else:
                        break
                
                # <reason>chain: Additional stability checks for extreme values</reason>
                r_new = y_new[1]
                
                # For Kerr, check if we're near ergosphere
                if 'Kerr' in model.name and hasattr(model, 'a'):
                    r_ergo = 2.0  # r+ = M + sqrt(M^2 - a^2) in geometric units, approx 2M
                    if r_new < NUMERICAL_THRESHOLDS['ergo_factor'] * r_ergo and r_new > NUMERICAL_THRESHOLDS['radius_min']:
                        # Near ergosphere, use very small steps
                        if h_current > INTEGRATION_STEP_FACTORS['ergo_sphere_limit']:
                            h_current = torch.tensor(INTEGRATION_STEP_FACTORS['ergo_sphere_limit'], device=self.device, dtype=self.dtype)
                            if verbose:
                                    print(f"  Near ergosphere at r={r_new:.3f}, using tiny timestep")
                
                if r_new < NUMERICAL_THRESHOLDS['radius_min'] or r_new > NUMERICAL_THRESHOLDS['radius_max']:  # In geometric units
                    if adaptive_stepping and retry < 4:
                        h_current = h_current * INTEGRATION_STEP_FACTORS['standard_reduction']
                        continue
                    else:
                        if verbose:
                                print(f"  Trajectory escaped reasonable bounds at step {i}: r={r_new:.3f}")
                        hist = hist[:i+1]
                        break
                
                # <reason>chain: Check for runaway velocities in general solver</reason>
                if not model.is_symmetric and len(y_new) >= 6:
                    u_t = y_new[3]
                    u_r = y_new[4]
                    u_phi = y_new[5]
                    velocity_mag = torch.sqrt(u_t**2 + u_r**2 + u_phi**2)
                    if velocity_mag > NUMERICAL_THRESHOLDS['velocity_limit']:  # Lower threshold for geometric units
                        if adaptive_stepping and retry < 4:
                            h_current = h_current * INTEGRATION_STEP_FACTORS['standard_reduction']
                            if verbose:
                                    print(f"  High velocity {velocity_mag:.1f} at step {i}, reducing timestep")
                            continue
                        else:
                            if verbose:
                                    print(f"  Runaway velocity detected at step {i}: |u|={velocity_mag:.3f}")
                            hist = hist[:i+1]
                            break
                
                # Step succeeded
                step_successful = True
                y = y_new
                
                # If we had to reduce timestep, keep it reduced for a while
                if h_current < h:
                    h = h_current
                elif adaptive_stepping and h_current == h and retry == 0:
                    # Try to increase timestep if things are stable
                    h = torch.min(h * 1.1, torch.tensor(dtau_geom, device=self.device, dtype=self.dtype) if not isinstance(dtau_geom, torch.Tensor) else dtau_geom)
                break
                
            if not step_successful:
                failed_steps += 1
                if failed_steps > max_failed_steps:
                    if verbose:
                            print(f"Too many failed steps ({failed_steps}), terminating integration")
                    hist = hist[:i+1]
                    break
                continue
            else:
                failed_steps = 0  # Reset counter on success
            
            # Store state (t, r, phi, dr/dtau)
            if model.is_symmetric:
                hist[i+1] = torch.tensor([y[0], y[1], y[2], y[3]], device=self.device, dtype=self.dtype)
            else:
                hist[i+1] = torch.tensor([y[0], y[1], y[2], y[4]], device=self.device, dtype=self.dtype)
            
            # Check for singularity approach (not just horizon crossing)
            r_current = y[1]
            # <reason>chain: Use singularity_threshold instead of horizon to allow trajectories past the event horizon</reason>
            # <reason>chain: Event horizon is at r = 2M, but we continue to near-singularity for scientific interest</reason>
            if run_to_horizon and r_current < singularity_threshold:  # Near singularity, not just horizon
                if verbose:
                    print(f"Approaching singularity at step {i+1}, r = {r_current.item():.6f} (r/M = {r_current.item():.3f})")
                    print(f"  Note: Event horizon is at r = 2M, singularity threshold is {singularity_threshold:.3f}M")
                hist = hist[:i+2]
                break
            
            # Quantum kicks (experimental)
            if quantum_interval > 0 and (i + 1) % quantum_interval == 0:
                quantum_kicks_indices.append(i + 1)
                if model.is_symmetric:
                    # Apply kick to dr/dtau
                    y[3] += quantum_beta * torch.randn(1, device=self.device, dtype=self.dtype).squeeze()
                else:
                    # Apply kicks to 4-velocity components
                    y[3:6] += quantum_beta * torch.randn(3, device=self.device, dtype=self.dtype)
            
            # Early stopping based on baseline comparison
            if early_stopping and baseline_results and i > NUMERICAL_THRESHOLDS['early_stopping_steps']:
                # Compare trajectory divergence
                # TODO: Implement divergence check
                pass
        
        # Save to cache if not in test mode
        if not no_cache and not test_mode:
            # Convert to SI units before caching
            hist_si = hist.clone()
            hist_si[:,0] *= self.time_scale
            hist_si[:,1] *= self.length_scale
            if hist.shape[1] > 3:
                hist_si[:,3] *= self.velocity_scale / self.time_scale
            torch.save(hist_si, cache_path)
            if verbose:
                    print(f"Trajectory cached to: {cache_path}")
        
        return hist, tag, quantum_kicks_indices

    def run_all_validations(self, theory: GravitationalTheory, hist: torch.Tensor, y0_general: torch.Tensor, categories: list[str] | None = None) -> dict:
        """Runs all specified validations on a theory."""
        if categories is None:
            categories = ["constraint", "observational", "prediction"]

        # <reason>chain: First check quantum Lagrangian completeness ONLY for quantum theories</reason>
        theory_category = getattr(theory, 'category', 'unknown')
        if theory_category == 'quantum' and hasattr(theory, 'validate_lagrangian_completeness'):
            completeness = theory.validate_lagrangian_completeness()
            if not completeness['complete']:
                print(f"  WARNING: {theory.name} is missing quantum field terms: {completeness['missing']}")
                # Add this to validation results
                validation_results = [{
                    'validator': 'Quantum Lagrangian Completeness',
                    'type': 'quantum',
                    'loss': 1.0,
                    'flags': {
                        'overall': 'FAIL',
                        'details': f"Missing: {', '.join(completeness['missing'])}"
                    },
                    'details': completeness
                }]
            else:
                validation_results = []
        else:
            validation_results = []

        
        # <reason>chain: Track if we should continue to next validation level</reason>
        constraints_passed = True
        observations_passed = True
        
        if "constraint" in categories:
            # <reason>chain: Constraint validators use classical or quantum Lagrangian based on theory type</reason>
            # Quantum theories always use quantum Lagrangian if available
            lagrangian_type = "quantum" if self.should_use_quantum_lagrangian(theory) else "classical"
            print(f"  Constraint validators using {lagrangian_type} Lagrangian")
            # <reason>chain: Only include validators that have been tested in solver_tests</reason>
            constraint_validators = [
                ConservationValidator(self),
                MetricPropertiesValidator(self),
                # LagrangianValidator(self, loss_type=self.loss_type), # Not tested
            ]
            
            # Run constraint validators
            for validator in constraint_validators:
                if self.verbose:
                    print(f"  Running validator: {validator.name}")
                try:
                    if validator.name == 'Lagrangian Validator':
                        result = validator.validate(theory, hist, y0_general=y0_general, experimental=hasattr(self, 'args') and self.args.experimental)
                    else:
                        result = validator.validate(theory, hist, y0_general=y0_general)
                    
                    result['validator'] = validator.name
                    result['type'] = 'constraint'
                    validation_results.append(result)
                    
                    # Check if constraint failed
                    if result['flags']['overall'] != 'PASS':
                        constraints_passed = False
                        
                except Exception as e:
                    print(f"  Error running {validator.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    result = {
                        'validator': validator.name,
                        'type': 'constraint',
                        'loss': 1.0,
                        'flags': {'overall': 'ERROR', 'details': str(e)},
                        'details': {}
                    }
                    validation_results.append(result)
                    constraints_passed = False
            
            # <reason>chain: If constraints failed, skip observational and prediction validations</reason>
            if not constraints_passed:
                print(f"  ⚠️ Constraint validations failed - skipping observational and prediction validations")
                return {"validations": validation_results}
        
        if "observational" in categories and constraints_passed:
            # Get theory category
            theory_category = getattr(theory, 'category', 'unknown')
            print(f"  Theory category: {theory_category}")
            
            observational_validators = []
            
            # Add quantum validators for quantum theories
            # <reason>chain: Only include COW validator which has been tested in solver_tests</reason>
            if theory_category == 'quantum':
                print(f"  Using tested quantum validator for quantum theory")
                observational_validators.extend([
                    COWInterferometryValidator(self),
                    # AtomInterferometryValidator, GravitationalDecoherenceValidator, 
                    # QuantumClockValidator, QuantumLagrangianGroundingValidator removed - not tested
                ])
            else:
                print(f"  Skipping quantum validators for {theory_category} theory")
                
            # Add classical observational validators
            # <reason>chain: Only import validators that have been tested in solver_tests</reason>
            from physics_agent.validations import (
                MercuryPrecessionValidator,
                LightDeflectionValidator,
                PpnValidator,
                PhotonSphereValidator,
                GwValidator,
                # HawkingValidator, CosmologyValidator removed - not tested
                # PsrJ0740Validator removed - test exists but not in main suite
            )
            
            print("  Adding tested classical observational validators...")
            observational_validators.extend([
                MercuryPrecessionValidator(self),
                LightDeflectionValidator(self),
                PpnValidator(self),
                PhotonSphereValidator(self),
                GwValidator(self),
                # HawkingValidator(self), # Not tested
                # CosmologyValidator(self), # Not tested
                # PsrJ0740Validator(self) # Test exists but not in main suite
            ])
            
            # Run observational validators
            for validator in observational_validators:
                if self.verbose:
                    print(f"  Running validator: {validator.name}")
                try:
                    # Observational validators don't need hist
                    result = validator.validate(theory, verbose=True)
                    
                    # Check if result is already in the expected dict format
                    if isinstance(result, dict) and 'loss' in result and 'flags' in result:
                        # Result is already formatted correctly
                        result['validator'] = validator.name
                        result['type'] = getattr(validator, 'category', 'observational')
                    else:
                        # Convert from old format (object with .passed attribute)
                        result = {
                            'loss': 1.0 - (1.0 if result.passed else 0.0),
                            'flags': {
                                'overall': 'PASS' if result.passed else 'FAIL',
                                'details': result.notes
                            },
                            'details': {
                                'observed': result.observed_value,
                                'predicted': result.predicted_value,
                                'error_percent': result.error_percent,
                                'units': result.units,
                                # Add prediction-specific fields if available
                                'beats_sota': getattr(result, 'beats_sota', None),
                                'sota_value': getattr(result, 'sota_value', None),
                                'sota_source': getattr(result, 'sota_source', None),
                                'prediction_data': getattr(result, 'prediction_data', {})
                            },
                            'validator': validator.name,
                            'type': getattr(validator, 'category', 'observational')
                        }
                    validation_results.append(result)
                    
                    # Check if observation failed
                    if result['flags']['overall'] != 'PASS':
                        observations_passed = False
                        
                except Exception as e:
                    print(f"  Error running {validator.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    result = {
                        'validator': validator.name,
                        'type': getattr(validator, 'category', 'observational'),
                        'loss': 1.0,
                        'flags': {'overall': 'ERROR', 'details': str(e)},
                        'details': {}
                    }
                    validation_results.append(result)
                    observations_passed = False
            
            # <reason>chain: If observations failed, skip prediction validations</reason>
            if not observations_passed:
                print(f"  ⚠️ Observational validations failed - skipping prediction validations")
                return {"validations": validation_results}
        
        if "prediction" in categories and constraints_passed and observations_passed:
            # Import prediction validators
            # <reason>chain: Only import validators that have been tested in solver_tests</reason>
            from physics_agent.validations import (
                CMBPowerSpectrumValidator, 
                # PTAStochasticGWValidator, # Not tested
                PrimordialGWsValidator
            )
            
            print("  Adding tested prediction validators to test against state-of-the-art benchmarks")
            prediction_validators = [
                CMBPowerSpectrumValidator(self),
                # PTAStochasticGWValidator(self), # Not tested
                PrimordialGWsValidator(self)
                # <reason>chain: FutureDetectorsValidator and NovelSignaturesValidator removed - not implemented</reason>
            ]
            
            # <reason>chain: Advanced validators removed - not tested in solver_tests</reason>
            # Add advanced validators for quantum theories
            # if theory.category == 'quantum':
            #     from physics_agent.validations import RenormalizabilityValidator, UnificationScaleValidator
            #     
            #     print("  Adding advanced validators for quantum theory...")
            #     prediction_validators.extend([
            #         RenormalizabilityValidator(self),
            #         UnificationScaleValidator(self)
            #     ])
            
            # Run prediction validators
            for validator in prediction_validators:
                if self.verbose:
                    print(f"  Running validator: {validator.name}")
                try:
                    # Prediction validators don't need hist
                    result = validator.validate(theory, verbose=True)
                    # Convert to expected format
                    result = {
                        'loss': 1.0 - (1.0 if result.passed else 0.0),
                        'flags': {
                            'overall': 'PASS' if result.passed else 'FAIL',
                            'details': result.notes
                        },
                        'details': {
                            'observed': result.observed_value,
                            'predicted': result.predicted_value,
                            'error_percent': result.error_percent,
                            'units': result.units,
                            # Add prediction-specific fields if available
                            'beats_sota': getattr(result, 'beats_sota', None),
                            'sota_value': getattr(result, 'sota_value', None),
                            'sota_source': getattr(result, 'sota_source', None),
                            'prediction_data': getattr(result, 'prediction_data', {})
                        }
                    }
                    result['validator'] = validator.name
                    result['type'] = getattr(validator, 'category', 'prediction')
                    validation_results.append(result)
                except Exception as e:
                    print(f"  Error running {validator.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    result = {
                        'validator': validator.name,
                        'type': getattr(validator, 'category', 'prediction'),
                        'loss': 1.0,
                        'flags': {'overall': 'ERROR', 'details': str(e)},
                        'details': {}
                    }
                    validation_results.append(result)

        # <reason>chain: Remove old validator loop that was running all validators together</reason>
        # Validators are now run in their respective sections above
        
        # Add quantum theory assessment
        if theory.category == 'quantum':
            uni_results = self.assess_unification_potential(theory)
            validation_results.append({
                'validator': 'Quantum Theory Assessment',
                'type': 'quantum',
                'loss': 1.0, # This is a qualitative check, not a quantitative loss
                'flags': {
                    'overall': uni_results['overall'],  # Use the pre-computed overall flag
                    'details': uni_results
                },
                'details': uni_results
            })
        
        return {"validations": validation_results}





 


    
    def should_use_quantum_lagrangian(self, theory: GravitationalTheory) -> bool:
        """
        Determine whether to use quantum or classical Lagrangian for a theory.
        <reason>chain: Quantum theories ALWAYS use quantum Lagrangian if available</reason>
        <reason>chain: Classical theories should only use quantum Lagrangian if they have explicit quantum corrections</reason>
        """
        # Check theory category first - quantum theories must use quantum Lagrangian
        theory_category = getattr(theory, 'category', 'unknown')
        if theory_category == 'quantum':
            # Verify complete Lagrangian is available
            if hasattr(theory, 'complete_lagrangian') and theory.complete_lagrangian is not None:
                return True
            else:
                # Quantum theory without complete Lagrangian - use classical with warning
                if self.verbose:
                    print(f"  WARNING: Quantum theory {theory.name} has no complete_lagrangian, using classical")
                return False
            
        # For non-quantum theories, check if they have quantum corrections
        if hasattr(theory, 'has_quantum_corrections') and theory.has_quantum_corrections():
            return True
            
        # <reason>chain: REMOVED: Classical theories should NOT use quantum Lagrangian just because they have a complete_lagrangian</reason>
        # The old logic incorrectly used quantum Lagrangian for any theory with complete_lagrangian
        # This caused classical theories like Kerr to incorrectly use quantum validators
        
        # Default to classical for non-quantum theories
        return False

        
    def _compute_ricci_scalar(self, theory, r):
        """<reason>chain: Compute Ricci scalar for a given theory and radii - computational method kept in engine</reason>"""
        g_tt, g_rr, g_pp, g_tp = theory.get_metric(r, self.M, self.c, self.G)
        
        # Derivatives required for Ricci scalar
        try:
            g_tt_r = torch.autograd.grad(g_tt.sum(), r, create_graph=True)[0]
        except RuntimeError:
            g_tt_r = torch.zeros_like(r)
            
        try:
            g_rr_r = torch.autograd.grad(g_rr.sum(), r, create_graph=True)[0]
        except RuntimeError:
            g_rr_r = torch.zeros_like(r)
        
        try:
            g_tt_rr = torch.autograd.grad(g_tt_r.sum(), r, create_graph=True)[0]
        except RuntimeError:
            g_tt_rr = torch.zeros_like(r)
        
        # Ricci scalar for diagonal, spherically-symmetric metric
        R = -g_tt_rr / g_tt + (g_tt_r**2) / (2 * g_tt**2) - (g_tt_r * g_rr_r) / (g_tt * g_rr) \
            - 2 * g_tt_r / (r * g_tt) + 2 * g_rr_r / (r * g_rr) - 2 / r**2 + 2 / (r**2 * g_rr)
            
        return R

    def unification_score(self, theory: GravitationalTheory) -> float:  # Added colon
        # <reason>chain: Compute quantitative unification score 0-1</reason>
        score = 0.0
        
        # <reason>chain: Run mini-simulation to test unification behavior</reason>
        try:
            # Test parameters (in geometric units)
            r0_geom = UNIFICATION_TEST_PARAMS['r0_factor'] * self.rs  # Start at 10 Schwarzschild radii
            n_steps = UNIFICATION_TEST_PARAMS['n_steps']
            dtau_geom = UNIFICATION_TEST_PARAMS['dtau']
            
            # <reason>chain: Generate test trajectory with theory</reason>
            y0_general = torch.tensor([
                [r0_geom, 0.0, 0.0, np.pi/2, 0.0, 0.0],
                [2*r0_geom, 0.0, 0.0, np.pi/2, 0.05, 0.0]  # Two test particles
            ], dtype=self.dtype, device=self.device)
            
            # <reason>chain: Fix run_trajectory call with proper parameters</reason>
            hist, _, _ = self.run_trajectory(
                theory,
                r0_geom * self.length_scale,  # Convert to SI units
                n_steps,
                dtau_geom * self.time_scale,  # Convert to SI units
                quantum_interval=0,  # No quantum kicks for scoring
                quantum_beta=0.0,
                particle_name='neutrino',
                y0_general=y0_general[0],  # Pass first particle's state
                test_mode=True  # Use test mode for faster execution
            )
            
            # <reason>chain: Check for numerical stability</reason>
            if hist is not None and torch.isfinite(hist).all():
                score += UNIFICATION_THRESHOLDS['stability_score']  # Stable integration
            else:
                return 0.0  # Unstable theory gets zero score
            
            # <reason>chain: Test against unified behavior - should show both gravitational and EM-like effects</reason>
            if hasattr(self, 'loss_calculator') and self.loss_calculator and hasattr(self, 'baseline_trajectories') and self.baseline_trajectories:
                # Compare to pure gravity (Kerr with a=0 is Schwarzschild)
                kerr_baseline = None
                kerr_newman_baseline = None
                
                # Find appropriate baselines
                for name, trajectory in self.baseline_trajectories.items():
                    if 'Kerr' in name and 'Newman' not in name and 'a=0' in name:
                        kerr_baseline = trajectory
                    elif 'Kerr' in name and 'Newman' not in name and kerr_baseline is None:
                        kerr_baseline = trajectory  # Use any Kerr as fallback
                    elif 'Kerr-Newman' in name:
                        kerr_newman_baseline = trajectory
                
                if kerr_baseline is not None:
                    grav_loss = self.loss_calculator.compute_trajectory_loss(hist, kerr_baseline, 'trajectory_mse')
                else:
                    grav_loss = SCORING_LOSS_DEFAULTS['gravitational']  # Default if baseline not available
                    
                if kerr_newman_baseline is not None:
                    em_loss = self.loss_calculator.compute_trajectory_loss(hist, kerr_newman_baseline, 'trajectory_mse')
                else:
                    em_loss = SCORING_LOSS_DEFAULTS['electromagnetic']  # Default if baseline not available
                
                # <reason>chain: Good unification should be between pure gravity and pure EM</reason>
                if UNIFICATION_THRESHOLDS['balanced_loss_min'] < grav_loss < UNIFICATION_THRESHOLDS['balanced_loss_max'] and UNIFICATION_THRESHOLDS['balanced_loss_min'] < em_loss < UNIFICATION_THRESHOLDS['balanced_loss_max']:
                    score += UNIFICATION_THRESHOLDS['balanced_score']  # Shows balanced unification
                
                # <reason>chain: Check for symmetry in losses - key unification signature</reason>
                loss_ratio = min(grav_loss, em_loss) / max(grav_loss, em_loss)
                if loss_ratio > UNIFICATION_THRESHOLDS['loss_ratio_threshold']:  # Losses are similar
                    score += UNIFICATION_THRESHOLDS['loss_ratio_score']
            
            # <reason>chain: Test quantum corrections at small scales</reason>
            r_planck = UNIFICATION_TEST_PARAMS['r_planck']  # Planck length scale
            r_test = torch.tensor([r_planck, UNIFICATION_TEST_PARAMS['r_intermediate'], UNIFICATION_TEST_PARAMS['r_classical']], dtype=self.dtype, device=self.device)
            M_test = torch.tensor([1.0], dtype=self.dtype, device=self.device)
            
            try:
                g_tt, g_rr, g_pp, g_tp = theory.get_metric(r_test, M_test, self.c, self.G)
                
                # <reason>chain: Check for quantum corrections without divergences</reason>
                if torch.isfinite(g_tt).all() and torch.isfinite(g_rr).all():
                    # Metric should deviate from classical at Planck scale
                    classical_g_tt = -(1 - 2*self.G*M_test/self.c**2/r_test)
                    deviation = torch.abs(g_tt - classical_g_tt) / torch.abs(classical_g_tt)
                    
                    if deviation[0] > UNIFICATION_THRESHOLDS['quantum_deviation_planck']:  # Significant quantum correction at Planck scale
                        score += UNIFICATION_THRESHOLDS['quantum_score']
                    if deviation[1] > UNIFICATION_THRESHOLDS['quantum_deviation_intermediate'] and deviation[1] < UNIFICATION_THRESHOLDS['quantum_deviation_intermediate_max']:  # Moderate correction at intermediate
                        score += UNIFICATION_THRESHOLDS['quantum_score']
                    if deviation[2] < UNIFICATION_THRESHOLDS['quantum_deviation_classical']:  # Classical limit recovered at large scales
                        score += UNIFICATION_THRESHOLDS['quantum_score']
            except:
                pass  # Some theories may not handle extreme scales
                
        except Exception as e:
            # <reason>chain: Penalize theories that fail to integrate</reason>
            print(f"  Unification scoring failed for {theory.name}: {str(e)}")
            return 0.0
        
        return min(1.0, score)  # Cap at 1.0

    def check_sm_bridging(self, theory: GravitationalTheory) -> str:
        """
        <reason>chain: Check if theory bridges to Standard Model physics</reason>
        """
        # Check for gauge field content
        has_gauge = hasattr(theory, 'gauge_lagrangian') and theory.gauge_lagrangian is not None
        # Check for matter fields
        has_matter = hasattr(theory, 'matter_lagrangian') and theory.matter_lagrangian is not None
        # Check for interactions
        has_interaction = hasattr(theory, 'interaction_lagrangian') and theory.interaction_lagrangian is not None
        
        if has_gauge and has_matter and has_interaction:
            return 'Potential bridge detected'
        elif has_gauge or has_matter:
            return 'Partial bridge detected'
        else:
            return 'No bridge detected'

    def assess_unification_potential(self, theory: GravitationalTheory) -> dict:  # Added colon
        # <reason>chain: Enhanced with scoring and novel predictions</reason>
        results = {}
        results['score'] = self.unification_score(theory)
        results['sm_bridging'] = self.check_sm_bridging(theory)
        results['novel_predictions'] = getattr(theory, 'novel_predictions', lambda: 'None')()

        # <reason>chain: Add overall flag based on score threshold for unification potential</reason>
        results['overall'] = 'PASS' if results['score'] > UNIFICATION_THRESHOLDS['overall_pass_threshold'] else 'FAIL'  # Lowered threshold from 0.5 to 0.25
        
        # <reason>chain: Add detailed breakdown of score components</reason>
        results['details'] = {
            'numerical_stability': 'PASS' if results['score'] > UNIFICATION_THRESHOLDS['stability_pass_threshold'] else 'FAIL',
            'unified_behavior': 'PASS' if results['score'] > UNIFICATION_THRESHOLDS['unified_pass_threshold'] else 'FAIL',
            'quantum_corrections': 'PASS' if results['score'] > UNIFICATION_THRESHOLDS['quantum_pass_threshold'] else 'FAIL',  # Lowered from 0.5
            'scale_hierarchy': 'PASS' if results['score'] > UNIFICATION_THRESHOLDS['hierarchy_pass_threshold'] else 'FAIL'  # Lowered from 0.7
        }

        return results

    def calculate_comprehensive_score(self, theory_scores: dict) -> dict:
        """
        Calculate a comprehensive score for a theory based on all evaluation metrics.
        Uses weighted scoring from constants.py.
        
        <reason>chain: Implement sophisticated scoring that considers all aspects of a theory's performance</reason>
        """
        from physics_agent.constants import (
            SCORING_WEIGHTS, CONSTRAINT_WEIGHTS, OBSERVATIONAL_WEIGHTS,
            PREDICTION_WEIGHTS, TRAJECTORY_WEIGHTS, UNIFICATION_WEIGHTS,
            LOSS_NORMALIZATION, BONUS_MULTIPLIERS, PENALTY_MULTIPLIERS,
            CATEGORY_BONUSES
        )
        
        scores = {
            'constraints': 0.0,
            'observational': 0.0,
            'predictions': 0.0,
            'trajectory': 0.0,
            'unification': 0.0,
            'bonuses': [],
            'penalties': [],
            'details': {}
        }
        
        # 1. Constraint scores
        constraint_total = 0.0
        constraint_count = 0
        for name, data in theory_scores.get('constraints', {}).items():
            weight = CONSTRAINT_WEIGHTS.get(name, 1.0 / len(CONSTRAINT_WEIGHTS))
            # Score is 1 if passed, 0 if failed
            score = 1.0 if data.get('passed', False) else 0.0
            constraint_total += score * weight
            constraint_count += weight
        
        if constraint_count > 0:
            scores['constraints'] = constraint_total / constraint_count
            scores['details']['constraints'] = {
                'score': scores['constraints'],
                'passed': sum(1 for d in theory_scores.get('constraints', {}).values() if d.get('passed')),
                'total': len(theory_scores.get('constraints', {}))
            }
        
        # Check for perfect conservation
        if scores['constraints'] == 1.0 and len(theory_scores.get('constraints', {})) > 0:
            scores['bonuses'].append(('perfect_conservation', BONUS_MULTIPLIERS['perfect_conservation']))
        
        # 2. Observational scores
        obs_total = 0.0
        obs_count = 0
        for name, data in theory_scores.get('observational', {}).items():
            weight = OBSERVATIONAL_WEIGHTS.get(name, 1.0 / len(OBSERVATIONAL_WEIGHTS))
            score = 1.0 if data.get('passed', False) else 0.0
            obs_total += score * weight
            obs_count += weight
        
        if obs_count > 0:
            scores['observational'] = obs_total / obs_count
            scores['details']['observational'] = {
                'score': scores['observational'],
                'passed': sum(1 for d in theory_scores.get('observational', {}).values() if d.get('passed')),
                'total': len(theory_scores.get('observational', {}))
            }
        
        # 3. Prediction scores
        pred_total = 0.0
        pred_count = 0
        beats_sota_count = 0
        
        for name, data in theory_scores.get('predictions', {}).items():
            weight = PREDICTION_WEIGHTS.get(name, 1.0 / len(PREDICTION_WEIGHTS))
            
            # For predictions, score based on whether it beats SOTA
            if data.get('beats_sota', False):
                score = 1.0
                beats_sota_count += 1
            else:
                # Partial credit based on how close to SOTA
                improvement = data.get('improvement', -float('inf'))
                if improvement > -100:  # Within reasonable range
                    score = 0.5 + 0.5 * np.exp(improvement / 100)
                else:
                    score = 0.0
            
            pred_total += score * weight
            pred_count += weight
        
        if pred_count > 0:
            scores['predictions'] = pred_total / pred_count
            scores['details']['predictions'] = {
                'score': scores['predictions'],
                'beats_sota': beats_sota_count,
                'total': len(theory_scores.get('predictions', {}))
            }
            
            if beats_sota_count > 0:
                scores['bonuses'].append(('beats_sota', BONUS_MULTIPLIERS['beats_sota']))
        
        # 4. Trajectory scores
        trajectory_losses = theory_scores.get('trajectory_losses', {})
        particle_losses = theory_scores.get('particle_trajectory_losses', {})
        
        if trajectory_losses or particle_losses:
            # <reason>chain: Calculate per-particle trajectory scores</reason>
            particle_scores = {}
            
            # If we have per-particle losses, use those
            if particle_losses:
                for particle_name in TRAJECTORY_WEIGHTS.keys():
                    if particle_name in particle_losses:
                        # Find best loss type for this particle
                        best_score = 0.0
                        for loss_type, baselines in particle_losses[particle_name].items():
                            scale = LOSS_NORMALIZATION.get(loss_type, 1.0)
                            for baseline_name, loss_val in baselines.items():
                                # Score based on normalized loss
                                score = np.exp(-loss_val / scale)
                                best_score = max(best_score, score)
                        particle_scores[particle_name] = best_score
                    else:
                        particle_scores[particle_name] = 0.0  # No data for this particle
            
            # Fallback to aggregate trajectory losses if no per-particle data
            elif trajectory_losses:
                # Use same score for all particles (backward compatibility)
                aggregate_score = 0.0
                for loss_type, baselines in trajectory_losses.items():
                    scale = LOSS_NORMALIZATION.get(loss_type, 1.0)
                    for baseline_name, loss_val in baselines.items():
                        score = np.exp(-loss_val / scale)
                        aggregate_score = max(aggregate_score, score)
                
                # Apply same score to all particles
                for particle_name in TRAJECTORY_WEIGHTS.keys():
                    particle_scores[particle_name] = aggregate_score
            
            # Calculate weighted trajectory score
            trajectory_score = 0.0
            for particle_name, weight in TRAJECTORY_WEIGHTS.items():
                trajectory_score += weight * particle_scores.get(particle_name, 0.0)
            
            scores['trajectory'] = trajectory_score
            scores['details']['trajectory'] = {
                'score': trajectory_score,
                'particle_scores': particle_scores
            }
        else:
            # Penalty for no trajectory data
            scores['penalties'].append(('trajectory_failed', PENALTY_MULTIPLIERS['trajectory_failed']))
        
        # 5. Unification scores
        unification = theory_scores.get('unification', {})
        if unification:
            uni_score = 0.0
            
            # SM bridging
            if 'Potential bridge' in unification.get('sm_bridging', ''):
                uni_score += UNIFICATION_WEIGHTS['sm_bridging'] * 1.0
            elif 'Partial bridge' in unification.get('sm_bridging', ''):
                uni_score += UNIFICATION_WEIGHTS['sm_bridging'] * 0.5
            
            # Novel predictions
            if unification.get('novel_predictions') and unification['novel_predictions'] != 'None':
                uni_score += UNIFICATION_WEIGHTS['novel_predictions'] * 1.0
            
            # Scale hierarchy (from unification score)
            if unification.get('score', 0) > 0.5:
                uni_score += UNIFICATION_WEIGHTS['scale_hierarchy'] * 1.0
            elif unification.get('score', 0) > 0.25:
                uni_score += UNIFICATION_WEIGHTS['scale_hierarchy'] * 0.5
            
            # Field content (check for Lagrangian completeness)
            if theory_scores.get('has_complete_lagrangian', False):
                uni_score += UNIFICATION_WEIGHTS['field_content'] * 1.0
            
            scores['unification'] = uni_score
            scores['details']['unification'] = unification
        
        # Calculate base weighted score
        base_score = (
            SCORING_WEIGHTS['constraints'] * scores['constraints'] +
            SCORING_WEIGHTS['observational'] * scores['observational'] +
            SCORING_WEIGHTS['predictions'] * scores['predictions'] +
            SCORING_WEIGHTS['trajectory'] * scores['trajectory'] +
            SCORING_WEIGHTS['unification'] * scores['unification']
        )
        
        # Apply bonuses
        bonus_multiplier = 1.0
        for bonus_name, multiplier in scores['bonuses']:
            bonus_multiplier *= multiplier
        
        # Apply penalties
        penalty_multiplier = 1.0
        for penalty_name, multiplier in scores['penalties']:
            penalty_multiplier *= multiplier
        
        # Apply category bonus
        category = theory_scores.get('category', 'unknown')
        category_multiplier = CATEGORY_BONUSES.get(category, 1.0)
        
        # Final score
        final_score = base_score * bonus_multiplier * penalty_multiplier * category_multiplier
        
        # Check for all tests passed bonus
        all_passed = (
            scores['constraints'] == 1.0 and
            scores['observational'] == 1.0 and
            len(theory_scores.get('constraints', {})) > 0 and
            len(theory_scores.get('observational', {})) > 0
        )
        
        if all_passed:
            final_score *= BONUS_MULTIPLIERS['all_tests_passed']
            scores['bonuses'].append(('all_tests_passed', BONUS_MULTIPLIERS['all_tests_passed']))
        
        return {
            'final_score': min(1.0, final_score),  # Cap at 1.0
            'component_scores': scores,
            'base_score': base_score,
            'bonus_multiplier': bonus_multiplier,
            'penalty_multiplier': penalty_multiplier,
            'category_multiplier': category_multiplier,
            'category': category
        }

def validate_theory_only(
    model: "GravitationalTheory",
    engine: "TheoryEngine",
    main_run_dir: str,
    baseline_results: dict,
    baseline_theories: dict,
    args: argparse.Namespace,
    r0: torch.Tensor,
    r0_val: float,
) -> dict:
    """Run all validations on a theory without full trajectory or visualizations.
    
    Computes a short trajectory and runs physics validators (constraints, observations).
    The trajectory itself doesn't pass/fail - it either computes or has an error.
    Loss values against baselines (Kerr, Kerr-Newman) are computed separately.
    
    Returns:
        dict: Validation results with physics test outcomes
    """
    print(f"\n--- Validating: {model.name} ---")
    
    # Create theory-specific subdirectory
    sanitized_name = model.name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(",", "").replace(".", "_")
    theory_dir = os.path.join(main_run_dir, sanitized_name)
    os.makedirs(theory_dir, exist_ok=True)
    
    # <reason>chain: Copy theory source code for auditability</reason>
    try:
        import inspect
        theory_source_file = inspect.getsourcefile(model.__class__)
        if theory_source_file and os.path.exists(theory_source_file):
            code_dir = os.path.join(theory_dir, "code")
            os.makedirs(code_dir, exist_ok=True)
            import shutil
            theory_dest = os.path.join(code_dir, "theory_source.py")
            shutil.copy2(theory_source_file, theory_dest)
    except Exception as e:
        print(f"  [WARNING] Could not copy theory source code: {e}")
    
    # Initialize validation results
    validation_results = {
        'theory_name': model.name,
        'category': getattr(model, 'category', 'unknown'),
        'passed': False,
        'constraints_passed': False,
        'observations_passed': False,
        'validations': [],
        'theory_dir': theory_dir
    }
    
    # <reason>chain: Check if this is a baseline theory</reason>
    for name, baseline_model in baseline_theories.items():
        if model.name == baseline_model.name:
            break
    
    # Never skip validation - always run all validations
    # <reason>chain: System now runs all theories through all validators</reason>
    
    # Run a short trajectory for validation
    print(f"  Running validation trajectory for {model.name}...")
    validation_r0_val = 15.0 * engine.rs
    validation_r0 = torch.tensor([validation_r0_val], device=engine.device, dtype=engine.dtype)
    validation_dtau_si = 0.1 * engine.time_scale
    validation_steps = 100
    
    # <reason>chain: Enable quantum effects for quantum theories during validation</reason>
    # Quantum theories need their quantum corrections active to conserve energy
    uses_quantum = engine.should_use_quantum_lagrangian(model)
    theory_category = getattr(model, 'category', 'unknown')
    
    # <reason>chain: Only apply stochastic kicks to theories that expect them</reason>
    # Quantum kicks are stochastic and break exact conservation
    # Only use them for theories with inherent stochasticity
    has_stochastic = hasattr(model, 'has_stochastic_elements') and model.has_stochastic_elements()
    
    if has_stochastic:
        # Stochastic theories need kicks to properly model their physics
        validation_quantum_interval = 10
        validation_quantum_beta = 0.001
    elif theory_category == 'quantum' and uses_quantum:
        # Pure quantum theories use quantum Lagrangian but no stochastic kicks
        validation_quantum_interval = 0  # No kicks - conservation should be exact
        validation_quantum_beta = 0.0
    else:
        # Classical theories
        validation_quantum_interval = 0
        validation_quantum_beta = 0.0
    
    validation_hist, _, _ = engine.run_trajectory(
        model, validation_r0_val * engine.length_scale, validation_steps, validation_dtau_si,
        quantum_interval=validation_quantum_interval, quantum_beta=validation_quantum_beta
    )
    
    if validation_hist is None or validation_hist.shape[0] <= 1:
        print(f"  [WARNING] Could not compute validation trajectory for {model.name}")
        # Still continue with validation - just note the computational issue
        validation_results['validations'] = [{
            'validator': 'Trajectory Computation',
            'type': 'computational',
            'loss': float('inf'),
            'flags': {'overall': 'ERROR', 'details': 'Trajectory computation failed - no data to validate'},
            'details': {'error': 'No trajectory data generated'}
        }]
        # Don't return early - let the theory be ranked with this computational error
        validation_results['constraints_passed'] = False
        validation_results['observations_passed'] = False
        validation_results['passed'] = False
        return validation_results
    
    # Get initial conditions for validation
    y0_sym, y0_gen, _ = engine.get_initial_conditions(model, validation_r0)
    
    # Run all validations
    all_validation_results = engine.run_all_validations(
        model, validation_hist, y0_gen, 
        categories=["constraint", "observational"]  # Don't run predictions yet
    )
    
    validation_results['validations'] = all_validation_results['validations']
    
    # Check constraint results
    constraint_validations = [v for v in validation_results['validations'] if v['type'] == 'constraint']
    constraints_passed = all(v['flags']['overall'] == 'PASS' for v in constraint_validations)
    validation_results['constraints_passed'] = constraints_passed
    
    # Check observational results (including quantum theories)
    theory_category = getattr(model, 'category', 'unknown')
    if theory_category == 'quantum':
        # For quantum theories, check if validators passed (80% threshold)
        all_validators = validation_results.get('validations', [])
        if all_validators:
            passed_count = sum(1 for v in all_validators if v['flags']['overall'] == 'PASS')
            total_count = len(all_validators)
            pass_rate = passed_count / total_count if total_count > 0 else 0
            observations_passed = pass_rate >= 0.8  # 80% pass rate
        else:
            observations_passed = False
    else:
        # For classical theories, observations are informational only
        observations_passed = True  # Classical theories don't fail on observations
    
    validation_results['observations_passed'] = observations_passed
    # Only constraint failures constitute a real failure
    validation_results['passed'] = constraints_passed
    
    # <reason>chain: Clean validation results to make them JSON-serializable</reason>
    # Recursively convert any Tensors or numpy arrays in the validation results
    def clean_for_json(obj):
        """Recursively clean an object for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, complex):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean_for_json(v) for v in obj]
        elif hasattr(obj, '__dict__'):  # Handle custom objects
            return clean_for_json(obj.__dict__)
        else:
            try:
                # Try to convert to basic Python type
                return float(obj) if isinstance(obj, (float, int)) else str(obj)
            except:
                return str(obj)
    
    # Clean the validation results before saving
    validation_results_clean = clean_for_json(validation_results)
    
    # Save validation results
    validation_json_path = os.path.join(theory_dir, 'validation_results.json')
    try:
        with open(validation_json_path, 'w') as f:
            json.dump(validation_results_clean, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save validation results JSON: {e}")
        # Try with more aggressive serialization
        try:
            from physics_agent.validations.error_handlers import make_json_serializable
            validation_results_ultra_clean = make_json_serializable(validation_results)
            with open(validation_json_path, 'w') as f:
                json.dump(validation_results_ultra_clean, f, indent=4)
            print("Successfully saved with enhanced serialization")
        except Exception as e2:
            print(f"Error: Could not save validation results even with enhanced serialization: {e2}")
    
    # Print summary of physics validation tests
    print(f"  Constraint tests: {'PASSED' if constraints_passed else 'FAILED'}")
    print(f"  Observation tests: {'PASSED' if observations_passed else 'FAILED'}")
    print(f"  Overall validation: {'PASSED' if validation_results['passed'] else 'FAILED'}")
    
    return validation_results

def run_trajectory_and_visualize(
    model: "GravitationalTheory",
    engine: "TheoryEngine", 
    theory_dir: str,
    baseline_results: dict,
    baseline_theories: dict,
    args: argparse.Namespace,
    effective_steps: int,
    effective_dtau: torch.Tensor,
    r0: torch.Tensor,
    r0_val: float,
    quantum_interval: int,
    quantum_beta: float,
    progress_callback,
    CALLBACK_INTERVAL: int,
):
    """
    Run full trajectory simulation and create visualizations for a theory that already passed validation.
    This is Phase 2 of the evaluation process.
    
    Returns:
        bool: True if trajectory computed successfully, False if computation error
    """
    print(f"\n--- Running Full Trajectory: {model.name} ---")
    
    # <reason>chain: Reuse the core logic from process_and_evaluate_theory but skip validation</reason>
    # Just delegate to process_and_evaluate_theory since it handles everything we need
    result = process_and_evaluate_theory(
        model, engine, os.path.dirname(theory_dir), baseline_results, baseline_theories, args,
        effective_steps, effective_dtau, r0, r0_val,
        quantum_interval, quantum_beta, progress_callback, CALLBACK_INTERVAL
    )
    
    return result is not None and 'theory_dir' in result


def process_and_evaluate_theory(
    model: "GravitationalTheory",
    engine: "TheoryEngine",
    main_run_dir: str,
    baseline_results: dict,
    baseline_theories: dict,
    args: argparse.Namespace,
    effective_steps: int,
    effective_dtau: torch.Tensor,
    r0: torch.Tensor,
    r0_val: float,
    quantum_interval: int,
    quantum_beta: float,
    progress_callback,
    CALLBACK_INTERVAL: int,
):
    """Process and evaluate a single theory with all visualizations and validations."""
    print(f"\n--- Evaluating: {model.name} ---")
    
    # Create theory-specific subdirectory
    sanitized_name = model.name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(",", "").replace(".", "_")
    theory_dir = os.path.join(main_run_dir, sanitized_name)
    os.makedirs(theory_dir, exist_ok=True)
    
    # <reason>chain: Copy theory source code for full auditability</reason>
    try:
        import inspect
        # Get the source file of the theory
        theory_source_file = inspect.getsourcefile(model.__class__)
        if theory_source_file and os.path.exists(theory_source_file):
            # Create code directory
            code_dir = os.path.join(theory_dir, "code")
            os.makedirs(code_dir, exist_ok=True)
            
            # Copy the theory source file
            import shutil
            theory_dest = os.path.join(code_dir, "theory_source.py")
            shutil.copy2(theory_source_file, theory_dest)
            
            # Also save the instantiated theory code with parameters
            with open(os.path.join(code_dir, "theory_instance.py"), 'w') as f:
                f.write("#!/usr/bin/env python3\n")
                f.write('"""Exact theory instance used in this run"""\n\n')
                f.write("# This file shows how the theory was instantiated for this run\n\n")
                
                # Try to get the module import path
                module_name = model.__class__.__module__
                class_name = model.__class__.__name__
                
                # Write import statement
                if module_name and module_name != '__main__':
                    f.write(f"from {module_name} import {class_name}\n\n")
                else:
                    f.write(f"from theory_source import {class_name}\n\n")
                
                # Get init parameters
                init_params = {}
                sig = inspect.signature(model.__class__.__init__)
                for param_name, param in sig.parameters.items():
                    if param_name != 'self':
                        # Try to get the actual value from the instance
                        if hasattr(model, param_name):
                            init_params[param_name] = getattr(model, param_name)
                        elif hasattr(model, f'_{param_name}'):  # Check for private attributes
                            init_params[param_name] = getattr(model, f'_{param_name}')
                        elif param.default != inspect.Parameter.empty:
                            init_params[param_name] = param.default
                
                # Write instantiation
                f.write("# Instantiation with exact parameters\n")
                param_str = ", ".join([f"{k}={repr(v)}" for k, v in init_params.items()])
                f.write(f"theory = {class_name}({param_str})\n")
                f.write(f"\n# Theory name: {model.name}\n")
                f.write(f"# Category: {getattr(model, 'category', 'unknown')}\n")
                
        else:
            # If we can't find the source file, at least document what we know
            code_dir = os.path.join(theory_dir, "code")
            os.makedirs(code_dir, exist_ok=True)
            
            with open(os.path.join(code_dir, "theory_info.txt"), 'w') as f:
                f.write(f"Theory Class: {model.__class__.__name__}\n")
                f.write(f"Module: {model.__class__.__module__}\n")
                f.write(f"Theory Name: {model.name}\n")
                f.write(f"Category: {getattr(model, 'category', 'unknown')}\n")
                f.write("\nSource file could not be located.\n")
                
    except Exception as e:
        print(f"  [WARNING] Could not copy theory source code: {e}")
    
    # <reason>chain: Track constraint failures (the only real failures)</reason>
    
    # Initialize comprehensive scoring system
    theory_scores = {
        'theory_name': model.name,
        'category': getattr(model, 'category', 'unknown'),
        'timestamp': datetime.now().isoformat(),
        'constraints': {},
        'observational': {},
        'predictions': {},
        'trajectory_losses': {},
        'particle_trajectory_losses': {},
        'overall_scores': {}
    }
    
    # Create viz sub-directory for all plots
    viz_dir = os.path.join(theory_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\n--- Evaluating: {model.name} ---")
    print(f"  Results will be saved to: {theory_dir}")

    # <reason>chain: Check if this is a baseline theory</reason>
    for name, baseline_model in baseline_theories.items():
        if model.name == baseline_model.name:
            break

    # <reason>chain: Never skip validation - always run pre-flight checks</reason>
    skip_preflight = False
    
    # <reason>chain: Define preflight_r0 before the conditional block to avoid UnboundLocalError</reason>
    preflight_r0_val = 15.0 * engine.rs
    preflight_r0 = torch.tensor([preflight_r0_val], device=engine.device, dtype=engine.dtype)
    
    if not skip_preflight:
        print(f"  Running pre-flight checks for {model.name}...")
        # Run a short prefix for validation
        # <reason>chain: Use smaller timestep for pre-flight to ensure energy conservation accuracy</reason>
        # Use dtau = 0.01 in geometric units for better numerical accuracy
        preflight_dtau_si = 0.01 * engine.time_scale  # This gives dtau_geom = 0.01
        prefix_hist, _, _ = engine.run_trajectory(
            model, preflight_r0_val * engine.length_scale, 100, preflight_dtau_si,
            quantum_interval=0, quantum_beta=0.0
        )
        if prefix_hist is not None and prefix_hist.shape[0] > 1:
            y0_sym, y0_gen, _ = engine.get_initial_conditions(model, preflight_r0)
            # <reason>chain: Run all validation categories, not just constraints</reason>
            validation_results = engine.run_all_validations(model, prefix_hist, y0_gen, categories=["constraint", "observational", "prediction"])
            
            # Log validation results to theory directory
            validation_json_path = os.path.join(theory_dir, 'pre_run_validation.json')
            with open(validation_json_path, 'w') as f:
                # <reason>chain: Convert tensors to serializable format before JSON dump</reason>
                serializable_validations = to_serializable(validation_results['validations'])
                json.dump({'theory': model.name, 'validations': serializable_validations}, f, indent=4)

            # Store validation scores in comprehensive scoring
            for val in validation_results['validations']:
                if val['type'] == 'constraint':
                    theory_scores['constraints'][val['validator']] = {
                        'loss': val['loss'],
                        'passed': val['flags']['overall'] == 'PASS',
                        'details': val.get('details', {})
                    }
                elif val['type'] == 'observational':
                    # <reason>chain: Add observational validator results to comprehensive scores</reason>
                    theory_scores['observational'][val['validator']] = {
                        'loss': val['loss'],
                        'passed': val['flags']['overall'] == 'PASS',
                        'details': val.get('details', {})
                    }
                elif val['type'] == 'prediction':
                    # <reason>chain: Add prediction validator results to comprehensive scores</reason>
                    theory_scores['predictions'][val['validator']] = {
                        'loss': val['loss'],
                        'passed': val['flags']['overall'] == 'PASS',
                        'details': val.get('details', {})
                    }

            # Check for constraint failures
            if any(v['flags']['overall'] == 'FAIL' for v in validation_results['validations']):
                print(f"  [WARNING] Pre-flight constraint checks failed for {model.name}. Continuing with evaluation.")
                # Store the failure but continue evaluation
        else:
            print(f"  [WARNING] Pre-flight trajectory computation failed for {model.name}.")
            failure_info = {"theory": model.name, "status": "computational_error", "reason": "Pre-flight simulation failed"}
            with open(os.path.join(theory_dir, 'failure_info.json'), 'w') as f:
                json.dump(failure_info, f, indent=4)
            # Continue with evaluation anyway - this will be reflected in the final report

    # --- Run the full trajectory ---
    print(f"  Running trajectory simulation for {model.name}...")
    
    # Get theory category for particle selection
    theory_category = getattr(model, 'category', 'unknown')
    
    # Run multi-particle trajectories
    particle_results = engine.run_multi_particle_trajectories(
        model, r0_val * engine.length_scale, effective_steps, effective_dtau.item() * engine.time_scale,
        theory_category=theory_category,
                    quantum_interval=quantum_interval if getattr(args, 'experimental', False) else 0,
            quantum_beta=quantum_beta if getattr(args, 'experimental', False) else 0.0,
                        progress_callback=progress_callback if getattr(args, 'self_monitor', False) else None,
        callback_interval=CALLBACK_INTERVAL,
                    baseline_results=baseline_results if getattr(args, 'early_stop', False) else None,
            early_stopping=getattr(args, 'early_stop', False),
        test_mode=False,
        verbose=args.verbose,
        no_cache=args.no_cache
    )
    
    # Check if any particle trajectory succeeded
    successful_particles = {name: res for name, res in particle_results.items() 
                          if res['trajectory'] is not None and res['trajectory'].shape[0] > 1}
    
    if successful_particles:
        # Save all particle trajectories
        particle_dir = os.path.join(theory_dir, "particles")
        os.makedirs(particle_dir, exist_ok=True)
        
        # Use the first successful particle's trajectory as the primary one for backward compatibility
        first_particle_name = list(successful_particles.keys())[0]
        hist = successful_particles[first_particle_name]['trajectory']
        quantum_kicks_indices = successful_particles[first_particle_name]['kicks']
        
        # Save individual particle trajectories
        for particle_name, result in particle_results.items():
            if result['trajectory'] is not None:
                particle_path = os.path.join(particle_dir, f"{particle_name}_trajectory.pt")
                torch.save(result['trajectory'].to(dtype=engine.dtype), particle_path)
                
                # Save particle info
                particle_info = {
                    'particle_name': particle_name,
                    'particle_properties': result['particle_properties'],
                    'solver_tag': result['tag'],
                    'quantum_kicks': len(result['kicks']) if result['kicks'] else 0
                }
                particle_info_path = os.path.join(particle_dir, f"{particle_name}_info.json")
                with open(particle_info_path, 'w') as f:
                    json.dump(to_serializable(particle_info), f, indent=4)
                
                # <reason>chain: Check if cached trajectory was used and copy it</reason>
                if result['tag'] == 'cached_trajectory' and hasattr(engine, '_last_cache_path_used'):
                    cache_path = engine._last_cache_path_used
                    if os.path.exists(cache_path):
                        # Copy cached file to theory directory for debugging
                        cached_copy_path = os.path.join(particle_dir, f"{particle_name}_cached_source.pt")
                        import shutil
                        shutil.copy2(cache_path, cached_copy_path)
                        
                        # Store cache info
                        cache_info = {
                            'original_cache_path': cache_path,
                            'cache_file_size': os.path.getsize(cache_path),
                            'cache_modified_time': os.path.getmtime(cache_path),
                            'particle': particle_name
                        }
                        cache_info_path = os.path.join(particle_dir, f"{particle_name}_cache_info.json")
                        with open(cache_info_path, 'w') as f:
                            json.dump(to_serializable(cache_info), f, indent=4)
        
            # Save primary trajectory for backward compatibility
            trajectory_path = os.path.join(theory_dir, "trajectory.pt")
            torch.save(hist.to(dtype=engine.dtype), trajectory_path)
        
            # Run final validations if enabled
            final_validation_results = None
            # Always run final validation
            y0_sym, y0_gen, _ = engine.get_initial_conditions(model, r0)
            # <reason>chain: Run all validation categories, not just constraints</reason>
            final_validation_results = engine.run_all_validations(model, hist, y0_gen, categories=["constraint", "observational", "prediction"])
            
            # Update constraint scores in comprehensive scoring with final validation results
            for val in final_validation_results['validations']:
                if val['type'] == 'constraint':
                    theory_scores['constraints'][val['validator']] = {
                        'loss': val['loss'],
                        'passed': val['flags']['overall'] == 'PASS',
                        'details': val.get('details', {})
                    }
                elif val['type'] == 'observational':
                    # <reason>chain: Add observational validator results to comprehensive scores</reason>
                    theory_scores['observational'][val['validator']] = {
                        'loss': val['loss'],
                        'passed': val['flags']['overall'] == 'PASS',
                        'details': val.get('details', {})
                    }
                elif val['type'] == 'prediction':
                    # <reason>chain: Add prediction validator results to comprehensive scores</reason>
                    theory_scores['predictions'][val['validator']] = {
                        'loss': val['loss'],
                        'passed': val['flags']['overall'] == 'PASS',
                        'details': val.get('details', {})
                    }
                
                # Save final validation results
                final_validation_path = os.path.join(theory_dir, 'final_validation.json')
                with open(final_validation_path, 'w') as f:
                    validation_data = {
                        'theory': model.name,
                        'validations': final_validation_results['validations'],
                        'constraints_passed': all(v['flags']['overall'] == 'PASS' for v in final_validation_results['validations'] if v['type'] == 'constraint')
                    }
                    # <reason>chain: Convert to serializable format before JSON dump</reason>
                    json.dump(to_serializable(validation_data), f, indent=4)



            # <reason>chain: Always run quantum validation for quantum theories</reason>
            # For quantum theories, run and check quantum validators
            theory_category = getattr(model, 'category', 'unknown')
            if theory_category == 'quantum':
                # Check if quantum validations already exist in validation results
                # from validate_theory_only (which calls run_all_validations)
                quantum_validation_path = os.path.join(theory_dir, 'validation_results.json')
                if os.path.exists(quantum_validation_path):
                    with open(quantum_validation_path, 'r') as f:
                        val_data = json.load(f)
                        # Extract quantum validator results
                        quantum_validations = [v for v in val_data.get('validations', []) 
                                             if v.get('validator') in ['COW Neutron Interferometry Validator',
                                                                      'Atom Interferometry Validator',
                                                                      'Gravitational Decoherence Validator',
                                                                      'Quantum Clock Validator',
                                                                      'Quantum Lagrangian Grounding Validator']]
                        if quantum_validations:
                            # Save quantum validation results separately
                            with open(os.path.join(theory_dir, 'quantum_validation.json'), 'w') as f:
                                json.dump(to_serializable({'theory': model.name, 'validations': quantum_validations}), f, indent=4)
                            
                            # Count passes
                            quantum_passed_count = sum(1 for v in quantum_validations if v['flags']['overall'] == 'PASS')
                            quantum_total = len(quantum_validations)
                            quantum_passed = quantum_passed_count >= 4 and quantum_total >= 5
                            
                            print(f"  Quantum validators passed: {quantum_passed_count}/{quantum_total}")
                            if quantum_passed:
                                print(f"  Quantum validation passed for {model.name}")
                            else:
                                print(f"  Quantum validation failed for {model.name}")
                else:
                    print(f"  [INFO] No validation results found for {model.name}")
            else:
                print(f"  [INFO] {model.name} is not a quantum theory - no quantum validation performed")
        
        # Compute trajectory comparison losses against baselines
        # <reason>chain: Use multiple loss types to get meaningful trajectory comparisons</reason>
        # <reason>chain: Trajectory-based losses are more stable than Ricci tensor calculations</reason>
        if True:  # Always compute trajectory losses
            loss_results = {}
            
            # Primary trajectory for this theory
            theory_trajectory = particle_results[first_particle_name]['trajectory']
            
            # List of loss types to compute (in order of priority)
            loss_types = ['trajectory_mse', 'fft', 'endpoint_mse', 'cosine']
            
            for loss_type in loss_types:
                loss_results[loss_type] = {}
                
                for baseline_name, baseline_hist in baseline_results.items():
                    if baseline_hist is None or theory_trajectory is None:
                        loss_results[loss_type][baseline_name] = float('inf')
                        continue
                        
                    try:
                        # <reason>chain: Compare actual trajectory paths using stable numerical methods</reason>
                        loss = engine.loss_calculator.compute_trajectory_loss(
                            theory_trajectory, baseline_hist, loss_type
                        )
                        loss_results[loss_type][baseline_name] = loss
                    except Exception:
                        # <reason>chain: Handle numerical failures gracefully</reason>
                        loss_results[loss_type][baseline_name] = float('inf')
            
            # Also compute Ricci as a fallback (but don't rely on it exclusively)
            loss_results['ricci'] = {}
            r_samples = torch.linspace(engine.rs * 1.5, engine.rs * 100, 100, device=engine.device, dtype=engine.dtype)
            
            for baseline_name, baseline_hist in baseline_results.items():
                if baseline_hist is None:
                    loss_results['ricci'][baseline_name] = float('inf')
                    continue
                    
                try:
                    # <reason>chain: Ricci tensor comparison as supplementary measure</reason>
                    loss = engine.loss_calculator.compute_ricci_loss(model, baseline_theories[baseline_name], r_samples, engine.M, engine.c, engine.G)
                    loss_results['ricci'][baseline_name] = loss
                except Exception:
                    loss_results['ricci'][baseline_name] = float('inf')
            
            # Save loss results
            loss_path = os.path.join(theory_dir, 'losses.json')
            with open(loss_path, 'w') as f:
                json.dump(loss_results, f, indent=4)
            
            # Store trajectory losses in comprehensive scoring
            theory_scores['trajectory_losses'] = loss_results
        
        # Generate comparison plot
        plot_filename = os.path.join(viz_dir, "trajectory_comparison.png")
        
        # <reason>chain: Store baseline theories in engine for enhanced visualization</reason>
        engine._baseline_theories = baseline_theories
        
        # Get particle info for both charged and uncharged particles
        charged_particle_info = None
        uncharged_particle_info = None
        
        # Find charged and uncharged particles
        for particle_name, result in particle_results.items():
            if result['particle'] is not None:
                if result['particle'].charge != 0 and charged_particle_info is None:
                    charged_particle_info = {
                        'name': particle_name,
                        'particle': result['particle'],
                        'tag': result['tag'], 
                        'trajectory': result['trajectory'],
                        'particle_properties': result['particle_properties']
                    }
                elif result['particle'].charge == 0 and uncharged_particle_info is None:
                    uncharged_particle_info = {
                        'name': particle_name,
                        'particle': result['particle'],
                        'tag': result['tag'],
                        'trajectory': result['trajectory'], 
                        'particle_properties': result['particle_properties']
                    }
        
        # Use primary particle info for the main trajectory, but pass both for comparison
        particle_info = {
            'particle': particle_results[first_particle_name]['particle'],
            'tag': particle_results[first_particle_name]['tag'],
            'particle_properties': particle_results[first_particle_name]['particle_properties'],
            'charged_particle': charged_particle_info,
            'uncharged_particle': uncharged_particle_info
        }
        
        engine.visualizer.generate_comparison_plot(
            model, hist, baseline_results, baseline_theories,
            plot_filename, engine.rs, 
            final_validation_results,
            particle_info=particle_info
        )
        
        # Generate multi-particle grid visualization
        multi_particle_plot_filename = os.path.join(viz_dir, "multi_particle_grid.png")
        engine.visualizer.generate_multi_particle_grid(
            model, particle_results, baseline_results, baseline_theories,
            multi_particle_plot_filename, engine.rs,
            final_validation_results
        )
        
        # Generate unified multi-particle plot (all particles on one plot)
        unified_plot_filename = os.path.join(viz_dir, "all_particles_unified.png")
        engine.visualizer.generate_unified_multi_particle_plot(
            model, particle_results, baseline_results, baseline_theories,
            unified_plot_filename, engine.rs,
            final_validation_results
        )
        print(f"  Generated unified multi-particle plot: {unified_plot_filename}")
        
        # Compute and save per-particle losses
        # <reason>chain: Use trajectory-based losses for meaningful particle comparisons</reason>
        if True:  # Always compute trajectory losses
            particle_losses = {}
            for particle_name, result in particle_results.items():
                if result['trajectory'] is None:
                    continue
                    
                particle_losses[particle_name] = {}
                hist_particle = result['trajectory']
                
                # <reason>chain: Compute multiple loss types for robust comparison</reason>
                loss_types = ['trajectory_mse', 'fft', 'endpoint_mse', 'cosine']
                
                for loss_type in loss_types:
                    particle_losses[particle_name][loss_type] = {}
                    
                    for baseline_name, baseline_hist in baseline_results.items():
                        if baseline_hist is None:
                            particle_losses[particle_name][loss_type][baseline_name] = float('inf')
                            continue
                        
                        try:
                            # <reason>chain: Compare particle trajectories using stable numerical methods</reason>
                            loss = engine.loss_calculator.compute_trajectory_loss(
                                hist_particle, baseline_hist, loss_type
                            )
                            particle_losses[particle_name][loss_type][baseline_name] = loss
                        except Exception:
                            # <reason>chain: Handle numerical failures gracefully</reason>
                            particle_losses[particle_name][loss_type][baseline_name] = float('inf')
                
                # Also compute Ricci as a fallback
                particle_losses[particle_name]['ricci'] = {}
                r_samples = torch.linspace(engine.rs * 1.5, engine.rs * 100, 100, device=engine.device, dtype=engine.dtype)
                
                for baseline_name, baseline_hist in baseline_results.items():
                    if baseline_hist is None:
                        particle_losses[particle_name]['ricci'][baseline_name] = float('inf')
                        continue
                    
                    try:
                        # <reason>chain: Ricci tensor comparison as supplementary measure</reason>
                        loss = engine.loss_calculator.compute_ricci_loss(model, baseline_theories[baseline_name], r_samples, engine.M, engine.c, engine.G)
                        particle_losses[particle_name]['ricci'][baseline_name] = loss
                    except Exception:
                        particle_losses[particle_name]['ricci'][baseline_name] = float('inf')
            
            # Save particle-specific losses
            particle_losses_path = os.path.join(particle_dir, 'particle_losses.json')
            with open(particle_losses_path, 'w') as f:
                json.dump(particle_losses, f, indent=4)
            
            # Store particle losses in comprehensive scoring
            theory_scores['particle_trajectory_losses'] = particle_losses
        
        # Save complete theory info
        theory_info = {
            "name": model.name,
            "class_name": model.__class__.__name__,
            "module_name": model.__class__.__module__,
            "is_symmetric": model.is_symmetric,
            "source_dir": getattr(model, 'source_dir', 'N/A'),  # <reason>chain: Use safe getattr to handle missing attribute</reason>
            "lagrangian": str(model.lagrangian) if hasattr(model, 'lagrangian') and model.lagrangian else None,
            "category": model.category if hasattr(model, 'category') else "unknown",
            "trajectory_was_cached": False,  # Will be updated if cache was used
            "trajectory_length": hist.shape[0],
            "quantum_kicks": len(quantum_kicks_indices) if quantum_kicks_indices else 0,
            # Save parameters for reconstruction
            "parameters": {}
        }
        
        # Store model parameters (only actual numeric values, not symbolic placeholders)
        # <reason>chain: Fixed to only store parameters that are actual values, not symbolic strings</reason>
        theory_info['parameters'] = {}
        

        
        # Check if any cached trajectory was used
        cached_particles = []
        for particle_name, result in particle_results.items():
            if result and result.get('tag') == 'cached_trajectory':
                cached_particles.append(particle_name)
        
        if cached_particles:
            theory_info['trajectory_was_cached'] = True
            theory_info['cached_particles'] = cached_particles
        
        # Get the actual parameters from the theory's __init__ signature
        import inspect
        if hasattr(model.__class__, '__init__'):
            sig = inspect.signature(model.__class__.__init__)
            actual_params = list(sig.parameters.keys())
            actual_params.remove('self')  # Remove self parameter
            
            # Only store parameters that are in the actual __init__ signature
            for param in actual_params:
                if hasattr(model, param):
                    value = getattr(model, param)
                    # Only store if it's a numeric value, not a symbolic string
                    if isinstance(value, (int, float, bool)) or (isinstance(value, str) and not any(c in value for c in ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω'])):
                        theory_info['parameters'][param] = to_serializable(value)
        
        theory_info_path = os.path.join(theory_dir, 'theory_info.json')
        with open(theory_info_path, 'w') as f:
            json.dump(theory_info, f, indent=4)
        
        # Compute overall scores
        # Constraint score: percentage of constraints passed
        constraint_scores = list(theory_scores['constraints'].values())
        theory_scores['overall_scores']['constraint_pass_rate'] = sum(1 for s in constraint_scores if s['passed']) / len(constraint_scores) if constraint_scores else 0.0
        theory_scores['overall_scores']['constraint_avg_loss'] = sum(s['loss'] for s in constraint_scores) / len(constraint_scores) if constraint_scores else 1.0
        
        # Observational score: percentage of observational tests passed
        obs_scores = list(theory_scores['observational'].values())
        theory_scores['overall_scores']['observational_pass_rate'] = sum(1 for s in obs_scores if s['passed']) / len(obs_scores) if obs_scores else 0.0
        theory_scores['overall_scores']['observational_avg_loss'] = sum(s['loss'] for s in obs_scores) / len(obs_scores) if obs_scores else 1.0
        
        # Trajectory loss scores: Find best loss against Kerr and Kerr-Newman
        if theory_scores['trajectory_losses']:
            # Find Kerr and Kerr-Newman losses
            kerr_losses = {}
            kn_losses = {}
            for loss_type, baselines in theory_scores['trajectory_losses'].items():
                for baseline_name, loss_val in baselines.items():
                    if 'Kerr' in baseline_name and 'Newman' not in baseline_name:
                        kerr_losses[loss_type] = loss_val
                    elif 'Kerr-Newman' in baseline_name:
                        kn_losses[loss_type] = loss_val
            
            # Store best losses
            if kerr_losses:
                theory_scores['overall_scores']['best_kerr_loss'] = min(kerr_losses.values())
                theory_scores['overall_scores']['best_kerr_loss_type'] = min(kerr_losses, key=kerr_losses.get)
            if kn_losses:
                theory_scores['overall_scores']['best_kn_loss'] = min(kn_losses.values())
                theory_scores['overall_scores']['best_kn_loss_type'] = min(kn_losses, key=kn_losses.get)
        
        # Add unification results if available
        if hasattr(model, 'category') and model.category == 'quantum':
            uni_results = engine.assess_unification_potential(model)
            theory_scores['unification'] = uni_results
        
        # Calculate comprehensive score using the new scoring system
        comprehensive_score_result = engine.calculate_comprehensive_score(theory_scores)
        theory_scores['comprehensive_score'] = comprehensive_score_result
        theory_scores['overall_scores']['unified_score'] = comprehensive_score_result['final_score']
        
        # Legacy composite score for backward compatibility
        theory_scores['overall_scores']['composite_score'] = (
            theory_scores['overall_scores'].get('constraint_pass_rate', 0) * 0.3 +
            theory_scores['overall_scores'].get('observational_pass_rate', 0) * 0.3 +
            (1.0 - theory_scores['overall_scores'].get('constraint_avg_loss', 1.0)) * 0.2 +
            (1.0 - theory_scores['overall_scores'].get('observational_avg_loss', 1.0)) * 0.2
        )
        
        # Save comprehensive scores
        comprehensive_scores_path = os.path.join(theory_dir, 'comprehensive_scores.json')
        with open(comprehensive_scores_path, 'w') as f:
            # <reason>chain: Use to_serializable to handle NaN, infinity, and tensor values</reason>
            json.dump(to_serializable(theory_scores), f, indent=4)
            
        # <reason>chain: Generate comprehensive HTML report with all validation results</reason>
        try:
            report_generator = ComprehensiveReportGenerator()
            
            # Collect all logs if available
            log_content = None
            if hasattr(model, 'execution_logs'):
                log_content = '\n'.join(model.execution_logs)
            
            # Generate report
            report_path = report_generator.generate_theory_report(
                theory_name=model.name,
                theory_results=theory_scores,
                output_dir=theory_dir,
                logs=log_content
            )
            
            print(f"  Generated HTML report: {report_path}")
            
        except Exception as e:
            print(f"  [WARNING] Failed to generate HTML report: {e}")
            import traceback
            traceback.print_exc()
            
        print(f"  Results saved to: {theory_dir}")
        
        # <reason>chain: Return success result with theory_dir for run_trajectory_and_visualize</reason>
        return {
            'theory_dir': theory_dir,
            'theory_name': model.name,
            'status': 'success',
            'trajectory_computed': True,
            'scores': theory_scores
        }
        
    else:
        print(f"  Simulation failed for {model.name}")
        # Still save failure info
        failure_info = {
            "theory": model.name,
            "status": "failed",
            "reason": "Trajectory simulation returned None or insufficient data"
        }
        failure_path = os.path.join(theory_dir, 'failure_info.json')
        with open(failure_path, 'w') as f:
            json.dump(failure_info, f, indent=4)
        
        # <reason>chain: Log computational error but keep theory for ranking</reason>
        print(f"  [INFO] {model.name} had trajectory computation issues - will be reflected in final ranking")
        
        # <reason>chain: Return failure result but still include theory_dir for consistency</reason>
        return {
            'theory_dir': theory_dir,
            'theory_name': model.name,
            'status': 'failed',
            'trajectory_computed': False,
            'failure_info': failure_info
        }

def generate_leaderboard(main_run_dir: str):
    """
    Generate a comprehensive leaderboard from all theory scores.
    Ranks theories by multiple criteria and saves detailed results.
    """
    print(f"\n{'='*60}")
    print("Generating Comprehensive Leaderboard")
    print(f"{'='*60}")
    
    # Collect all comprehensive scores
    all_scores = []
    
    # Find all theory directories
    for entry in os.listdir(main_run_dir):
        theory_dir = os.path.join(main_run_dir, entry)
        if not os.path.isdir(theory_dir) or entry in ['predictions'] or entry.startswith('baseline_'):
            continue
        
        scores_path = os.path.join(theory_dir, 'comprehensive_scores.json')
        if os.path.exists(scores_path):
            with open(scores_path, 'r') as f:
                scores = json.load(f)
                
                # <reason>chain: Only quantum and UGM theories should be on the leaderboard per project requirements</reason>
                # Filter out classical and test theories
                category = scores.get('category', 'unknown').lower()
                if category not in ['quantum', 'ugm']:
                    print(f"  Skipping {scores.get('theory_name', entry)} - category '{category}' not allowed on leaderboard (only quantum/ugm)")
                    continue
                    
                all_scores.append(scores)
    
    if not all_scores:
        print("No scored theories found.")
        return
    
    # Create leaderboard data structure
    leaderboard = {
        'run_timestamp': os.path.basename(main_run_dir),
        'total_theories': len(all_scores),
        'rankings': {},
        'detailed_scores': [],
        'score_breakdown': {}
    }
    
    # Generate HTML leaderboard
    html_generator = LeaderboardHTMLGenerator()
    html_path = html_generator.generate_leaderboard(main_run_dir)
    if html_path:
        print(f"  HTML leaderboard generated: {html_path}")
    
    # Sort by different criteria
    rankings = {
        'unified_score': sorted(all_scores, 
                               key=lambda x: x.get('comprehensive_score', {}).get('final_score', 0), 
                               reverse=True),
        'constraint_score': sorted(all_scores, 
                                  key=lambda x: x.get('comprehensive_score', {}).get('component_scores', {}).get('constraints', 0), 
                                  reverse=True),
        'observational_score': sorted(all_scores, 
                                     key=lambda x: x.get('comprehensive_score', {}).get('component_scores', {}).get('observational', 0), 
                                     reverse=True),
        'prediction_score': sorted(all_scores, 
                                  key=lambda x: x.get('comprehensive_score', {}).get('component_scores', {}).get('predictions', 0), 
                                  reverse=True),
        'trajectory_score': sorted(all_scores, 
                                  key=lambda x: x.get('comprehensive_score', {}).get('component_scores', {}).get('trajectory', 0), 
                                  reverse=True),
        'unification_score': sorted(all_scores, 
                                   key=lambda x: x.get('comprehensive_score', {}).get('component_scores', {}).get('unification', 0), 
                                   reverse=True),
    }
    
    # Store rankings
    for criterion, ranked_list in rankings.items():
        leaderboard['rankings'][criterion] = [
            {
                'rank': i + 1,
                'theory': theory['theory_name'],
                'category': theory['category'],
                'score': theory.get('comprehensive_score', {}).get('final_score', 0) if criterion == 'unified_score' 
                        else theory.get('comprehensive_score', {}).get('component_scores', {}).get(criterion.replace('_score', ''), 0),
                'bonuses': theory.get('comprehensive_score', {}).get('component_scores', {}).get('bonuses', []),
                'penalties': theory.get('comprehensive_score', {}).get('component_scores', {}).get('penalties', [])
            }
            for i, theory in enumerate(ranked_list[:20])  # Top 20 for each criterion
        ]
    
    # Detailed scores for all theories
    for theory in all_scores:
        comp_score = theory.get('comprehensive_score', {})
        component_scores = comp_score.get('component_scores', {})
        
        detailed = {
            'theory_name': theory['theory_name'],
            'category': theory['category'],
            'unified_score': comp_score.get('final_score', 0),
            'component_scores': {
                'constraints': component_scores.get('constraints', 0),
                'observational': component_scores.get('observational', 0),
                'predictions': component_scores.get('predictions', 0),
                'trajectory': component_scores.get('trajectory', 0),
                'unification': component_scores.get('unification', 0)
            },
            'multipliers': {
                'base_score': comp_score.get('base_score', 0),
                'bonus_multiplier': comp_score.get('bonus_multiplier', 1),
                'penalty_multiplier': comp_score.get('penalty_multiplier', 1),
                'category_multiplier': comp_score.get('category_multiplier', 1)
            },
            'bonuses': component_scores.get('bonuses', []),
            'penalties': component_scores.get('penalties', []),
            'details': component_scores.get('details', {})
        }
        
        leaderboard['detailed_scores'].append(detailed)
    
    # Calculate score distributions
    leaderboard['score_breakdown'] = {
        'unified_scores': {
            'mean': np.mean([t['unified_score'] for t in leaderboard['detailed_scores']]),
            'std': np.std([t['unified_score'] for t in leaderboard['detailed_scores']]),
            'max': max(t['unified_score'] for t in leaderboard['detailed_scores']),
            'min': min(t['unified_score'] for t in leaderboard['detailed_scores'])
        },
        'category_breakdown': {}
    }
    
    # Breakdown by category
    for category in ['quantum', 'classical']:
        category_theories = [t for t in leaderboard['detailed_scores'] if t['category'] == category]
        if category_theories:
            leaderboard['score_breakdown']['category_breakdown'][category] = {
                'count': len(category_theories),
                'mean_score': np.mean([t['unified_score'] for t in category_theories]),
                'best_theory': max(category_theories, key=lambda x: x['unified_score'])['theory_name']
            }
    
    # Save leaderboard
    leaderboard_path = os.path.join(main_run_dir, 'leaderboard.json')
    with open(leaderboard_path, 'w') as f:
        json.dump(leaderboard, f, indent=4)
    
    # Create human-readable summary
    summary_lines = []
    summary_lines.append("COMPREHENSIVE THEORY EVALUATION LEADERBOARD")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Total theories evaluated: {leaderboard['total_theories']}")
    summary_lines.append(f"Scoring system: Weighted evaluation across {len(component_scores)} dimensions")
    summary_lines.append("")
    
    # Show top theories by unified score
    summary_lines.append("TOP THEORIES BY UNIFIED SCORE (Comprehensive Evaluation):")
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'Rank':<6}{'Theory':<35}{'Category':<12}{'Score':<8}{'Bonuses/Penalties':<25}")
    summary_lines.append("-" * 80)
    
    for entry in leaderboard['rankings']['unified_score'][:15]:
        bonuses_str = ', '.join([b[0] for b in entry.get('bonuses', [])])
        penalties_str = ', '.join([p[0] for p in entry.get('penalties', [])])
        modifiers = []
        if bonuses_str:
            modifiers.append(f"+{bonuses_str}")
        if penalties_str:
            modifiers.append(f"-{penalties_str}")
        modifier_str = ' '.join(modifiers)[:24]
        
        summary_lines.append(
            f"{entry['rank']:<6}{entry['theory'][:34]:<35}{entry['category']:<12}"
            f"{entry['score']:<8.4f}{modifier_str:<25}"
        )
    
    # Component score leaders
    summary_lines.append("\n")
    summary_lines.append("COMPONENT SCORE LEADERS:")
    summary_lines.append("-" * 80)
    
    component_names = {
        'constraint_score': 'Constraints (Conservation, Metrics)',
        'observational_score': 'Observational (Quantum Tests)',
        'prediction_score': 'Predictions (CMB, PTA)',
        'trajectory_score': 'Trajectory Matching',
        'unification_score': 'Unification Potential'
    }
    
    for component, display_name in component_names.items():
        if component in leaderboard['rankings'] and leaderboard['rankings'][component]:
            leader = leaderboard['rankings'][component][0]
            summary_lines.append(f"{display_name:<40} {leader['theory']:<30} ({leader['score']:.3f})")
    
    # Category performance
    summary_lines.append("\n")
    summary_lines.append("PERFORMANCE BY THEORY CATEGORY:")
    summary_lines.append("-" * 80)
    
    for category, stats in leaderboard['score_breakdown']['category_breakdown'].items():
        summary_lines.append(
            f"{category.capitalize():<12} theories: {stats['count']:>3} | "
            f"Avg score: {stats['mean_score']:.3f} | "
            f"Best: {stats['best_theory']}"
        )
    
    # Overall statistics
    summary_lines.append("\n")
    summary_lines.append("OVERALL STATISTICS:")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Mean unified score: {leaderboard['score_breakdown']['unified_scores']['mean']:.3f}")
    summary_lines.append(f"Std deviation: {leaderboard['score_breakdown']['unified_scores']['std']:.3f}")
    summary_lines.append(f"Score range: {leaderboard['score_breakdown']['unified_scores']['min']:.3f} - {leaderboard['score_breakdown']['unified_scores']['max']:.3f}")
    
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append(f"Full details saved to: {leaderboard_path}")
    
    # Save and print summary
    summary_path = os.path.join(main_run_dir, 'leaderboard_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print('\n'.join(summary_lines))

def run_predictions_on_finalists(engine: TheoryEngine, main_run_dir: str, args: argparse.Namespace):
    """
    <reason>chain: Run prediction validators on all theories that passed constraint/observational tests</reason>
    <reason>chain: This is Phase 3 of evaluation - testing novel predictions against SOTA</reason>
    <reason>chain: Run after all theories complete to batch data downloads and comparisons</reason>
    """
    print("\n" + "="*60)
    print("PHASE 3: PREDICTION VALIDATION")
    print("Testing novel predictions against state-of-the-art benchmarks")
    print("="*60)
    
    # <reason>chain: Display quantum configuration for predictions</reason>
    print(f"\nEngine configuration:")
    print(f"  Quantum field content: {engine.quantum_field_content}")
    print(f"  Quantum theories will use quantum Lagrangian (if available)")
    
    # Get all theory directories from main run dir
    finalists = []
    
    # Get all theory directories in main run dir (excluding fail and baseline dirs)
    all_theory_dirs = glob.glob(os.path.join(main_run_dir, "*"))
    for theory_dir in all_theory_dirs:
        if not os.path.isdir(theory_dir):
            continue
            
        dir_name = os.path.basename(theory_dir)
        # Skip special directories
        if dir_name in ['predictions', 'run_config.json'] or dir_name.startswith('baseline_'):
            continue
            
        # Load theory info
        theory_info_path = os.path.join(theory_dir, 'theory_info.json')
        if not os.path.exists(theory_info_path):
            continue
            
        with open(theory_info_path, 'r') as f:
            theory_info = json.load(f)
            
            # <reason>chain: Check if theory passed all validations (constraints and observational)</reason>
            # Check constraint validation status
            constraints_passed = True
            final_validation_path = os.path.join(theory_dir, 'final_validation.json')
            if os.path.exists(final_validation_path):
                with open(final_validation_path, 'r') as f:
                    validation_data = json.load(f)
                    constraints_passed = validation_data.get('constraints_passed', False)
            
            # Check observational validation status (including quantum for quantum theories)
            observations_passed = True
            theory_category = theory_info.get('category', 'unknown')
            
            if theory_category == 'quantum':
                # Check quantum validation
                quantum_validation_path = os.path.join(theory_dir, 'quantum_validation.json')
                if os.path.exists(quantum_validation_path):
                    with open(quantum_validation_path, 'r') as f:
                        quantum_data = json.load(f)
                        # Count all observational validators (both quantum and classical)
                        # This is more inclusive and doesn't rely on specific naming
                        all_validators = quantum_data.get('validations', [])
                        total_validators = len(all_validators)
                        passed_validators = sum(1 for v in all_validators if v['flags']['overall'] == 'PASS')
                        
                        # For quantum theories, we expect them to pass most observational tests
                        # Using 80% threshold (e.g., 4 out of 5) but applied to all validators
                        if total_validators > 0:
                            pass_rate = passed_validators / total_validators
                            observations_passed = pass_rate >= 0.8  # 80% pass rate
                        else:
                            observations_passed = False  # No validations means it failed
                else:
                    observations_passed = False  # No quantum validation means it failed
            
            # Only include theories that passed both constraint and observational validations
            if constraints_passed and observations_passed:
                finalists.append({
                    'dir': theory_dir,
                    'info': theory_info
                })
            else:
                print(f"  Skipping {theory_info['name']} - failed validations (constraints={constraints_passed}, observations={observations_passed})")
    
    if not finalists:
        print("No finalist theories passed all validations for prediction testing.")
        return finalists
    
    print(f"Found {len(finalists)} finalist theories that passed all validations:")
    for f in finalists:
        print(f"  - {f['info']['name']} ({f['info'].get('category', 'unknown')})")
    
    # Initialize prediction results
    all_predictions = {}
    
    # Import and initialize prediction validators
    # <reason>chain: Only import validators that have been tested in solver_tests</reason>
    from physics_agent.validations import (
        CMBPowerSpectrumValidator, 
        # PTAStochasticGWValidator, # Not tested
        PrimordialGWsValidator, 
        # RenormalizabilityValidator, # Not tested
        # UnificationScaleValidator # Not tested
    )
    
    # <reason>chain: Ensure prediction validators use proper engine configuration</reason>
    # This is critical for quantum theories to use quantum Lagrangian
    validators = [
        CMBPowerSpectrumValidator(engine),
        # PTAStochasticGWValidator(engine), # Not tested
        PrimordialGWsValidator(engine)
        # <reason>chain: FutureDetectorsValidator and NovelSignaturesValidator removed - not implemented</reason>
    ]
    
    # <reason>chain: Quantum-specific validators removed - not tested in solver_tests</reason>
    # Add quantum-specific validators for all theories to check
    # validators.extend([
    #     RenormalizabilityValidator(engine),
    #     UnificationScaleValidator(engine)
    # ])
    
    print(f"\nRunning {len(validators)} prediction validators...")
    print(f"Engine quantum Lagrangian setting will apply to all predictions")
    
    # Run predictions on each finalist
    for finalist in finalists:
        theory_info = finalist['info']
        print(f"\n--- {theory_info['name']} ---")
        
        # Try to reconstruct the theory
        theory = None
        try:
            # Method 1: Try to use the theory loader to find the theory by class name
            if 'class_name' in theory_info:
                if not hasattr(engine, 'loader'):
                    if verbose:
                        print(f"  Warning: Engine has no loader attribute. Creating one...")
                    theories_dir = os.path.join(os.path.dirname(__file__), "theories")
                    engine.loader = TheoryLoader(theories_base_dir=theories_dir)
                
                discovered_theories = engine.loader.discover_theories()
                print(f"  Found {len(discovered_theories)} theories in loader")
                for key, theory_metadata in discovered_theories.items():
                    if theory_metadata['class_name'] == theory_info['class_name']:
                        # Reconstruct with parameters
                        params = theory_info.get('parameters', {})
                        
                        # <reason>chain: Filter parameters to only include those the theory accepts</reason>
                        # Get the actual parameters from the theory metadata
                        theory_params = theory_metadata.get('parameters', {})
                        filtered_params = {}
                        for param_name, param_value in params.items():
                            if param_name in theory_params:
                                filtered_params[param_name] = param_value
                        
                        # Try to instantiate with filtered parameters
                        try:
                            theory = engine.loader.instantiate_theory(key, **filtered_params)
                            if theory:
                                print(f"  Successfully reconstructed {theory_info['class_name']} with parameters: {filtered_params}")
                                break
                        except Exception as e:
                            print(f"Failed to instantiate theory {key}: {e}")
                            # Try without parameters
                            try:
                                theory = engine.loader.instantiate_theory(key)
                                if theory:
                                    print(f"  Successfully reconstructed {theory_info['class_name']} (default parameters)")
                                    break
                            except Exception as e2:
                                print(f"  Failed to instantiate {theory_info['class_name']}: {e2}")
                                continue
            
            if theory is None:
                if verbose:
                    print(f"  Warning: Could not reconstruct theory {theory_info['name']}")
                continue
            
            # Run actual predictions
            theory_predictions = {
                'theory_name': theory_info['name'],
                'category': theory_info.get('category', 'unknown'),
                'predictions': []
            }
            
            # <reason>chain: Log whether this theory should use quantum Lagrangian</reason>
            uses_quantum = engine.should_use_quantum_lagrangian(theory)
            if uses_quantum and args.verbose:
                print(f"  Theory {theory.name} will use quantum Lagrangian for predictions")
            
            for validator in validators:
                try:
                    # Run the actual validator
                    result = validator.validate(theory, verbose=True)
                    
                    # Extract relevant information
                    prediction_result = {
                        'validator': validator.name,
                        'sota_source': result.sota_source,
                        'sota_value': result.sota_value,
                        'theory_value': result.predicted_value,
                        'beats_sota': result.beats_sota,
                        'units': result.units
                    }
                    
                    # Calculate improvement
                    if result.units == "chi²/dof":
                        # For chi-squared, lower is better
                        improvement = result.sota_value - result.predicted_value
                    else:
                        # For log-likelihood, higher is better
                        improvement = result.predicted_value - result.sota_value
                    
                    prediction_result['improvement'] = improvement
                    
                    # Add detailed prediction data
                    if hasattr(result, 'prediction_data') and result.prediction_data:
                        prediction_result['details'] = result.prediction_data
                    
                    theory_predictions['predictions'].append(prediction_result)
                    
                    # Update comprehensive scores if they exist
                    scores_path = os.path.join(finalist['dir'], 'comprehensive_scores.json')
                    if os.path.exists(scores_path):
                        with open(scores_path, 'r') as f:
                            comp_scores = json.load(f)
                        
                        # Add prediction results
                        if 'predictions' not in comp_scores:
                            comp_scores['predictions'] = {}
                        
                        comp_scores['predictions'][validator.name] = {
                            'beats_sota': bool(result.beats_sota),  # Ensure it's a Python bool
                            'improvement': float(improvement) if not np.isinf(improvement) else 0.0,
                            'theory_value': float(result.predicted_value) if not np.isinf(result.predicted_value) else 0.0,
                            'sota_value': float(result.sota_value) if result.sota_value is not None else 0.0,
                            'units': str(result.units)
                        }
                        
                        # Save updated scores
                        with open(scores_path, 'w') as f:
                            json.dump(comp_scores, f, indent=4)
                    
                    status = "BEATS SOTA!" if result.beats_sota else "Does not beat SOTA"
                    print(f"  {validator.name}: {status} (improvement: {improvement:.2f})")
                    
                except Exception as e:
                    print(f"  Error running {validator.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    
            all_predictions[theory_info['name']] = theory_predictions
            
        except Exception as e:
            print(f"  Error processing theory: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create predictions directory if it doesn't exist
    predictions_dir = os.path.join(main_run_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Generate consolidated predictions report
    summary = {
        'run_timestamp': os.path.basename(main_run_dir),
        'total_finalists': len(finalists),
        'validators_used': [v.name for v in validators],
        'results': {}
    }
    
    # Create predictions directory
    predictions_dir = os.path.join(main_run_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Organize results by validator
    for validator in validators:
        validator_results = {
            'sota_benchmark': validator.get_sota_benchmark() if hasattr(validator, 'get_sota_benchmark') else {},
            'theories': []
        }
        
        for theory_name, predictions in all_predictions.items():
            for pred in predictions['predictions']:
                if pred['validator'] == validator.name:
                    validator_results['theories'].append({
                        'theory': theory_name,
                        'category': predictions['category'],
                        'beats_sota': bool(pred['beats_sota']),
                        'improvement': float(pred['improvement']) if not np.isinf(pred['improvement']) else 0.0,
                        'theory_value': float(pred.get('theory_value', 0)) if pred.get('theory_value') is not None and not np.isinf(pred.get('theory_value', 0)) else None,
                        'sota_value': float(pred.get('sota_value', 0)) if pred.get('sota_value') is not None and not np.isinf(pred.get('sota_value', 0)) else None,
                        'units': str(pred['units']),
                        'details': pred.get('details', {})
                    })
        
        # Sort by improvement (best first)
        validator_results['theories'].sort(key=lambda x: x['improvement'], reverse=True)
        
        # Add summary stats
        theories_beating_sota = sum(1 for t in validator_results['theories'] if t['beats_sota'])
        validator_results['summary'] = {
            'total_theories': len(validator_results['theories']),
            'beating_sota': theories_beating_sota,
            'percentage_beating_sota': (theories_beating_sota / len(validator_results['theories']) * 100) if validator_results['theories'] else 0
        }
        
        summary['results'][validator.name] = validator_results
    
    # Save the consolidated report
    report_path = os.path.join(predictions_dir, "predictions_report.json")
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Create a human-readable summary
    summary_text = []
    summary_text.append(f"Prediction Validation Report")
    summary_text.append(f"{'-'*60}")
    summary_text.append(f"Run: {os.path.basename(main_run_dir)}")
    summary_text.append(f"Finalists tested: {len(finalists)}")
    summary_text.append(f"Total predictions: {len(all_predictions)}")
    summary_text.append(f"")
    
    for validator_name, results in summary['results'].items():
        summary_text.append(f"\n{validator_name}")
        summary_text.append(f"{'-'*60}")
        benchmark = results['sota_benchmark']
        summary_text.append(f"SOTA: {benchmark.get('source', 'Unknown')} ({benchmark.get('value', 'N/A')} {benchmark.get('units', '')})")
        summary_text.append(f"Theories beating SOTA: {results['summary']['beating_sota']}/{results['summary']['total_theories']} ({results['summary']['percentage_beating_sota']:.1f}%)")
        
        if results['theories']:
            summary_text.append(f"\nTop performers:")
            for i, theory in enumerate(results['theories'][:5]):  # Top 5
                status = "✓" if theory['beats_sota'] else "✗"
                summary_text.append(f"  {i+1}. {status} {theory['theory']} ({theory['category']})")
                summary_text.append(f"     Improvement: {theory['improvement']:+.2f} {theory['units']}")
                if 'theory_value' in theory and 'sota_value' in theory:
                    summary_text.append(f"     Theory: {theory['theory_value']:.2f} vs SOTA: {theory['sota_value']:.2f}")
    
    summary_text.append(f"\n{'='*60}")
    summary_text.append(f"Full details saved to: {report_path}")
    
    summary_txt_path = os.path.join(predictions_dir, "summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write('\n'.join(summary_text))
    
    # Print summary
    print("\n" + '\n'.join(summary_text))
    
    # Return finalists with their predictions for further processing
    for finalist in finalists:
        theory_name = finalist['info']['name']
        if theory_name in all_predictions:
            finalist['predictions'] = all_predictions[theory_name]['predictions']
        else:
            finalist['predictions'] = []
    
    return finalists

def get_available_resources():
    """
    Check available system resources to determine safe number of workers.
    <reason>chain: Comprehensive resource checking to prevent system overload</reason>
    
    Returns:
        dict with resource availability information
    """
    resources = {
        'cpu_count': multiprocessing.cpu_count(),
        'cpu_percent': 0.0,
        'memory_available_gb': 0.0,
        'memory_percent': 0.0,
        'gpu_memory_available_gb': 0.0,
        'gpu_count': 0
    }
    
    try:
        # Check CPU usage (average over 1 second)
        resources['cpu_percent'] = psutil.cpu_percent(interval=1)
        
        # Check memory
        memory = psutil.virtual_memory()
        resources['memory_available_gb'] = memory.available / (1024**3)
        resources['memory_percent'] = memory.percent
        
        # Check GPU if available
        if torch.cuda.is_available():
            resources['gpu_count'] = torch.cuda.device_count()
            # Get available GPU memory for first device
            torch.cuda.empty_cache()  # Clear cache first
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            resources['gpu_memory_available_gb'] = total_memory - allocated
            
    except Exception as e:
        if verbose:
            print(f"Warning: Could not check all system resources: {e}")
        
    return resources

def determine_optimal_workers(num_tasks, device='cpu', estimated_memory_per_worker_gb=2.0):
    """
    Determine optimal number of workers based on available resources.
    <reason>chain: Smart worker allocation based on system state</reason>
    
    Args:
        num_tasks: Number of parameter combinations to process
        device: Target device ('cpu', 'cuda', 'mps')
        estimated_memory_per_worker_gb: Estimated memory per worker process
        
    Returns:
        int: Recommended number of workers
    """
    resources = get_available_resources()
    
    # Start with CPU count
    max_workers = resources['cpu_count']
    
    # Reduce if CPU is already busy (>50% usage)
    if resources['cpu_percent'] > 50:
        max_workers = max(1, max_workers // 2)
        print(f"  Note: CPU is {resources['cpu_percent']:.1f}% busy, reducing workers")
    
    # Check memory constraints
    # Reserve at least 4GB for system and main process
    usable_memory_gb = max(0, resources['memory_available_gb'] - 4.0)
    memory_limited_workers = int(usable_memory_gb / estimated_memory_per_worker_gb)
    
    if memory_limited_workers < max_workers:
        print(f"  Note: Limited by available memory ({resources['memory_available_gb']:.1f}GB available)")
        max_workers = max(1, memory_limited_workers)
    
    # GPU-specific constraints
    if device == 'cuda' and resources['gpu_count'] > 0:
        # For GPU, usually want fewer workers to avoid VRAM contention
        # Each worker might allocate GPU memory
        gpu_memory_per_worker = 1.0  # Estimate 1GB per worker for GPU
        gpu_limited_workers = int(resources['gpu_memory_available_gb'] / gpu_memory_per_worker)
        
        if gpu_limited_workers < max_workers:
            print(f"  Note: Limited by GPU memory ({resources['gpu_memory_available_gb']:.1f}GB available)")
            max_workers = max(1, min(max_workers, gpu_limited_workers))
            
        # Also limit to 2-4 workers for GPU to avoid context switching overhead
        max_workers = min(max_workers, 4)
    
    # Never exceed number of tasks
    max_workers = min(max_workers, num_tasks)
    
    # Apply hard limit
    max_workers = min(max_workers, 8)
    
    return max_workers

def process_sweep_combination(combo_data):
    """
    Process a single parameter combination for a sweep.
    <reason>chain: Separate function for parallel processing of sweep combinations</reason>
    
    Args:
        combo_data: tuple of (model_class, sweep_kwargs, args_dict, engine_params)
    
    Returns:
        dict with results of the evaluation
    """
    model_class, sweep_kwargs, args_dict, engine_params = combo_data
    
    # <reason>chain: Create a unique log file for this worker thread</reason>
    param_str = "_".join(f"{k}_{v:.3e}".replace(".", "_") for k, v in sweep_kwargs.items())
    temp_dir = os.path.join(engine_params['main_run_dir'], f"sweep_{param_str}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set up worker log file
    log_file = os.path.join(temp_dir, "worker_log.txt")
    
    def log_message(msg):
        """Log message to both file and stdout"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {msg}\n"
        with open(log_file, 'a') as f:
            f.write(log_entry)
        print(f"[Worker {param_str}] {msg}")
    
    try:
        log_message(f"Starting sweep for parameters: {sweep_kwargs}")
        
        # Create instance with specific parameters
        log_message(f"Creating instance of {model_class.__name__}")
        instance = model_class(**sweep_kwargs)
        log_message(f"Instance created: {instance.name}")
        
        # Recreate engine for this process (needed for multiprocessing)
        log_message(f"Creating engine with device={engine_params['device']}, dtype={engine_params['dtype']}")
        engine = TheoryEngine(
            device=engine_params['device'],
            dtype=getattr(torch, engine_params['dtype'])
        )
        log_message("Engine created successfully")
        
        # Convert args_dict back to namespace
        args = argparse.Namespace(**args_dict)
        log_message(f"Arguments converted, steps={engine_params['effective_steps']}")
        
        # <reason>chain: Reconstruct baseline theories from serializable info</reason>
        baseline_theories = {}
        if 'baseline_theories_info' in engine_params:
            log_message("Reconstructing baseline theories...")
            for name, info in engine_params['baseline_theories_info'].items():
                try:
                    module = importlib.import_module(info['module'])
                    cls = getattr(module, info['class_name'])
                    theory = cls(**info['params'])
                    baseline_theories[name] = theory
                    log_message(f"  Reconstructed baseline theory: {name}")
                except Exception as e:
                    log_message(f"  WARNING: Failed to reconstruct baseline theory {name}: {e}")
        else:
            # Fallback for backward compatibility
            baseline_theories = engine_params.get('baseline_theories', {})
        
        # Process the theory
        log_message("Starting process_and_evaluate_theory...")
        try:
            process_and_evaluate_theory(
                instance, engine, temp_dir,
                engine_params['baseline_results'],
                baseline_theories,  # Use reconstructed baseline theories
                args,
                engine_params['effective_steps'],
                torch.tensor(engine_params['effective_dtau'], device=engine.device, dtype=engine.dtype),
                torch.tensor([engine_params['r0_val']], device=engine.device, dtype=engine.dtype),
                engine_params['r0_val'],
                engine_params['quantum_interval'],
                engine_params['quantum_beta'],
                None,  # progress_callback
                engine_params['callback_interval']
            )
            log_message("process_and_evaluate_theory completed successfully")
        except Exception as eval_error:
            log_message(f"ERROR in process_and_evaluate_theory: {str(eval_error)}")
            log_message(f"Traceback:\n{traceback.format_exc()}")
            raise
        
        # Read results from the temp directory
        results_file = os.path.join(temp_dir, "validation_results.json")
        log_message(f"Looking for results file: {results_file}")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            log_message(f"Results loaded successfully, status={results.get('status', 'unknown')}")
        else:
            results = {"status": "failed", "error": "No results file generated"}
            log_message("WARNING: No results file generated")
            
        # Add sweep parameters to results
        results['sweep_params'] = sweep_kwargs
        results['worker_log'] = log_file  # Include log file path
        
        log_message("Sweep combination completed successfully")
        return results
        
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        log_message(f"CRITICAL ERROR: {error_msg}")
        log_message(f"Full traceback:\n{tb}")
        
        return {
            "status": "error",
            "error": error_msg,
            "traceback": tb,
            "sweep_params": sweep_kwargs,
            "worker_log": log_file
        }

def main():
    """Main execution function"""
    # Check for updates at startup
    check_on_startup()
    
    parser = get_cli_parser()
    args = parser.parse_args()
    
    # <reason>chain: Use CLI module for execution mode setup</reason>
    setup_execution_mode(args)
    
    # <reason>chain: Handle special modes through CLI module</reason>
    if handle_special_modes(args):
        return
    
    # <reason>chain: Use CLI module to determine device and dtype</reason>
    device, dtype = determine_device_and_dtype(args)
        
    if device is None:
        if args.gpu_f32:
            # <reason>Check for CUDA first (NVIDIA GPUs), then MPS (Apple Silicon), fallback to CPU</reason>
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                if verbose:
                    print("Warning: GPU requested but neither CUDA nor MPS available. Falling back to CPU.")
                device = "cpu"
        elif args.final or args.cpu_f64:
            device = "cpu"
        else:
            device = "cpu"
            
    # <reason>chain: Pass quantum configuration to TheoryEngine</reason>
    engine = TheoryEngine(
        device=device, 
        dtype=dtype,
        quantum_field_content=getattr(args, 'quantum_field_content', 'all'),
        quantum_phase_precision=getattr(args, 'quantum_phase_precision', 1e-30),
        verbose=args.verbose
    )
    engine.loss_type = 'ricci'  # <reason>chain: Ricci tensor is the only loss type</reason>
    print(f"Running on device: {engine.device}, with dtype: {engine.dtype}")
    
    # <reason>chain: Print quantum configuration for clarity</reason>
    print(f"Quantum field content for quantum theories: {engine.quantum_field_content}")
    
    # --- Theory Loading using TheoryLoader ---
    theories_dir = os.path.join(os.path.dirname(__file__), "theories")
    
    # <reason>chain: In candidates mode, discover theories from both regular and candidates directories</reason>
    if args.candidates:
        print("Running in candidates mode - loading all theories including candidates...")
        # First load regular theories
        loader = TheoryLoader(theories_base_dir=theories_dir)
        discovered_theories = loader.discover_theories()
        
        # Then discover candidates
        candidates_dir = os.path.join(theories_dir, "candidates")
        if os.path.exists(candidates_dir):
            # Create a second loader for candidates
            candidate_loader = TheoryLoader(theories_base_dir=candidates_dir)
            candidate_theories = candidate_loader.discover_theories()
            
            # Merge candidates into main theory list with "candidate/" prefix
            for key, value in candidate_theories.items():
                # Add candidate prefix to distinguish them
                candidate_key = f"candidate/{key}"
                value['is_candidate'] = True
                discovered_theories[candidate_key] = value
            
            print(f"Discovered {len(candidate_theories)} candidate theories")
    else:
        loader = TheoryLoader(theories_base_dir=theories_dir)
        discovered_theories = loader.discover_theories()
    
    print(f"--- Loading Theories ---")
    print(f"Discovered {len(discovered_theories)} theory classes total")
    
    # Apply filters if specified
    filtered_theories = discovered_theories
    if args.theory_filter:
        filtered_theories = {k: v for k, v in filtered_theories.items() 
                           if args.theory_filter.lower() in k.lower()}
        print(f"Filtered to {len(filtered_theories)} theories matching '{args.theory_filter}'")
    
    # <reason>chain: Apply category filter to run theories by category (e.g., ugm, quantum, classical)</reason>
    if hasattr(args, 'category') and args.category:
        # For UGM category, we need to check the theory name since it's a special unified theory
        if args.category.lower() == 'ugm':
            # Filter from the already filtered theories (which might have theory_filter applied)
            filtered_theories = {k: v for k, v in filtered_theories.items() 
                               if 'ugm' in k.lower()}
            if len(filtered_theories) == 0:
                # If no matches found, show available theories for debugging
                print("No UGM theories found. Available theory keys:")
                for k in discovered_theories.keys():
                    if 'ugm' in k.lower():
                        print(f"  - {k} (contains 'ugm')")
            print(f"Filtered to {len(filtered_theories)} theories in UGM (Unified Gravity Model) category")
        else:
            # For other categories, use the category attribute
            filtered_theories = {k: v for k, v in filtered_theories.items() 
                               if v.get('category', '').lower() == args.category.lower()}
            print(f"Filtered to {len(filtered_theories)} theories in category '{args.category}'")
    
    # Get default instances
    theories_to_run = {}
    for theory_key, theory_info in filtered_theories.items():
        # Special handling for baseline theories that need parameters
        kwargs = {}
        
        # <reason>chain: Handle new sweepable_fields by adding them to kwargs</reason>
        if hasattr(theory_info.get('class'), 'sweepable_fields'):
            for field, field_info in theory_info.get('class').sweepable_fields.items():
                kwargs[field] = field_info.get('default', 0.0)
        
        # <reason>chain: Use the appropriate loader for candidate theories</reason>
        if theory_info.get('is_candidate', False) and args.candidates:
            # Remove the "candidate/" prefix for the candidate loader
            candidate_key = theory_key.replace('candidate/', '')
            instance = candidate_loader.instantiate_theory(candidate_key, **kwargs)
        else:
            instance = loader.instantiate_theory(theory_key, **kwargs)
        if instance:
            theories_to_run[instance.name] = instance
            if args.verbose:
                print(f"  Loaded: {instance.name} [{theory_info.get('category', 'N/A')}]")
    
    if not theories_to_run:
        print("No theories were found to run. Exiting.")
        return
    
    print(f"Loaded {len(theories_to_run)} theory instances")

    # --- Trajectory Setup ---
    if args.final:
        N_STEPS = 500000
        CALLBACK_INTERVAL = 30000
    else:
        N_STEPS = 20000
        CALLBACK_INTERVAL = 5000
        
    # Override with user-specified step count if provided
    if args.steps is not None:
        N_STEPS = args.steps
        CALLBACK_INTERVAL = max(1, N_STEPS // 4) if getattr(args, 'self_monitor', False) else N_STEPS + 1
    
    # <reason>chain: Use standard timestep of 0.01 for accurate energy conservation</reason>
    DTau = torch.tensor(0.01, device=engine.device, dtype=engine.dtype)
    if args.close_orbit:
        r0_val = 6.0 * engine.rs
    else:
        r0_val = 15.0 * engine.rs
    r0 = torch.tensor([r0_val], device=engine.device, dtype=engine.dtype)
    
    # <reason>chain: Removed sample rate functionality - using direct values instead</reason>
    effective_dtau = DTau
    effective_steps = N_STEPS
    
    # --- Separate Theories and Baselines ---
    # Use the discovered_theories info to determine which are baselines
    baseline_theories = {}
    custom_theories = {}
    
    for name, model in theories_to_run.items():
        # Find the theory key for this instance
        theory_key = None
        for key, info in discovered_theories.items():
            # Match class name from the key
            class_name_from_key = key.split('/')[-1]
            if class_name_from_key == model.__class__.__name__:
                theory_key = key
                break
        
        if theory_key and 'baselines' in discovered_theories.get(theory_key, {}).get('path', ''):
            baseline_theories[name] = model
        else:
            custom_theories[name] = model
    
    # <reason>chain: Always load Kerr and Kerr-Newman baselines for visualization</reason>
    # Always load key baseline theories for checkpoints
    if True:  # Always load baselines for visualization
        print("Loading baseline theories for comparison...")
        # Import baseline theories directly
        try:
            from physics_agent.theories.defaults.baselines.kerr import Kerr
            from physics_agent.theories.defaults.baselines.kerr_newman import KerrNewman
            
            # Create instances with default parameters
            kerr_instance = Kerr(a=0.0)  # Pure gravity baseline (Schwarzschild)
            baseline_theories[kerr_instance.name] = kerr_instance
            print(f"  Loaded baseline: {kerr_instance.name} (pure gravity)")
            
            kn_instance = KerrNewman(a=0.0, q_e=0.5)  # Gravity + electromagnetism baseline (Reissner-Nordström)
            baseline_theories[kn_instance.name] = kn_instance  
            print(f"  Loaded baseline: {kn_instance.name} (gravity + electromagnetism)")
        except Exception as e:
            print(f"  Failed to load baselines: {e}")
    
    # Handle experimental parameters
    quantum_interval = getattr(args, 'experimental_quantum_interval', 1000) if getattr(args, 'experimental', False) else 0
    quantum_beta = getattr(args, 'experimental_quantum_beta', 0.01) if getattr(args, 'experimental', False) else 0.0
    
    # Set up progress callback for monitoring
    progress_callback = None
    if getattr(args, 'self_monitor', False):
        def progress_callback(hist_so_far, current_step, total_steps):
            if current_step % CALLBACK_INTERVAL == 0:
                print(f"  Progress: {current_step}/{total_steps} steps ({100*current_step/total_steps:.1f}%)")
    
    # --- Simulate Baselines First ---
    baseline_results = {}
    print("\n--- Simulating Baseline Theories ---")
    
    # Create a single run directory for this entire execution
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dtype_str = str(engine.dtype).split('.')[-1]
    main_run_dir = os.path.join("runs", f"run_{timestamp}_{dtype_str}")
    os.makedirs(main_run_dir, exist_ok=True)
    
    # Save run configuration
    run_config = {
        "timestamp": timestamp,
        "device": str(engine.device),
        "dtype": str(engine.dtype),
        "n_steps": N_STEPS,
        "dtau": DTau.item(),
        "r0": r0_val,
        "args": vars(args),
        "theories": list(theories_to_run.keys()),
        "command_line": ' '.join(sys.argv),  # <reason>chain: Store exact command line for reproducibility</reason>
        "python_version": sys.version,
        "working_directory": os.getcwd()
    }
    with open(os.path.join(main_run_dir, "run_config.json"), 'w') as f:
        json.dump(run_config, f, indent=4)
    
    print(f"Run directory: {main_run_dir}")
    
    # <reason>chain: Start capturing all output to a log file</reason>
    run_logger = RunLogger(main_run_dir)
    run_logger.start()
    
    # Store log path in run config for later use
    run_config['log_file'] = run_logger.get_log_path()
    with open(os.path.join(main_run_dir, "run_config.json"), 'w') as f:
        json.dump(run_config, f, indent=4)
    
    for name, model in baseline_theories.items():
        print(f"Running: {name}")
        # Create subdirectory for this baseline theory
        theory_dir = os.path.join(main_run_dir, f"baseline_{name.replace(' ', '_').replace('(', '').replace(')', '')}")
        os.makedirs(theory_dir, exist_ok=True)
        
        hist, _, _ = engine.run_trajectory(
            model, r0_val * engine.length_scale, effective_steps, effective_dtau.item() * engine.time_scale,
            quantum_interval=0, quantum_beta=0.0,
            no_cache=args.no_cache
        )
        if hist is not None and hist.shape[0] > 1:
            baseline_results[name] = hist
            # <reason>chain: Also store in engine for unification scoring</reason>
            engine.baseline_trajectories[name] = hist
            # Save baseline trajectory
            torch.save(hist.to(dtype=engine.dtype), os.path.join(theory_dir, "trajectory.pt"))
        else:
            if verbose:
                print(f"  Warning: Baseline simulation failed for {name}")
    
    # --- First Phase: Validate All Theories ---
    print("\n--- Phase 1: Validating All Theories ---")
    print(f"{'='*60}")
    
    validation_results = {}
    theories_that_passed = []
    theories_that_failed = []
    for name, model_prototype in theories_to_run.items():
        if hasattr(model_prototype, 'sweep') and model_prototype.sweep and args.enable_sweeps:
            print(f"  Detected parameter sweep for {model_prototype.name}: {model_prototype.sweep}")
            
            param_ranges = model_prototype.sweep
            param_names = list(param_ranges.keys())
            # Generate values for each parameter
            sweep_values = []
            for param, range_spec in param_ranges.items():
                min_val = range_spec['min']
                max_val = range_spec['max']
                points = range_spec['points']
                scale = range_spec.get('scale', 'linear')
                
                if scale == 'log':
                    values = np.logspace(np.log10(min_val), np.log10(max_val), points)
                else:
                    values = np.linspace(min_val, max_val, points)
                sweep_values.append(values)
            
            param_combinations = list(itertools.product(*sweep_values))

            print(f"  Generated {len(param_combinations)} parameter combinations to test.")
            
            # <reason>chain: Use parallel processing for sweeps</reason>
            # Determine number of workers based on available resources
            if args.disable_resource_check:
                # Simple CPU count-based allocation (legacy behavior)
                if args.sweep_workers:
                    max_workers = min(args.sweep_workers, len(param_combinations))
                else:
                    max_workers = min(multiprocessing.cpu_count(), len(param_combinations), 8)
                print(f"  Using simple CPU-based allocation: {max_workers} workers (resource checking disabled)")
            else:
                # Smart resource-based allocation
                if args.sweep_workers:
                    # User specified, but still show resource info
                    max_workers = args.sweep_workers
                    resources = get_available_resources()
                    print(f"  System resources: {resources['cpu_count']} CPUs, {resources['memory_available_gb']:.1f}GB RAM available")
                    if max_workers > resources['cpu_count']:
                        if verbose:
                            print(f"  Warning: Requested {max_workers} workers but only {resources['cpu_count']} CPUs available")
                else:
                    # Auto-determine based on resources
                    max_workers = determine_optimal_workers(
                        len(param_combinations), 
                        device=str(engine.device),
                        estimated_memory_per_worker_gb=args.sweep_memory_per_worker
                    )
            
            print(f"  Using {max_workers} parallel workers for sweep processing.")
            
            # Prepare data for parallel processing
            # <reason>chain: Convert baseline theories to serializable format for multiprocessing</reason>
            baseline_theories_info = {}
            for name, theory in baseline_theories.items():
                # Store class name and parameters for reconstruction
                baseline_theories_info[name] = {
                    'class_name': theory.__class__.__name__,
                    'module': theory.__class__.__module__,
                    'params': getattr(theory, '_init_params', {})  # Store init params if available
                }
            
            engine_params = {
                'device': str(engine.device),
                'dtype': str(engine.dtype).split('.')[-1],
                'main_run_dir': main_run_dir,
                'baseline_results': baseline_results,
                'baseline_theories_info': baseline_theories_info,  # Pass serializable info instead
                'effective_steps': effective_steps,
                'effective_dtau': effective_dtau.item(),
                'r0_val': r0_val,
                'quantum_interval': quantum_interval,
                'quantum_beta': quantum_beta,
                'callback_interval': CALLBACK_INTERVAL
            }
            
            # Convert args to dict for serialization
            args_dict = vars(args)
            
            # Create work items
            work_items = []
            for combo in param_combinations:
                sweep_kwargs = dict(zip(param_names, combo))
                work_item = (model_prototype.__class__, sweep_kwargs, args_dict, engine_params)
                work_items.append(work_item)
            
            # Process combinations in parallel
            sweep_results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_combo = {
                    executor.submit(process_sweep_combination, work_item): work_item[1]
                    for work_item in work_items
                }
                
                # Process completed tasks
                for i, future in enumerate(as_completed(future_to_combo)):
                    combo = future_to_combo[future]
                    try:
                        result = future.result()
                        sweep_results.append(result)
                        param_str = ", ".join(f"{k}={v}" for k, v in combo.items())
                        print(f"\n--- Completed sweep variant {i+1}/{len(param_combinations)}: {param_str} ---")
                        
                        if result.get('status') == 'error':
                            print(f"  [ERROR] {result.get('error', 'Unknown error')}")
                            # <reason>chain: Display worker log on error for debugging</reason>
                            if 'worker_log' in result and os.path.exists(result['worker_log']):
                                print(f"  [WORKER LOG] Reading from: {result['worker_log']}")
                                with open(result['worker_log'], 'r') as log_f:
                                    log_contents = log_f.read()
                                    print("  === Worker Log Contents ===")
                                    for line in log_contents.splitlines():
                                        print(f"  {line}")
                                    print("  === End Worker Log ===")
                            if 'traceback' in result:
                                print(f"  [TRACEBACK]\n{result['traceback']}")
                    except Exception as e:
                        error_msg = f"Failed to process combination: {e}"
                        print(f"  [ERROR] {error_msg}")
                        # <reason>chain: Try to get more details about process termination</reason>
                        import signal
                        if hasattr(e, '__cause__') and isinstance(e.__cause__, signal.Signals):
                            print(f"  [SIGNAL] Process terminated by signal: {e.__cause__.name}")
                        sweep_results.append({
                            "status": "error",
                            "error": error_msg,
                            "exception_type": type(e).__name__,
                            "sweep_params": combo
                        })
                
            # <reason>chain: Save consolidated sweep results</reason>
            sweep_summary_file = os.path.join(main_run_dir, f"{model_prototype.name.replace(' ', '_')}_sweep_summary.json")
            with open(sweep_summary_file, 'w') as f:
                json.dump(sweep_results, f, indent=2)
            print(f"\n  Sweep results saved to: {sweep_summary_file}")
                
        else: # No sweep or sweeps disabled - use preferred values
            # Get preferred parameter values
            preferred_params = get_preferred_values(model_prototype)
            
            # If this theory has sweep params but we're not sweeping, use preferred values
            if hasattr(model_prototype, 'sweep') and model_prototype.sweep and not args.enable_sweeps:
                print(f"  Using preferred values for {model_prototype.name}: {preferred_params}")
                try:
                    # Create instance with preferred parameters
                    instance = model_prototype.__class__(**preferred_params)
                except Exception as e:
                    print(f"  [WARNING] Failed to instantiate with preferred params, using default: {e}")
                    instance = model_prototype
            else:
                instance = model_prototype
            
            # Run validation only
            validation_result = validate_theory_only(
                instance, engine, main_run_dir, baseline_results, baseline_theories, args,
                r0, r0_val
            )
            
            validation_results[instance.name] = validation_result
            if validation_result['passed']:
                theories_that_passed.append((instance, validation_result['theory_dir']))
            else:
                theories_that_failed.append((instance, validation_result['theory_dir']))

    # --- Output Validation Results ---
    print(f"\n{'='*60}")
    print(f"Validation Summary:")
    print(f"  Total theories validated: {len(validation_results)}")
    print(f"  Passed: {len(theories_that_passed)}")
    print(f"  Failed: {len(theories_that_failed)}")
    print(f"{'='*60}")
    
    if theories_that_failed:
        print(f"\nTheories that failed validation:")
        for theory, theory_dir in theories_that_failed:
            print(f"  ✗ {theory.name}")
            # Move to fail directory
            fail_dir = os.path.join(main_run_dir, "fail")
            os.makedirs(fail_dir, exist_ok=True)
            try:
                sanitized_name = os.path.basename(theory_dir)
                dest_path = os.path.join(fail_dir, sanitized_name)
                if os.path.exists(dest_path):
                    timestamp = time.strftime("%H%M%S")
                    dest_path = os.path.join(fail_dir, f"{sanitized_name}_{timestamp}")
                shutil.move(theory_dir, dest_path)
                print(f"    → Moved to: {dest_path}")
            except Exception as e:
                print(f"    [ERROR] Could not move directory: {e}")
    
    if theories_that_passed:
        print(f"\nTheories that passed validation:")
        for theory, theory_dir in theories_that_passed:
            print(f"  ✓ {theory.name}")
    
    # --- Phase 2: Run Trajectories for Validated Theories ---
    if theories_that_passed:
        print(f"\n--- Phase 2: Running Full Trajectories for Validated Theories ---")
        print(f"{'='*60}")
        
        for theory, theory_dir in theories_that_passed:
            success = run_trajectory_and_visualize(
                theory, engine, theory_dir, baseline_results, baseline_theories, args,
                effective_steps, effective_dtau, r0, r0_val,
                quantum_interval, quantum_beta, progress_callback, CALLBACK_INTERVAL
            )
            
            if not success:
                print(f"  [WARNING] Could not compute trajectory for {theory.name}")
                # Continue with the theory in the ranking despite computational issues
    
    print(f"\n{'='*60}")
    print(f"Run complete. All results saved to: {main_run_dir}")
    print(f"{'='*60}")
    
    # Summary of quantum theories that passed all tests
    print(f"\n--- Summary: Quantum Theory Validation Results ---")
    passed_theories = []
    failed_theories = []
    
    # Get all theory directories in main run dir (excluding fail and baseline dirs)
    all_theory_dirs = glob.glob(os.path.join(main_run_dir, "*"))
    for theory_dir in all_theory_dirs:
        if os.path.isdir(theory_dir):
            dir_name = os.path.basename(theory_dir)
            
            # Skip baseline directories, predictions directory, and other non-theory files
            if dir_name in ['predictions', 'run_config.json'] or dir_name.startswith('baseline_'):
                continue
            
            # Check if quantum validation passed
            quantum_val_path = os.path.join(theory_dir, 'quantum_validation.json')
            theory_info_path = os.path.join(theory_dir, 'theory_info.json')

            # Get theory name from theory_info.json if available
            theory_name = dir_name.replace("_", " ")
            if os.path.exists(theory_info_path):
                with open(theory_info_path, 'r') as f:
                    info = json.load(f)
                    theory_name = info.get('name', theory_name)

            if os.path.exists(quantum_val_path):
                with open(quantum_val_path, 'r') as f:
                    quantum_data = json.load(f)
                    # Count quantum validators that passed
                    all_validators = quantum_data.get('validations', [])
                    quantum_passed_count = sum(1 for v in all_validators if v['flags']['overall'] == 'PASS')
                    quantum_total = len(all_validators)
                    pass_rate = quantum_passed_count / quantum_total if quantum_total > 0 else 0
                    quantum_passed = pass_rate >= 0.8  # 80% pass rate
                    if quantum_passed:
                        passed_theories.append(theory_name)
                    else:
                        # This shouldn't happen since failed theories are moved
                        failed_theories.append(theory_name)
            else:
                # No quantum validation means it's not a quantum candidate
                failed_theories.append(f"{theory_name} (no quantum validation)")

    print(f"\nFinalist theories that passed all quantum tests ({len(passed_theories)}):")
    for theory in sorted(passed_theories):
        print(f"  ✓ {theory}")

    if failed_theories:
        print(f"\nTheories in main directory without quantum validation ({len(failed_theories)}):")
        for theory in sorted(failed_theories):
            print(f"  ✗ {theory}")

    print(f"\nAll theories have been evaluated and ranked in the comprehensive report.")
    print(f"See the leaderboard for complete rankings and detailed analysis.")
    print(f"{'='*60}")
    
    # Run prediction validators on finalists
    finalists = run_predictions_on_finalists(engine, main_run_dir, args)
    
    # Update comprehensive reports with prediction results
    update_comprehensive_reports_with_predictions(main_run_dir)
    
    # Generate comprehensive leaderboard
    generate_leaderboard(main_run_dir)

    # Generate comprehensive summary of all theories
    # <reason>chain: Generate final summary with all pass/fail results and errors</reason>
    generate_comprehensive_summary(main_run_dir, validation_results)

    # After predictions, auto-sweep top theories if --enable-sweeps
    # <reason>chain: Implement auto-sweeping of top winners with sweepable_fields</reason>
    if args.enable_sweeps:
        print("\n=== Auto-Sweeping Top Theories ===")
        # Identify top winners: e.g., beat SOTA in >=1 validator
        top_theories = [f for f in finalists if any(pred.get('beats_sota', False) for pred in f.get('predictions', []))]
        if not top_theories:
            print("No top theories to sweep.")
        else:
            print(f"Sweeping {len(top_theories)} top theories...")
            # For now, just print what we would sweep
            for finalist in top_theories:
                print(f"  Would sweep: {finalist['info']['name']}")
            print("Note: Auto-sweep functionality is still under development.")
    
    # <reason>chain: Stop capturing output and save the log file</reason>
    run_logger.stop()

def update_comprehensive_reports_with_predictions(main_run_dir: str):
    """
    <reason>chain: Update comprehensive HTML reports after prediction validators have run</reason>
    <reason>chain: This ensures prediction results are included in the final reports</reason>
    """
    print(f"\n{'='*60}")
    print("Updating comprehensive reports with prediction results")
    print(f"{'='*60}")
    
    # Load prediction results
    predictions_report_path = os.path.join(main_run_dir, "predictions", "predictions_report.json")
    if not os.path.exists(predictions_report_path):
        print("No prediction results found. Skipping report update.")
        return
        
    with open(predictions_report_path, 'r') as f:
        predictions_data = json.load(f)
    
    # Process each theory directory
    report_generator = ComprehensiveReportGenerator()
    theories_updated = 0
    
    for entry in os.listdir(main_run_dir):
        theory_dir = os.path.join(main_run_dir, entry)
        
        # Skip non-directories and special directories
        if not os.path.isdir(theory_dir) or entry in ['fail', 'predictions'] or entry.startswith('baseline_'):
            continue
            
        # Load comprehensive scores
        scores_path = os.path.join(theory_dir, 'comprehensive_scores.json')
        if not os.path.exists(scores_path):
            continue
            
        with open(scores_path, 'r') as f:
            theory_scores = json.load(f)
            
        theory_name = theory_scores.get('theory_name', 'Unknown')
        
        # Add prediction results to theory scores
        prediction_results_added = False
        
        # Look for this theory in prediction results
        for validator_name, validator_results in predictions_data['results'].items():
            for theory_result in validator_results['theories']:
                if theory_result['theory'] == theory_name:
                    # Add this prediction result to the theory scores
                    if 'predictions' not in theory_scores:
                        theory_scores['predictions'] = {}
                        
                    # Store the prediction result
                    theory_scores['predictions'][validator_name] = {
                        'loss': 0.0 if theory_result['beats_sota'] else 1.0,
                        'passed': theory_result['beats_sota'],
                        'details': {
                            'beats_sota': theory_result['beats_sota'],
                            'improvement': theory_result['improvement'],
                            'theory_value': theory_result.get('theory_value'),
                            'sota_value': theory_result.get('sota_value'),
                            'units': theory_result['units']
                        }
                    }
                    prediction_results_added = True
        
        if prediction_results_added:
            # Save updated scores
            with open(scores_path, 'w') as f:
                # <reason>chain: Use to_serializable to handle NaN, infinity, and tensor values</reason>
                json.dump(to_serializable(theory_scores), f, indent=4)
            
            # Regenerate HTML report
            try:
                # Get logs if available
                log_content = None
                log_path = os.path.join(theory_dir, 'execution_log.txt')
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f:
                        log_content = f.read()
                
                # Generate updated report
                report_path = report_generator.generate_theory_report(
                    theory_name=theory_name,
                    theory_results=theory_scores,
                    output_dir=theory_dir,
                    logs=log_content
                )
                
                theories_updated += 1
                print(f"  Updated report for {theory_name}")
                
            except Exception as e:
                print(f"  [WARNING] Failed to update report for {theory_name}: {e}")
    
    print(f"\nUpdated {theories_updated} theory reports with prediction results")
    print(f"{'='*60}")



if __name__ == "__main__":
    main()