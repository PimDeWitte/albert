#!/usr/bin/env python3
"""
Unified Trajectory Calculator for Gravitational Theories.

Combines classical geodesic calculations with quantum path integral trajectories
to enable comprehensive gravity unification studies.

<reason>chain: Main entry point for unified classical-quantum trajectory calculations</reason>
"""

import argparse
import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional
import json
import sys

from physics_agent.base_theory import GravitationalTheory
from physics_agent.cache import TrajectoryCache
from physics_agent.geodesic_integrator_stable import (
    GeodesicRK4Solver, 
    ChargedGeodesicRK4Solver, UGMGeodesicRK4Solver
)
# <reason>chain: Import true 6DOF solver from non-stable version for quantum theories</reason>
from physics_agent.geodesic_integrator_stable import GeneralGeodesicRK4Solver as GeneralGeodesicRK4Solver_4D
from physics_agent.geodesic_integrator import GeneralGeodesicRK4Solver as GeneralGeodesicRK4Solver_6D
from physics_agent.quantum_path_integrator import (
    QuantumLossCalculator, visualize_quantum_paths
)
# from physics_agent.theory_loader import TheoryLoader  # Commented out to avoid import issues


class UnifiedTrajectoryCalculator:
    """
    <reason>chain: Main calculator combining classical and quantum trajectory methods</reason>
    """
    
    def __init__(self, theory: GravitationalTheory, enable_quantum: bool = True,
                 enable_classical: bool = True, M: float = None, c: float = None, 
                 G: float = None, cache: 'TrajectoryCache' = None):
        """
        Initialize unified calculator.
        
        Args:
            theory: GravitationalTheory instance
            enable_quantum: Whether to compute quantum trajectories
            enable_classical: Whether to compute classical trajectories
            M: Central mass in kg (for unit conversion)
            c: Speed of light in m/s
            G: Gravitational constant in SI units
            cache: Optional TrajectoryCache instance for caching trajectories
        """
        self.theory = theory
        self.enable_quantum = enable_quantum and theory.enable_quantum
        self.enable_classical = enable_classical
        self._cache = cache  # <reason>chain: Store cache instance for trajectory caching</reason>
        
        # Store unit conversion parameters
        self.M = M if M is not None else 1.989e30  # Solar mass
        self.c = c if c is not None else 2.998e8   # Speed of light
        self.G = G if G is not None else 6.674e-11 # Gravitational constant
        
        # <reason>chain: Calculate conversion factors for geometric units</reason>
        self.length_scale = self.G * self.M / self.c**2  # GM/c^2
        self.time_scale = self.G * self.M / self.c**3   # GM/c^3
        
        # Initialize solvers
        self._init_classical_solver()
        self._init_quantum_solver()
        
    def _init_classical_solver(self):
        """<reason>chain: Initialize appropriate classical geodesic solver</reason>"""
        if not self.enable_classical:
            self.classical_solver = None
            return
            
        # Use the stored parameters
        M = torch.tensor(self.M, dtype=torch.float64)
        c = self.c
        G = self.G
        
        # <reason>chain: Check if this is a quantum theory that needs 6DOF even if symmetric</reason>
        is_quantum_theory = (
            hasattr(self.theory, 'category') and self.theory.category == 'quantum' or
            hasattr(self.theory, 'enable_quantum') and self.theory.enable_quantum or
            'quantum' in self.theory.__class__.__name__.lower()
        )
        
        # Choose solver based on theory properties
        if hasattr(self.theory, 'use_ugm_solver') and self.theory.use_ugm_solver:
            self.classical_solver = UGMGeodesicRK4Solver(
                self.theory, M, c, G,
                enable_quantum_corrections=self.enable_quantum
            )
        elif self.theory.__class__.__name__ == 'UnifiedGaugeModel':
            # <reason>chain: Explicitly handle UnifiedGaugeModel class</reason>
            self.classical_solver = UGMGeodesicRK4Solver(
                self.theory, M, c, G,
                enable_quantum_corrections=self.enable_quantum
            )
        elif hasattr(self.theory, 'charge') and self.theory.charge != 0:
            self.classical_solver = ChargedGeodesicRK4Solver(
                self.theory, M, c, G,
                q=self.theory.charge
            )
        elif is_quantum_theory:
            # <reason>chain: Force true 6DOF solver for quantum theories even if symmetric</reason>
            # Quantum theories need full phase space for proper quantum corrections
            self.classical_solver = GeneralGeodesicRK4Solver_6D(self.theory, M, c, G)
        elif self.theory.is_symmetric:
            self.classical_solver = GeodesicRK4Solver(self.theory, M, c, G)
        else:
            # <reason>chain: Use 4D version for non-quantum asymmetric theories for stability</reason>
            self.classical_solver = GeneralGeodesicRK4Solver_4D(self.theory, M, c, G)
            
    def _init_quantum_solver(self):
        """<reason>chain: Initialize quantum path integrator and loss calculator</reason>"""
        if not self.enable_quantum:
            self.quantum_integrator = None
            self.quantum_loss_calculator = None
            return
            
        self.quantum_integrator = self.theory.quantum_integrator
        self.quantum_loss_calculator = QuantumLossCalculator(self.quantum_integrator)
        
    def compute_classical_trajectory(self, initial_conditions: Dict, 
                                   time_steps: int = 1000,
                                   step_size: float = 0.01) -> Dict:
        """
        <reason>chain: Compute classical geodesic trajectory</reason>
        
        Args:
            initial_conditions: Dict with 'r', 'phi', 'E', 'Lz' (or full state)
                               NOTE: r should be in geometric units!
            time_steps: Number of integration steps
            step_size: Time step size (in geometric units)
            
        Returns:
            Dict with trajectory data
        """
        if not self.enable_classical or self.classical_solver is None:
            return {'error': 'Classical calculations disabled'}
        
        # <reason>chain: Check cache before computing</reason>
        cache_path = None
        if self._cache is not None:
            # Build cache key from all relevant parameters
            # <reason>chain: Remove r0 from cache_kwargs to avoid duplicate parameter error</reason>
            # r0 is passed as a positional argument to get_cache_path
            cache_kwargs = {
                'E': initial_conditions.get('E', 0.0),
                'Lz': initial_conditions.get('Lz', 0.0),
                'theory_module': self.theory.__class__.__module__,
                'theory_class': self.theory.__class__.__name__,
                'particle_name': initial_conditions.get('particle_name', 'unknown')
            }
            
            # Add theory-specific parameters
            metric_params = {}
            for param in ['a', 'q_e', 'alpha', 'beta', 'gamma', 'sigma', 'epsilon']:
                if hasattr(self.theory, param):
                    metric_params[param] = getattr(self.theory, param)
            if metric_params:
                cache_kwargs['metric_params'] = metric_params
                
            cache_path = self._cache.get_cache_path(
                self.theory.name,
                initial_conditions.get('r', 0.0) * self.length_scale,  # r0 in SI (positional arg)
                time_steps,
                step_size * self.time_scale,  # dtau in SI
                'float64',
                **cache_kwargs
            )
            
            # Try to load from cache
            cached_trajectory = self._cache.load_trajectory(cache_path, device='cpu', max_steps=time_steps)
            if cached_trajectory is not None:
                print(f"    Loaded trajectory from cache: {os.path.basename(cache_path)}")
                return {
                    'trajectory': cached_trajectory.numpy(),
                    'solver_type': 'cached',
                    'time_steps': len(cached_trajectory),
                    'step_size': step_size,
                    'cache_path': cache_path
                }
            
        # <reason>chain: Convert initial conditions from geometric to SI units</reason>
        # The engine passes r in geometric units, but classical solvers expect SI
        
        # Convert initial conditions to state vector
        if 'state' in initial_conditions:
            y0 = torch.tensor(initial_conditions['state'], dtype=torch.float64)
        else:
            # Build state from components
            t0 = initial_conditions.get('t', 0.0)
            r0_geom = initial_conditions['r']  # This is in geometric units
            
            # <reason>chain: Convert r from geometric to SI units</reason>
            r0_si = r0_geom * self.length_scale  # Convert to meters
            
            phi0 = initial_conditions.get('phi', 0.0)
            

            
            if isinstance(self.classical_solver, GeodesicRK4Solver):
                # 4D state: [t, r, phi, dr/dtau]
                # <reason>chain: GeodesicRK4Solver works in geometric units internally</reason>
                # Need to convert to geometric units
                E = initial_conditions['E']
                Lz = initial_conditions['Lz']
                
                # Set conserved quantities first
                self.classical_solver.E = E
                self.classical_solver.Lz = Lz
                

                
                # <reason>chain: 4D solver already works in geometric units - use r0_geom directly</reason>
                t0_geom = t0 / self.time_scale
                # r0_geom is already in geometric units - don't convert again!
                
                # Get radial velocity - convert u_r (SI) to dr/dtau (geometric)
                # u_r is in SI units (m/s), need to convert to geometric units
                u_r_si = initial_conditions.get('u_r', 0.0)
                dr_dtau0 = u_r_si * self.time_scale / self.length_scale  # Convert to geometric
                
                # Also check if dr_dtau is directly provided
                dr_dtau0 = initial_conditions.get('dr_dtau', dr_dtau0)
                
                y0 = torch.tensor([t0_geom, r0_geom, phi0, dr_dtau0], dtype=torch.float64)
            elif isinstance(self.classical_solver, GeneralGeodesicRK4Solver_6D):
                # <reason>chain: True 6D solver expects [t, r, phi, u^t, u^r, u^phi] in SI units</reason>
                u_t = initial_conditions.get('u_t', 1.0)
                u_r = initial_conditions.get('u_r', 0.0)
                u_phi = initial_conditions.get('u_phi', 0.0)
                
                y0 = torch.tensor([t0, r0_si, phi0, u_t, u_r, u_phi], dtype=torch.float64)
            else:
                # <reason>chain: 4D GeneralGeodesicRK4Solver_4D still uses conserved quantities</reason>
                E = initial_conditions['E']
                Lz = initial_conditions['Lz']
                
                # Set conserved quantities
                if hasattr(self.classical_solver, 'E'):
                    self.classical_solver.E = E
                if hasattr(self.classical_solver, 'Lz'):
                    self.classical_solver.Lz = Lz
                
                # Convert to geometric units for 4D solver
                t0_geom = t0 / self.time_scale
                u_r_si = initial_conditions.get('u_r', 0.0)
                dr_dtau0 = u_r_si * self.time_scale / self.length_scale
                
                y0 = torch.tensor([t0_geom, r0_geom, phi0, dr_dtau0], dtype=torch.float64)
                
        # Integrate trajectory
        trajectory = [y0.detach().numpy()]
        y = y0
        
        # <reason>chain: Step size handling depends on solver type</reason>
        if isinstance(self.classical_solver, GeodesicRK4Solver):
            # GeodesicRK4Solver expects step size in geometric units as tensor
            h = torch.tensor(step_size, dtype=torch.float64)
            
            for i in range(time_steps):
                y_new = self.classical_solver.rk4_step(y, h)
                if y_new is None:
                    break
                y = y_new
                trajectory.append(y.detach().numpy())
        else:
            # <reason>chain: GeneralGeodesicRK4Solver expects float step size in SI units</reason>
            h_si = step_size * self.time_scale
            
            for i in range(time_steps):
                y_new = self.classical_solver.rk4_step(y, h_si)
                if y_new is None:
                    break
                y = y_new
                trajectory.append(y.detach().numpy())
            

        
        # <reason>chain: Convert trajectory back to SI units if using geometric solver</reason>
        trajectory_array = np.array(trajectory)
        if isinstance(self.classical_solver, GeodesicRK4Solver):
            # Convert from geometric to SI units
            trajectory_array[:, 0] *= self.time_scale  # time
            trajectory_array[:, 1] *= self.length_scale  # radius
            # phi and dr/dtau stay the same
        
        # <reason>chain: Save trajectory to cache if caching is enabled</reason>
        if self._cache is not None and cache_path is not None:
            # Save as torch tensor
            trajectory_tensor = torch.tensor(trajectory_array, dtype=torch.float64)
            torch.save(trajectory_tensor, cache_path)
            print(f"    Trajectory saved to cache: {os.path.basename(cache_path)}")
            
        return {
            'trajectory': trajectory_array,
            'solver_type': type(self.classical_solver).__name__,
            'time_steps': len(trajectory),
            'step_size': step_size,
            'cache_path': cache_path if cache_path else None
        }
        
    def compute_quantum_trajectory(self, start_state: Tuple, end_state: Tuple,
                                 method: str = 'monte_carlo',
                                 num_samples: int = 1000,
                                 **kwargs) -> Dict:
        """
        <reason>chain: Compute quantum trajectory using path integrals</reason>
        
        Args:
            start_state: Initial (t, r, theta, phi)
            end_state: Final (t, r, theta, phi)
            method: 'monte_carlo' or 'wkb'
            num_samples: Number of paths to sample (for Monte Carlo)
            
        Returns:
            Dict with quantum trajectory data
        """
        if not self.enable_quantum or self.quantum_integrator is None:
            return {'error': 'Quantum calculations disabled'}
            
        # Compute amplitude and probability
        amplitude = None
        if method == 'monte_carlo':
            amplitude = self.quantum_integrator.compute_amplitude_monte_carlo(
                start_state, end_state, num_samples=num_samples, **kwargs
            )
        elif method == 'wkb':
            amplitude = self.quantum_integrator.compute_amplitude_wkb(
                start_state, end_state, **kwargs
            )
            
        probability = abs(amplitude) ** 2 if amplitude is not None else 0.0
        
        # Generate sample paths for visualization
        viz_data = visualize_quantum_paths(
            self.quantum_integrator,
            start_state, end_state,
            num_paths=min(10, num_samples // 100),
            **kwargs  # Pass particle properties
        )
        
        # Compute quantum corrections
        # Create a simple classical path for correction calculation
        classical_path = [start_state, end_state]
        corrections = self.quantum_integrator.compute_quantum_corrections(
            classical_path, **kwargs
        )
        
        return {
            'amplitude': complex(amplitude) if amplitude is not None else None,
            'probability': float(probability),
            'method': method,
            'num_samples': num_samples,
            'visualization_paths': viz_data['paths'],
            'path_actions': viz_data['actions'],
            'quantum_corrections': corrections
        }
        
    def compute_unified_trajectory(self, initial_conditions: Dict,
                                 final_state: Optional[Tuple] = None,
                                 **kwargs) -> Dict:
        """
        <reason>chain: Compute both classical and quantum trajectories</reason>
        
        Returns combined results from both methods.
        """
        results = {
            'theory': self.theory.name,
            'classical_enabled': self.enable_classical,
            'quantum_enabled': self.enable_quantum
        }
        
        # Classical trajectory
        if self.enable_classical:
            # Extract classical-specific kwargs - only pass what compute_classical_trajectory expects
            allowed_classical_kwargs = {'time_steps', 'step_size'}
            classical_kwargs = {k: v for k, v in kwargs.items() 
                               if k in allowed_classical_kwargs}
            classical_results = self.compute_classical_trajectory(
                initial_conditions, **classical_kwargs
            )
            results['classical'] = classical_results
            
            # Use classical endpoint for quantum if not specified
            if final_state is None and 'trajectory' in classical_results:
                traj = classical_results['trajectory']
                if len(traj) > 0:
                    final_point = traj[-1]
                    # Convert to (t, r, theta, phi) format
                    if len(final_point) >= 3:
                        # <reason>chain: Ensure proper time evolution for quantum paths</reason>
                        # If time hasn't evolved, use step_size * time_steps
                        t_final = float(final_point[0])
                        if abs(t_final) < 1e-10:  # Time hasn't evolved
                            t_final = kwargs.get('step_size', 0.01) * kwargs.get('time_steps', 1000) * self.time_scale
                        
                        final_state = (
                            t_final,  # t in SI units
                            float(final_point[1]),  # r in SI units
                            np.pi/2,  # theta (equatorial)
                            float(final_point[2]) if len(final_point) > 2 else 0.0  # phi
                        )
                        
        # Quantum trajectory
        if self.enable_quantum and final_state is not None:
            # Extract start state from initial conditions
            # <reason>chain: Convert r to SI units for quantum calculations</reason>
            r_si = initial_conditions['r'] * self.length_scale
            start_state = (
                initial_conditions.get('t', 0.0),
                r_si,  # Convert to SI units
                initial_conditions.get('theta', np.pi/2),
                initial_conditions.get('phi', 0.0)
            )
            
            quantum_results = self.compute_quantum_trajectory(
                start_state, final_state,
                method=kwargs.get('quantum_method', 'monte_carlo'),
                num_samples=kwargs.get('quantum_samples', 1000)
            )
            results['quantum'] = quantum_results
            
        # Combined loss calculation
        if self.enable_classical and self.enable_quantum:
            results['unified_metrics'] = self._compute_unified_metrics(results)
            
        return results
        
    def _compute_unified_metrics(self, results: Dict) -> Dict:
        """<reason>chain: Compute metrics combining classical and quantum results</reason>"""
        metrics = {}
        
        # Extract key values
        if 'classical' in results and 'trajectory' in results['classical']:
            classical_traj = results['classical']['trajectory']
            metrics['classical_path_length'] = self._compute_path_length(classical_traj)
            
        if 'quantum' in results:
            metrics['quantum_probability'] = results['quantum'].get('probability', 0.0)
            corrections = results['quantum'].get('quantum_corrections', {})
            metrics['decoherence_time'] = corrections.get('decoherence_time', float('inf'))
            
        # Classical-quantum correspondence
        if 'classical_path_length' in metrics and 'quantum_probability' in metrics:
            # High probability should correspond to classical path
            metrics['correspondence_score'] = metrics['quantum_probability']
            
        return metrics
        
    def _compute_path_length(self, trajectory: np.ndarray) -> float:
        """<reason>chain: Compute proper length along trajectory</reason>"""
        if len(trajectory) < 2:
            return 0.0
            
        length = 0.0
        for i in range(1, len(trajectory)):
            dt = trajectory[i][0] - trajectory[i-1][0]
            dr = trajectory[i][1] - trajectory[i-1][1]
            dphi = trajectory[i][2] - trajectory[i-1][2] if len(trajectory[i]) > 2 else 0.0
            
            # Simplified proper length (Minkowski approximation)
            ds = np.sqrt(abs(dt**2 - dr**2 - trajectory[i][1]**2 * dphi**2))
            length += ds
            
        return float(length)


def main():
    """<reason>chain: Command line interface for unified trajectory calculations</reason>"""
    parser = argparse.ArgumentParser(
        description='Unified Classical-Quantum Trajectory Calculator for Gravitational Theories'
    )
    
    # Theory selection
    parser.add_argument('theory', type=str, help='Theory name (e.g., kerr, weyl_em, einstein_unified)')
    
    # Calculation modes
    parser.add_argument('--classical-only', action='store_true',
                       help='Only compute classical geodesic trajectories')
    parser.add_argument('--quantum-only', action='store_true',
                       help='Only compute quantum path integral trajectories')
    parser.add_argument('--disable-quantum', action='store_true',
                       help='Disable quantum calculations entirely')
    
    # Initial conditions
    parser.add_argument('--r0', type=float, default=10.0,
                       help='Initial radial coordinate (in Schwarzschild radii)')
    parser.add_argument('--E', type=float, default=0.95,
                       help='Energy per unit mass (for symmetric spacetimes)')
    parser.add_argument('--Lz', type=float, default=4.0,
                       help='Angular momentum per unit mass')
    
    # Quantum parameters
    parser.add_argument('--quantum-method', choices=['monte_carlo', 'wkb'],
                       default='monte_carlo', help='Quantum calculation method')
    parser.add_argument('--quantum-samples', type=int, default=1000,
                       help='Number of paths to sample (Monte Carlo)')
    
    # Integration parameters
    parser.add_argument('--time-steps', type=int, default=1000,
                       help='Number of time steps for classical integration')
    parser.add_argument('--step-size', type=float, default=0.01,
                       help='Time step size')
    
    # Theory parameters
    parser.add_argument('--theory-params', type=str, default='{}',
                       help='JSON string of theory-specific parameters')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine calculation modes
    enable_classical = not args.quantum_only
    enable_quantum = not args.classical_only and not args.disable_quantum
    
    # Load theory
    try:
        # loader = TheoryLoader()  # Commented out to avoid import issues
        theory_params = json.loads(args.theory_params)
        
        # Add quantum enable flag to theory params
        theory_params['enable_quantum'] = enable_quantum
        
        # For demo, create a simple theory instead of loading
        print(f"Note: TheoryLoader disabled. Create your theory manually for testing.")
        sys.exit(1)
        
        # theory = loader.load_theory(args.theory, **theory_params)
        
        if args.verbose:
            print(f"Loaded theory: {theory.name}")
            print(f"Classical enabled: {enable_classical}")
            print(f"Quantum enabled: {enable_quantum and theory.enable_quantum}")
            
    except Exception as e:
        print(f"Error loading theory: {e}")
        sys.exit(1)
        
    # Create calculator
    calculator = UnifiedTrajectoryCalculator(
        theory, 
        enable_quantum=enable_quantum,
        enable_classical=enable_classical
    )
    
    # Set up initial conditions
    M = 1.989e30  # Solar mass in kg
    c = 2.998e8   # Speed of light in m/s
    G = 6.674e-11 # Gravitational constant
    rs = 2 * G * M / c**2  # Schwarzschild radius
    
    initial_conditions = {
        'r': args.r0 * rs,
        'E': args.E,
        'Lz': args.Lz,
        't': 0.0,
        'phi': 0.0
    }
    
    # Compute trajectory
    try:
        results = calculator.compute_unified_trajectory(
            initial_conditions,
            time_steps=args.time_steps,
            step_size=args.step_size,
            quantum_method=args.quantum_method,
            quantum_samples=args.quantum_samples,
            **theory_params
        )
        
        # Add metadata
        results['command_line_args'] = vars(args)
        results['units'] = {
            'distance': 'meters',
            'time': 'seconds',
            'mass': 'kg'
        }
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                def convert_arrays(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, complex):
                        return {'real': obj.real, 'imag': obj.imag}
                    elif isinstance(obj, dict):
                        return {k: convert_arrays(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_arrays(v) for v in obj]
                    return obj
                    
                json.dump(convert_arrays(results), f, indent=2)
                print(f"Results saved to {args.output}")
        else:
            # Print summary to stdout
            print("\n=== Unified Trajectory Results ===")
            print(f"Theory: {results['theory']}")
            
            if 'classical' in results and 'trajectory' in results['classical']:
                traj = results['classical']['trajectory']
                print(f"\nClassical trajectory:")
                print(f"  Steps computed: {len(traj)}")
                print(f"  Final position: r={traj[-1][1]/rs:.2f} rs")
                
            if 'quantum' in results:
                print(f"\nQuantum results:")
                print(f"  Transition probability: {results['quantum']['probability']:.6e}")
                print(f"  Method: {results['quantum']['method']}")
                
                if 'quantum_corrections' in results['quantum']:
                    corr = results['quantum']['quantum_corrections']
                    print(f"  Decoherence time: {corr['decoherence_time']:.3e} s")
                    print(f"  Position uncertainty: {corr['uncertainty_radius']:.3e} m")
                    
            if 'unified_metrics' in results:
                print(f"\nUnified metrics:")
                metrics = results['unified_metrics']
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"Error computing trajectory: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 