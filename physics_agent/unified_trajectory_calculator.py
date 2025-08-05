#!/usr/bin/env python3
"""
Unified Trajectory Calculator for Gravitational Theories.

Combines classical geodesic calculations with quantum path integral trajectories
to enable comprehensive gravity unification studies.

<reason>chain: Main entry point for unified classical-quantum trajectory calculations</reason>
"""

import argparse
import torch
import numpy as np
from typing import Dict, Tuple, Optional
import json
import sys

from physics_agent.base_theory import GravitationalTheory
from physics_agent.cache import TrajectoryCache
from physics_agent.geodesic_integrator_stable import (
    GeodesicRK4Solver, GeneralGeodesicRK4Solver, 
    ChargedGeodesicRK4Solver, UGMGeodesicRK4Solver
)
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
        elif self.theory.has_conserved_quantities:
            # <reason>chain: Use 4D solver for any stationary axisymmetric spacetime</reason>
            self.classical_solver = GeodesicRK4Solver(self.theory, M, c, G)
        else:
            self.classical_solver = GeneralGeodesicRK4Solver(self.theory, M, c, G)
            
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
                                   step_size: float = 0.01,
                                   particle_name: str = None) -> Dict:
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
            else:
                # 6D state: [t, r, phi, u^t, u^r, u^phi]
                # Note: velocities are already in SI units from the engine
                u_t = initial_conditions.get('u_t', 1.0)
                u_r = initial_conditions.get('u_r', 0.0)
                u_phi = initial_conditions.get('u_phi', 0.0)
                

                
                y0 = torch.tensor([t0, r0_si, phi0, u_t, u_r, u_phi], dtype=torch.float64)
                
        # Integrate trajectory
        trajectory = [y0.detach().numpy()]
        y = y0
        
        # <reason>chain: Debug which solver is being used</reason>
        # Map old solver names to new names for display
        solver_display_name = type(self.classical_solver).__name__
        name_mapping = {
            'GeodesicRK4Solver': 'ConservedQuantityGeodesicSolver',
            'GeneralGeodesicRK4Solver': 'GeneralRelativisticGeodesicSolver',
            'ChargedGeodesicRK4Solver': 'ChargedParticleGeodesicSolver',
            'NullGeodesicRK4Solver': 'PhotonGeodesicSolver',
            'UGMGeodesicRK4Solver': 'UnifiedGravityModelGeodesicSolver',
            'SymmetricChargedGeodesicRK4Solver': 'ConservedQuantityChargedGeodesicSolver',
            'QuantumGeodesicSimulator': 'QuantumCorrectedGeodesicSolver'
        }
        if solver_display_name in name_mapping:
            solver_display_name = name_mapping[solver_display_name]
        print(f"    Using solver: {solver_display_name}")
        
        # <reason>chain: Step size handling depends on solver type</reason>
        if isinstance(self.classical_solver, GeodesicRK4Solver):
            # GeodesicRK4Solver expects step size in geometric units
            h = torch.tensor(step_size, dtype=torch.float64)
        else:
            # Other solvers expect SI units
            h = torch.tensor(step_size * self.time_scale, dtype=torch.float64)
        
        # <reason>chain: Add progress bar for trajectory integration</reason>
        from tqdm import tqdm
        pbar_desc = f"      {particle_name}" if particle_name else "      Computing trajectory"
        pbar = tqdm(range(time_steps), 
                   desc=pbar_desc,
                   unit=' steps',
                   disable=False,  # Always show progress
                   leave=False,    # Don't leave the bar after completion
                   ncols=100,      # Fixed width for consistency
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for i in pbar:
            # <reason>chain: Convert h to float for rk4_step</reason>
            h_float = h.item() if torch.is_tensor(h) else h
            y_new = self.classical_solver.rk4_step(y, h_float)
            if y_new is None:
                print(f"    WARNING: Solver returned None at step {i}")
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
            
        return {
            'trajectory': trajectory_array,
            'solver_type': type(self.classical_solver).__name__,
            'time_steps': len(trajectory),
            'step_size': step_size
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
            # Check if quantum_integrator has the method (UnifiedQuantumSolver vs QuantumPathIntegrator)
            if hasattr(self.quantum_integrator, 'compute_amplitude_monte_carlo'):
                amplitude = self.quantum_integrator.compute_amplitude_monte_carlo(
                    start_state, end_state, num_samples=num_samples, **kwargs
                )
            elif hasattr(self.quantum_integrator, '_amplitude_monte_carlo'):
                amplitude = self.quantum_integrator._amplitude_monte_carlo(
                    start_state, end_state, num_paths=num_samples, **kwargs
                )
        elif method == 'wkb':
            if hasattr(self.quantum_integrator, 'compute_amplitude_wkb'):
                amplitude = self.quantum_integrator.compute_amplitude_wkb(
                    start_state, end_state, **kwargs
                )
            elif hasattr(self.quantum_integrator, '_amplitude_wkb'):
                amplitude = self.quantum_integrator._amplitude_wkb(
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
            # <reason>chain: Debug output to understand quantum trajectory computation</reason>
            print(f"\n  UnifiedTrajectoryCalculator: Computing classical trajectory for {self.theory.name}")
            print(f"    Initial conditions: r={initial_conditions.get('r', 'N/A')}, E={initial_conditions.get('E', 'N/A')}, Lz={initial_conditions.get('Lz', 'N/A')}")
            print(f"    Time steps: {kwargs.get('time_steps', 'N/A')}, Step size: {kwargs.get('step_size', 'N/A')}")
            
            # Extract classical-specific kwargs - only pass what compute_classical_trajectory expects
            allowed_classical_kwargs = {'time_steps', 'step_size', 'particle_name'}
            classical_kwargs = {k: v for k, v in kwargs.items() 
                               if k in allowed_classical_kwargs}
            classical_results = self.compute_classical_trajectory(
                initial_conditions, **classical_kwargs
            )
            results['classical'] = classical_results
            
            print(f"    Classical trajectory computed: {len(classical_results.get('trajectory', [])) if 'trajectory' in classical_results else 0} points")
            
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