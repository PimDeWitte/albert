#!/usr/bin/env python3
"""
Comprehensive trajectory test showing loss vs Kerr baseline over time.
Based on the trajectory test from theory_engine_core.py.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
import pandas as pd

# Import engine and constants
from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT

# Import all theories from test_theories_final.py
from physics_agent.theories.newtonian_limit.theory import NewtonianLimit
from physics_agent.theories.defaults.baselines.kerr import Kerr
from physics_agent.theories.defaults.baselines.kerr_newman import KerrNewman
from physics_agent.theories.yukawa.theory import Yukawa
from physics_agent.theories.einstein_teleparallel.theory import EinsteinTeleparallel
from physics_agent.theories.spinor_conformal.theory import SpinorConformal
from physics_agent.theories.defaults.baselines.schwarzschild import Schwarzschild
from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
from physics_agent.theories.string.theory import StringTheory
from physics_agent.theories.asymptotic_safety.theory import AsymptoticSafetyTheory
from physics_agent.theories.loop_quantum_gravity.theory import LoopQuantumGravity
from physics_agent.theories.non_commutative_geometry.theory import NonCommutativeGeometry
from physics_agent.theories.twistor_theory.theory import TwistorTheory
from physics_agent.theories.aalto_gauge_gravity.theory import AaltoGaugeGravity
from physics_agent.theories.causal_dynamical_triangulations.theory import CausalDynamicalTriangulations

# All theories to test
ALL_THEORIES = [
    # Baseline
    ("Schwarzschild", Schwarzschild, "baseline"),
    ("Kerr", Kerr, "baseline"),
    
    # Classical accepted
    ("Newtonian Limit", NewtonianLimit, "classical"),
    ("Kerr-Newman", KerrNewman, "classical"),
    ("Yukawa", Yukawa, "classical"),
    ("Einstein Teleparallel", EinsteinTeleparallel, "classical"),
    ("Spinor Conformal", SpinorConformal, "classical"),
    
    # Quantum accepted/in-progress
    ("Quantum Corrected", QuantumCorrected, "quantum"),
    ("String Theory", StringTheory, "quantum"),
    ("Asymptotic Safety", AsymptoticSafetyTheory, "quantum"),
    ("Loop Quantum Gravity", LoopQuantumGravity, "quantum"),
    ("Non-Commutative Geometry", NonCommutativeGeometry, "quantum"),
    ("Twistor Theory", TwistorTheory, "quantum"),
    ("Aalto Gauge Gravity", AaltoGaugeGravity, "quantum"),
    ("Causal Dynamical Triangulations", CausalDynamicalTriangulations, "quantum"),
]

def run_trajectory_with_loss(theory_name, theory_class, category, kerr_trajectory=None, 
                           n_steps=1000, compute_interval=10):
    """
    Run trajectory integration and compute loss vs Kerr at regular intervals.
    
    Args:
        theory_name: Name of theory
        theory_class: Theory class to instantiate
        category: Theory category (baseline/classical/quantum)
        kerr_trajectory: Kerr baseline trajectory for comparison
        n_steps: Number of integration steps
        compute_interval: Compute loss every N steps
        
    Returns:
        trajectory: Full trajectory
        loss_history: List of (step, loss) tuples
        timing_info: Timing statistics
    """
    print(f"\n{'='*60}")
    print(f"Testing: {theory_name} [{category}]")
    print(f"{'='*60}")
    
    # Initialize theory
    try:
        if theory_name == "Kerr":
            theory = theory_class(a=0.0)  # Non-rotating
        elif theory_name == "Kerr-Newman":
            theory = theory_class(a=0.0, Q=0.0)  # Non-rotating, uncharged
        else:
            theory = theory_class()
        print(f"Initialized: {theory.name}")
    except Exception as e:
        print(f"ERROR: Failed to initialize - {e}")
        return None, None, None
    
    # Create engine
    engine = TheoryEngine(verbose=False)
    
    # Initial conditions for circular orbit
    r0_si = 10 * 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
    dtau_si = 0.01
    
    # Convert to geometric units
    M_phys = torch.tensor(SOLAR_MASS, dtype=torch.float64)
    r0_geom = r0_si / (GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2)
    dtau_geom = dtau_si / (GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**3)
    
    # Get initial conditions
    try:
        y0_symmetric, y0_general, solver_info = engine.get_initial_conditions(theory, torch.tensor(r0_geom))
        
        # Create appropriate solver
        if hasattr(theory, 'has_conserved_quantities') and theory.has_conserved_quantities:
            from physics_agent.geodesic_integrator import ConservedQuantityGeodesicSolver
            solver = GeodesicRK4Solver(theory, M_phys)
            if isinstance(solver_info, dict):
                solver.E = solver_info.get('E', 0.95)
                solver.Lz = solver_info.get('Lz', 4.0)
            else:
                solver.E = 0.95
                solver.Lz = 4.0
            y0 = y0_symmetric  # 4D state
        else:
            from physics_agent.geodesic_integrator import GeneralRelativisticGeodesicSolver
            solver = GeneralGeodesicRK4Solver(theory, M_phys)
            y0 = y0_general  # 6D state
            
    except Exception as e:
        print(f"ERROR: Failed to initialize solver - {e}")
        return None, None, None
    
    # Initialize storage
    trajectory = []
    loss_history = []
    timing_info = {
        'total_time': 0.0,
        'solver_time': 0.0,
        'steps': 0
    }
    
    # Run integration
    print(f"\nIntegrating {n_steps} steps...")
    start_time = time.time()
    solver_time_total = 0.0
    
    y = y0.clone()
    trajectory.append(y.clone())
    
    with tqdm(total=n_steps, desc=f"{theory_name[:20]:<20}") as pbar:
        for step in range(n_steps):
            try:
                # Time the solver step
                solver_start = time.time()
                y_new = solver.rk4_step(y, dtau_geom)
                solver_time = time.time() - solver_start
                solver_time_total += solver_time
                
                if y_new is None:
                    print(f"\nIntegration failed at step {step}")
                    break
                    
                y = y_new
                trajectory.append(y.clone())
                
                # Compute loss vs Kerr at intervals
                if kerr_trajectory is not None and step % compute_interval == 0:
                    if step < len(kerr_trajectory) - 1:
                        # Compare full state vectors
                        r_theory = y[1]  # r is at index 1
                        r_kerr = kerr_trajectory[step+1][1]
                        
                        # Compute different loss metrics
                        radial_loss = float((r_theory - r_kerr)**2)
                        
                        # For angular momentum, compare phi evolution
                        phi_theory = y[2]  # phi at index 2
                        phi_kerr = kerr_trajectory[step+1][2]
                        angular_loss = float((phi_theory - phi_kerr)**2)
                        
                        # Combined loss
                        total_loss = radial_loss + 0.1 * angular_loss  # Weight angular less
                        
                        loss_history.append({
                            'step': step,
                            'radial_loss': radial_loss,
                            'angular_loss': angular_loss,
                            'total_loss': total_loss,
                            'r_theory': float(r_theory),
                            'r_kerr': float(r_kerr)
                        })
                
                # Check for horizon crossing
                r_current = y[1]
                if r_current <= 2.1:
                    print(f"\nReached horizon at step {step}")
                    break
                    
                # Update progress bar
                if step % 50 == 0:
                    pbar.set_postfix({
                        'r': f"{r_current:.3f}",
                        'ms/step': f"{solver_time*1000:.3f}"
                    })
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError at step {step}: {e}")
                break
    
    total_time = time.time() - start_time
    
    # Calculate timing statistics
    timing_info['total_time'] = total_time
    timing_info['solver_time'] = solver_time_total
    timing_info['steps'] = len(trajectory) - 1
    timing_info['time_per_step'] = solver_time_total / max(1, len(trajectory) - 1)
    
    print(f"\nCompleted {len(trajectory)-1} steps in {total_time:.3f}s")
    print(f"Average: {timing_info['time_per_step']*1000:.3f}ms/step")
    
    return trajectory, loss_history, timing_info

def plot_loss_evolution(results: Dict[str, Tuple], save_path: str = "loss_vs_kerr_evolution.png"):
    """Create comprehensive loss evolution plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Radial loss evolution
    ax1.set_title("Radial Loss vs Kerr Over Time", fontsize=14)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("(r_theory - r_kerr)²")
    ax1.set_yscale('log')
    
    # 2. Angular loss evolution  
    ax2.set_title("Angular Loss vs Kerr Over Time", fontsize=14)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("(φ_theory - φ_kerr)²")
    ax2.set_yscale('log')
    
    # 3. Total loss evolution
    ax3.set_title("Combined Loss Evolution", fontsize=14)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Total Loss")
    ax3.set_yscale('log')
    
    # 4. Theory ranking by final loss
    ax4.set_title("Theory Ranking by Average Loss", fontsize=14)
    
    # Colors for different categories
    colors = {
        'baseline': 'black',
        'classical': 'blue',
        'quantum': 'red'
    }
    
    final_losses = []
    
    for theory_name, (trajectory, loss_history, timing, category) in results.items():
        if loss_history and len(loss_history) > 0:
            # Convert to arrays for plotting
            steps = [l['step'] for l in loss_history]
            radial_losses = [l['radial_loss'] for l in loss_history]
            angular_losses = [l['angular_loss'] for l in loss_history]
            total_losses = [l['total_loss'] for l in loss_history]
            
            color = colors.get(category, 'gray')
            linestyle = '-' if category == 'quantum' else '--' if category == 'classical' else ':'
            
            # Plot losses
            ax1.plot(steps, radial_losses, label=theory_name, color=color, 
                    linestyle=linestyle, alpha=0.8)
            ax2.plot(steps, angular_losses, label=theory_name, color=color, 
                    linestyle=linestyle, alpha=0.8)
            ax3.plot(steps, total_losses, label=theory_name, color=color, 
                    linestyle=linestyle, alpha=0.8)
            
            # Calculate average loss for ranking
            avg_loss = np.mean(total_losses)
            final_losses.append((theory_name, avg_loss, category))
    
    # Create legend for first 3 plots
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot ranking
    final_losses.sort(key=lambda x: x[1])  # Sort by loss
    theories = [f[0] for f in final_losses]
    losses = [f[1] for f in final_losses]
    categories = [f[2] for f in final_losses]
    
    # Create color array for bars
    bar_colors = [colors.get(cat, 'gray') for cat in categories]
    
    y_pos = np.arange(len(theories))
    bars = ax4.barh(y_pos, losses, color=bar_colors, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(theories, fontsize=10)
    ax4.set_xlabel("Average Total Loss")
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, loss) in enumerate(zip(bars, losses)):
        ax4.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2, 
                f'{loss:.2e}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nLoss evolution plot saved to: {save_path}")

def create_summary_table(results: Dict[str, Tuple]) -> pd.DataFrame:
    """Create summary table of results"""
    summary_data = []
    
    for theory_name, (trajectory, loss_history, timing, category) in results.items():
        if loss_history and len(loss_history) > 0:
            # Calculate statistics
            total_losses = [l['total_loss'] for l in loss_history]
            radial_losses = [l['radial_loss'] for l in loss_history]
            
            summary_data.append({
                'Theory': theory_name,
                'Category': category,
                'Steps Completed': timing['steps'],
                'Avg Loss': np.mean(total_losses),
                'Min Loss': np.min(total_losses),
                'Max Loss': np.max(total_losses),
                'Final Loss': total_losses[-1],
                'ms/step': timing['time_per_step'] * 1000
            })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Avg Loss')
    return df

def main():
    """Run comprehensive trajectory loss analysis"""
    print("TRAJECTORY LOSS ANALYSIS VS KERR BASELINE")
    print("="*80)
    print(f"Testing {len(ALL_THEORIES)} theories with 1000 steps each")
    print("Computing loss metrics every 10 steps")
    
    results = {}
    kerr_trajectory = None
    
    # First, run Kerr to get baseline
    for theory_name, theory_class, category in ALL_THEORIES:
        if theory_name == "Kerr":
            print("\nGenerating Kerr baseline trajectory...")
            trajectory, _, timing = run_trajectory_with_loss(
                theory_name, theory_class, category, 
                kerr_trajectory=None,  # No comparison for baseline
                n_steps=1000
            )
            if trajectory is not None:
                kerr_trajectory = trajectory
                # Store Kerr results without loss (it's the baseline)
                results[theory_name] = (trajectory, [], timing, category)
            break
    
    if kerr_trajectory is None:
        print("ERROR: Failed to generate Kerr baseline trajectory")
        return
    
    # Run all other theories
    for theory_name, theory_class, category in ALL_THEORIES:
        if theory_name == "Kerr":
            continue  # Already done
            
        trajectory, loss_history, timing = run_trajectory_with_loss(
            theory_name, theory_class, category,
            kerr_trajectory=kerr_trajectory,
            n_steps=1000,
            compute_interval=10
        )
        
        if trajectory is not None:
            results[theory_name] = (trajectory, loss_history, timing, category)
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_loss_evolution(results)
    
    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    df = create_summary_table(results)
    print("\n" + df.to_string(index=False))
    
    # Save results
    df.to_csv("trajectory_loss_summary.csv", index=False)
    print("\nSummary saved to: trajectory_loss_summary.csv")
    
    # Highlight key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Best performing theories
    if len(df) > 0:
        print("\nTop 3 theories by average loss:")
        for i, row in df.head(3).iterrows():
            print(f"  {i+1}. {row['Theory']} ({row['Category']}): {row['Avg Loss']:.3e}")
        
        # Category comparison
        print("\nAverage loss by category:")
        category_avg = df.groupby('Category')['Avg Loss'].mean()
        for cat, avg in category_avg.items():
            print(f"  {cat}: {avg:.3e}")
        
        # Quantum vs Classical
        quantum_theories = df[df['Category'] == 'quantum']
        classical_theories = df[df['Category'] == 'classical']
        
        if len(quantum_theories) > 0 and len(classical_theories) > 0:
            print(f"\nBest quantum theory: {quantum_theories.iloc[0]['Theory']} ({quantum_theories.iloc[0]['Avg Loss']:.3e})")
            print(f"Best classical theory: {classical_theories.iloc[0]['Theory']} ({classical_theories.iloc[0]['Avg Loss']:.3e})")
    
    return results

if __name__ == "__main__":
    results = main()