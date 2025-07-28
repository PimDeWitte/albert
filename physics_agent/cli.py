#!/usr/bin/env python3
"""
CLI Module - Handles all command-line argument parsing and processing
<reason>chain: Separating CLI logic improves modularity and testability</reason>

This module provides the argument parser for the theory engine core.
It can be used directly via `python -m physics_agent.theory_engine_core`
or through the unified `albert run` command.
"""
import argparse
import sys
import torch
import numpy as np

def get_cli_parser() -> argparse.ArgumentParser:
    """
    Creates and returns the command-line argument parser.
    <reason>chain: Centralized CLI parser makes it easier to maintain and test command-line interface</reason>
    """
    p = argparse.ArgumentParser(
        description="Core engine for gravitational theory simulation and evaluation.",
        epilog="""
By default, theories use their preferred parameter values based on theoretical physics:
  - Einstein Final: α=0 (reduces to Schwarzschild)
  - Linear Signal Loss: γ=0.75 (achieves unification)
  - Log Corrected: β=0.5 (best performance from paper)
  
Use --enable-sweeps to test full parameter ranges instead of preferred values.
Use --sweep-only [param] to sweep only specific parameters (e.g., --sweep-only gamma).

PARALLELIZATION NOTE:
  - Parameter sweeps: Automatically parallelized when --enable-sweeps is used
  - Multiple theories: Run separate instances in parallel using shell job control (&)
    Example: python -m physics_agent.theory_engine_core --theories theory1 &
             python -m physics_agent.theory_engine_core --theories theory2 &
        """
    )

    # <reason>chain: Group related arguments for better organization</reason>
    # Execution mode arguments
    p.add_argument("--final", action="store_true", 
                   help="Run with final, high-step-count parameters for publication-quality data.")
    p.add_argument("--cpu-f64", action="store_true", 
                   help="Run on CPU with float64 precision for validation. Overrides default GPU/float32 settings.")
    p.add_argument("--gpu-f32", action="store_true", 
                   help="Run on GPU with float32 precision for speed. Uses CUDA if available, otherwise MPS (Apple Silicon).")
    
    # Theory selection arguments
    # <reason>chain: Allow running theories by category for unified theories like UGM</reason>
    p.add_argument("--category", type=str, default=None,
                   help="Run all theories in a specific category (e.g., 'ugm' for Unified Gravity Model, 'quantum', 'classical')")
    
    # Loss calculation arguments
    # <reason>chain: Ricci tensor is the fundamental measure of spacetime curvature - no other loss types needed</reason>
    # Loss type is now fixed to 'ricci' - the only scientifically rigorous comparison method
    
    # Cache arguments
    p.add_argument("--no-cache", action="store_true", 
                   help="Force recomputation by ignoring existing cache files.")
    p.add_argument("--clear-cache", action="store_true", 
                   help="Clear the trajectory cache directory and exit.")
    
    # Device and precision arguments
    p.add_argument("--device", type=str, default=None, 
                   help="PyTorch device (e.g., 'mps', 'cpu').")
    p.add_argument("--dtype", type=str, default=None, 
                   help="PyTorch data type (e.g., 'float32', 'float64').")
    
    # Experimental quantum features
    p.add_argument("--experimental", action="store_true", 
                   help="""Enable experimental quantum kick feature (unvalidated). 
                   Uses default interval=1000 steps and beta=0.01. This feature simulates stochastic quantum fluctuations 
                   in the geodesic trajectory but has NOT been validated against any quantum gravity theory.""")
    p.add_argument("--experimental-quantum-interval", type=int, default=1000, 
                   help="""[EXPERIMENTAL] Number of integration steps between quantum kicks.
                   Only used when --experimental is enabled. Default: 1000 steps.
                   Lower values = more frequent kicks = stronger quantum effects.
                   Example: --experimental --experimental-quantum-interval 500""")
    p.add_argument("--experimental-quantum-beta", type=float, default=0.01, 
                   help="""[EXPERIMENTAL] Magnitude of quantum velocity perturbations.
                   Only used when --experimental is enabled. Default: 0.01.
                   This scales the random normal distribution used for velocity kicks.
                   Higher values = stronger kicks = more deviation from classical trajectory.
                   Typical range: 0.001 (subtle) to 0.1 (strong).
                   Example: --experimental --experimental-quantum-beta 0.05""")
    p.add_argument("--experimental-quantum-method", type=str, default="kicks", 
                   choices=["kicks", "circuits"], 
                   help="""[EXPERIMENTAL] Method for quantum corrections. Default: 'kicks'.
                   'kicks': Simple stochastic velocity perturbations (current implementation).
                   'circuits': [NOT IMPLEMENTED] Future PennyLane quantum circuit integration.
                   The 'circuits' method would use quantum circuits to compute corrections based on
                   quantum field theory in curved spacetime, but requires significant development.""")
    
    # Quantum Lagrangian configuration
    p.add_argument("--quantum-field-content", type=str, default='all',
                   choices=['all', 'minimal', 'gauge', 'matter'],
                   help="Which quantum field content to include for quantum theories (default: all)")
    p.add_argument("--quantum-phase-precision", type=float, default=1e-30,
                   help="Precision for quantum phase calculations (default: 1e-30)")
    
    # Control arguments
    p.add_argument("--max-steps", "--steps", type=int, default=None, dest='steps',
                   help="Maximum number of simulation steps (may terminate early at event horizon). Default: 1000, Final mode: 100000")
    p.add_argument("-r", "--radius", type=float, default=6.0,
                   help="Starting radius in Schwarzschild radii (Rs = 2GM/c²). Default: 6.0 Rs. Use larger values (10-20) for longer trajectories before reaching event horizon.")
    p.add_argument("--early-stop", action="store_true", 
                   help="Enable early stopping based on loss convergence (credit: Ben Geist)")
    p.add_argument("--no-baselines", action="store_true",
                   help="Skip baseline theory calculations (saves memory)")
    
    # Filtering arguments
    p.add_argument("--theory-filter", type=str, default=None, 
                   help="Filter theories by name (substring match)")
    
    # Simulation parameters
    p.add_argument("--close-orbit", action="store_true", 
                   help="Use closer initial orbit (6RS) for stronger field effects in tests")
    
    # Parameter sweep control
    p.add_argument("--enable-sweeps", action="store_true",
                   help="Enable parameter sweeps for theories (default uses preferred values based on physics)")
    p.add_argument("--sweep-only", type=str, default=None,
                   help="Run sweep only for specified parameter (e.g., 'gamma' for Linear Signal Loss)")
    p.add_argument("--sweep-workers", type=int, default=None,
                   help="Override number of parallel workers (default: auto-detect based on CPU, RAM, and GPU resources)")
    p.add_argument("--sweep-memory-per-worker", type=float, default=2.0,
                   help="Estimated memory per worker in GB for resource planning (default: 2.0)")
    p.add_argument("--disable-resource-check", action="store_true",
                   help="Disable smart resource detection, fall back to simple CPU count (not recommended)")
    p.add_argument("--sweepable_fields", type=str, default='any', 
                   help="Text list (e.g., 'gamma,sigma') or JSON dict of fields to sweep (e.g., '{\"gamma\": [0.1,0.3]}'). Default: 'any' (all fields).")
    
    # Miscellaneous
    p.add_argument("--verbose", action="store_true", 
                   help="Enable verbose logging.")
    p.add_argument("--force-baseline-runs", action="store_true",
                   help="Force baseline theories to run full trajectory even if pre-flight checks fail")
    
    # Calibration options
    p.add_argument("--skip-calibration", action="store_true", 
                   help="Skip solver calibration tests before running theories")
    p.add_argument("--strict-calibration", action="store_true",
                   help="Exit if calibration tests fail (default: continue with warning)")
    p.add_argument("--skip-device-benchmark", action="store_true",
                   help="Skip device precision benchmarks during calibration (saves ~10s)")
    
    # Candidate mode arguments
    p.add_argument("--candidates", action="store_true",
                   help="Run in candidates mode: evaluate all theories in candidates/ folder along with regular theories")
    
    return p


def setup_deterministic_mode() -> None:
    """
    Enable deterministic mode for reproducible results.
    <reason>chain: Single responsibility - always enables deterministic mode when called</reason>
    <reason>chain: Deterministic mode ensures reproducible results for scientific validation</reason>
    """
    # <reason>chain: Set seeds for all random number generators to ensure reproducibility</reason>
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("Deterministic mode enabled: seeds set, non-deterministic optimizations disabled")


def handle_special_modes(args: argparse.Namespace) -> bool:
    """
    Handle special execution modes that bypass normal simulation.
    Returns True if special mode was handled and program should exit.
    <reason>chain: Separate special modes from main execution flow for clarity</reason>
    """
    if args.clear_cache:
        # <reason>chain: Cache clearing is now handled by cache module</reason>
        from .cache import clear_cache
        clear_cache()
        sys.exit(0)
        
    return False


def setup_execution_mode(args: argparse.Namespace) -> None:
    """
    Set up execution mode based on CLI arguments.
    <reason>chain: Always use deterministic mode for reproducibility</reason>
    """
    # <reason>chain: Always enable deterministic mode for reproducibility</reason>
    setup_deterministic_mode()


def determine_device_and_dtype(args: argparse.Namespace) -> tuple[str, torch.dtype]:
    """
    Determine the device and data type based on CLI arguments.
    <reason>chain: Centralized logic for device/dtype selection reduces duplication</reason>
    """
    device = args.device
    dtype = None
    
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float64":
        dtype = torch.float64
    elif args.gpu_f32:
        dtype = torch.float32
    elif args.final or args.cpu_f64:
        dtype = torch.float64
    else:
        dtype = torch.float64  # Default to float64
        
    # <reason>chain: Auto-select best available device if not specified</reason>
    if device is None:
        if args.cpu_f64:
            device = 'cpu'
        elif args.gpu_f32:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
                print("Warning: GPU requested but not available, falling back to CPU")
        else:
            device = 'cpu'
            
    return device, dtype


def get_simulation_parameters(args: argparse.Namespace) -> dict:
    """
    Extract simulation parameters from CLI arguments.
    <reason>chain: Consolidate parameter extraction for cleaner main function</reason>
    """
    params = {
        'loss_type': 'ricci', # Ricci tensor is the only loss type
        'no_cache': args.no_cache,
        'verbose': args.verbose,
        'early_stopping': args.early_stop,
    }
    
    # <reason>chain: Handle experimental quantum features</reason>
    if args.experimental:
        params['quantum_interval'] = args.experimental_quantum_interval
        params['quantum_beta'] = args.experimental_quantum_beta
        params['quantum_method'] = args.experimental_quantum_method
    else:
        params['quantum_interval'] = 0
        params['quantum_beta'] = 0.0
        params['quantum_method'] = None
        
    # <reason>chain: Determine initial radius based on orbit configuration</reason>
    if args.close_orbit:
        params['r0_multiplier'] = 6.0  # 6 RS for close orbit
    else:
        params['r0_multiplier'] = 12.0  # 12 RS for standard orbit
        
    # <reason>chain: Override step count if specified</reason>
    if args.steps is not None:
        params['override_steps'] = args.steps
    else:
        params['override_steps'] = None
        
    return params 