#!/usr/bin/env python3
"""
Albert CLI - Main entry point for the albert command
Provides a unified interface for all Albert operations
"""
import sys
import argparse
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main entry point for albert CLI"""
    parser = argparse.ArgumentParser(
        prog='albert',
        description='🌌 Albert: Physics at The Speed of AI - A timely agent for gravitational theory research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  albert run                           # Evaluate all theories with analytical and trajectory tests
  albert run --candidates              # Include candidate theories from candidates/ folder
  albert run --candidates-only         # Evaluate only candidate theories
  albert run --theory-filter "kerr"    # Run only theories matching "kerr"
  albert run --max-steps 10000         # Run with longer trajectories
  albert run --no-cache                # Force fresh computation, bypass cache
  albert run --enable-sweeps           # Run with parameter sweeps
  albert run --sweep-only gamma        # Sweep only specific parameter
  albert run --test                    # Run pre-flight tests before evaluation
  
  albert discover                      # Start AI-powered self-discovery mode
  albert discover --initial "quantum"  # Guide discovery toward quantum theories
  
  albert test                          # Run environment/solver tests
  albert validate theories/mytheory.py # Validate a specific theory file
  albert setup                         # Configure Albert instance
  albert benchmark                     # Run model benchmarks

For more information on each command, use: albert <command> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command - runs comprehensive evaluation
    run_parser = subparsers.add_parser(
        'run', 
        help='Run comprehensive theory evaluation and ranking',
        description='Evaluate all theories against analytical and trajectory-based tests with interactive visualizations'
    )
    
    # Import CLI arguments from cli.py for full functionality
    from physics_agent.cli import get_cli_parser
    cli_parser = get_cli_parser()
    
    # Copy all arguments from the original parser to run subcommand
    for action in cli_parser._actions:
        if action.dest == 'help':  # Skip help action
            continue
        
        # Build kwargs for the argument
        kwargs = {
            'help': action.help,
            'default': action.default
        }
        
        # Handle different action types
        if action.__class__.__name__ == 'StoreAction' or action.__class__.__name__ == '_StoreAction':
            kwargs['action'] = 'store'
            if action.type is not None:
                kwargs['type'] = action.type
            if hasattr(action, 'nargs') and action.nargs is not None:
                kwargs['nargs'] = action.nargs
            if hasattr(action, 'choices') and action.choices is not None:
                kwargs['choices'] = action.choices
            if hasattr(action, 'metavar') and action.metavar is not None:
                kwargs['metavar'] = action.metavar
        elif action.__class__.__name__ in ['StoreTrueAction', '_StoreTrueAction']:
            kwargs['action'] = 'store_true'
        elif action.__class__.__name__ in ['StoreFalseAction', '_StoreFalseAction']:
            kwargs['action'] = 'store_false'
        else:
            kwargs['action'] = 'store'
            if hasattr(action, 'type') and action.type is not None:
                kwargs['type'] = action.type
            if hasattr(action, 'nargs') and action.nargs is not None:
                kwargs['nargs'] = action.nargs
            if hasattr(action, 'choices') and action.choices is not None:
                kwargs['choices'] = action.choices
            
        run_parser.add_argument(*action.option_strings, **kwargs)
    
    # Add evaluation-specific arguments that aren't in cli.py
    eval_group = run_parser.add_argument_group('evaluation options')
    eval_group.add_argument('--candidates-status', choices=['proposed', 'new', 'rejected', 'all'],
                       default='proposed', help='Which candidate theories to include when --candidates is used')
    eval_group.add_argument('--candidates-only', action='store_true',
                       help='Run ONLY candidate theories (excludes regular theories)')
    eval_group.add_argument('--test', action='store_true',
                       help='Run pre-flight environment tests before evaluation')
    

    
    # Setup command
    setup_parser = subparsers.add_parser(
        'setup',
        help='Configure your Albert instance',
        description='Set up API keys, cryptographic identity, and network participation'
    )
    
    # Test command - runs environment/solver tests
    test_parser = subparsers.add_parser(
        'test',
        help='Run environment/solver tests',
        description='Test the physics solver and computational environment to ensure everything is working correctly'
    )
    test_parser.add_argument(
        '--full', 
        action='store_true',
        help='Run full test suite with more extensive tests'
    )
    test_parser.add_argument(
        '--benchmark-devices',
        action='store_true',
        default=True,
        help='Benchmark GPU vs CPU precision and performance (default: True)'
    )
    test_parser.add_argument(
        '--no-benchmark-devices',
        dest='benchmark_devices',
        action='store_false',
        help='Skip device benchmarking'
    )
    
    # Validate command - validates individual theories
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate a specific theory',
        description='Run validation tests on a specific gravitational theory'
    )
    validate_parser.add_argument(
        'theory_path',
        help='Path to the theory file to validate (e.g., theories/mytheory/theory.py)'
    )
    validate_parser.add_argument(
        '--steps',
        type=int,
        default=10000,
        help='Number of integration steps for validation (default: 10000)'
    )
    
    # Discover command
    discover_parser = subparsers.add_parser(
        'discover',
        help='Start AI-powered theory discovery',
        description='Use AI to automatically generate and test new gravitational theories'
    )
    discover_parser.add_argument('--initial', type=str, help='Initial prompt for theory generation')
    discover_parser.add_argument('--from', dest='from_theory', type=str, help='Base theory file to improve upon')
    discover_parser.add_argument('--self-monitor', action='store_true', help='Enable self-monitoring mode')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Run model benchmarks',
        description='Test AI models on physics discovery tasks'
    )
    benchmark_parser.add_argument('--model', type=str, required=True, help='Model name to benchmark')
    
    # Submit command
    submit_parser = subparsers.add_parser(
        'submit',
        help='Submit a candidate theory',
        description='Submit a discovered theory for community review'
    )
    submit_parser.add_argument('candidate_dir', help='Candidate directory name (e.g., c_20240723_140530_a7b9c2d1)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command provided, show help
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Handle commands
    if args.command == 'run':
        # Check if --test flag is set
        if hasattr(args, 'test') and args.test:
            print("🔬 Running pre-run environment tests...")
            from physics_agent.run_environment_tests import run_environment_tests
            if not run_environment_tests(steps=100):
                print("\n❌ Environment tests failed! Please fix the issues before running.")
                sys.exit(1)
            print("✅ Pre-run environment tests passed!\n")
        else:
            print("💡 Tip: Use --test flag to run pre-run environment tests to ensure solver correctness\n")
        
        # Run the evaluation
        from physics_agent.evaluation import main as run_evaluation
        
        # Set max steps globally if specified
        if args.max_steps:
            import physics_agent.geodesic_integrator as gi
            gi.DEFAULT_NUM_STEPS = args.max_steps
        
        # Convert namespace to list for evaluation
        sys.argv = ['albert-run']  # Set program name
        
        # Add all the arguments
        for key, value in vars(args).items():
            if key not in ['command'] and value is not None:
                if isinstance(value, bool):
                    if value:
                        sys.argv.append(f'--{key.replace("_", "-")}')
                else:
                    sys.argv.append(f'--{key.replace("_", "-")}')
                    sys.argv.append(str(value))
        
        # Run evaluation and get results
        results, run_dir = run_evaluation()
        print(f"\n📊 Evaluation complete! Results saved to: {run_dir}")
        

    elif args.command == 'setup':
        # Run the setup script
        from albert_setup import main as run_setup
        run_setup()
        
    elif args.command == 'test':
        # Run environment/solver tests
        print("🔬 Running environment/solver tests...\n")
        from physics_agent.run_environment_tests import run_environment_tests
        full_test = args.full if hasattr(args, 'full') else False
        benchmark_devices = args.benchmark_devices if hasattr(args, 'benchmark_devices') else False
        steps = 1000 if full_test else 100
        
        if run_environment_tests(steps=steps, full=full_test, benchmark_devices=benchmark_devices):
            print("\n✅ All environment tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some environment tests failed!")
            sys.exit(1)
            
    elif args.command == 'validate':
        # Validate a specific theory
        print(f"🔍 Validating theory: {args.theory_path}\n")
        
        # Check if theory file exists
        if not os.path.exists(args.theory_path):
            print(f"❌ Error: Theory file not found: {args.theory_path}")
            sys.exit(1)
        
        # Import necessary modules
        import torch
        import importlib.util
        from pathlib import Path
        from physics_agent.theory_engine_core import TheoryEngine
        from physics_agent.base_theory import GravitationalTheory
        
        # Load the theory
        theory_name = Path(args.theory_path).stem
        spec = importlib.util.spec_from_file_location(theory_name, args.theory_path)
        if spec is None or spec.loader is None:
            print(f"❌ Error: Could not load module from {args.theory_path}")
            sys.exit(1)
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[theory_name] = module
        spec.loader.exec_module(module)
        
        # Find the theory class
        theory_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, GravitationalTheory) and 
                obj != GravitationalTheory):
                theory_class = obj
                break
        
        if theory_class is None:
            print(f"❌ Error: No GravitationalTheory subclass found in {args.theory_path}")
            sys.exit(1)
        
        # Instantiate the theory
        try:
            theory = theory_class()
            print(f"✅ Successfully loaded: {theory.name}\n")
        except Exception as e:
            print(f"❌ Error instantiating theory: {str(e)}")
            sys.exit(1)
        
        # Initialize minimal engine for validation
        engine = TheoryEngine(device='cpu', dtype=torch.float64, verbose=False)
        
        # Run basic validation checks
        print("Running validation checks...\n")
        
        # 1. Check metric implementation
        print("1. Checking metric implementation...")
        try:
            # Test at a safe radius (10 Schwarzschild radii)
            from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT
            rs = 2 * GRAVITATIONAL_CONSTANT * SOLAR_MASS / SPEED_OF_LIGHT**2
            r = torch.tensor(10.0 * rs / engine.length_scale, dtype=engine.dtype)
            t = torch.tensor(0.0, dtype=engine.dtype)
            theta = torch.tensor(torch.pi/2, dtype=engine.dtype)
            phi = torch.tensor(0.0, dtype=engine.dtype)
            
            # Most theories use get_metric_components, not metric directly
            if hasattr(theory, 'get_metric_components'):
                components = theory.get_metric_components(r, theta, phi, t)
                if all(comp is not None for comp in components.values()):
                    print("   ✅ Metric components implementation OK")
                else:
                    print("   ❌ Some metric components are None")
            elif hasattr(theory, 'metric'):
                g = theory.metric(r, theta, phi, t)
                if g.shape != (4, 4):
                    print(f"   ❌ Metric has wrong shape: {g.shape} (expected (4, 4))")
                elif torch.isnan(g).any() or torch.isinf(g).any():
                    print("   ❌ Metric contains NaN or Inf values")
                else:
                    print("   ✅ Metric implementation OK")
            else:
                print("   ✅ Theory uses base metric implementation")
        except Exception as e:
            print(f"   ❌ Metric implementation failed: {str(e)}")
        
        # 2. Check if theory is symmetric
        print("\n2. Checking symmetry properties...")
        print(f"   Is symmetric: {theory.is_symmetric}")
        print(f"   Theory category: {getattr(theory, 'category', 'unknown')}")
        
        # 3. Run a short trajectory test
        print("\n3. Running trajectory test...")
        hist = None
        try:
            r0_si = float(r * engine.length_scale)  # Convert to SI and ensure it's a float
            n_steps = min(args.steps, 100)
            dtau_si = 0.1 * engine.time_scale
            
            hist, tag, _ = engine.run_trajectory(
                theory, 
                r0_si,  # Pass as float
                n_steps,  # Number of steps
                dtau_si,  # Time step in SI
                no_cache=True,
                verbose=False
            )
            
            if hist is None:
                print("   ❌ Trajectory computation failed")
            else:
                print(f"   ✅ Trajectory computed successfully ({hist.shape[0]} steps)")
        except Exception as e:
            print(f"   ❌ Trajectory test failed: {str(e)}")
        
        # 4. Check conservation laws if trajectory succeeded
        if hist is not None and len(hist) > 10:
            print("\n4. Checking conservation laws...")
            try:
                # Extract trajectory components
                r_traj = hist[:, 1] / engine.length_scale  # Convert to geometric units
                phi_traj = hist[:, 2]
                
                # Compute angular momentum
                if len(hist) > 1:
                    dphi_dtau = torch.diff(phi_traj) / 0.1
                    dphi_dtau = torch.cat([dphi_dtau[:1], dphi_dtau])
                    L = r_traj**2 * dphi_dtau
                    
                    L_mean = L.mean().abs()
                    if L_mean > 1e-10:
                        L_variation = (L.max() - L.min()) / L_mean
                        if L_variation < 1e-6:
                            print(f"   ✅ Angular momentum conserved (variation: {L_variation:.2e})")
                        else:
                            print(f"   ⚠️  Angular momentum variation: {L_variation:.2e}")
                    else:
                        print("   ℹ️  No angular momentum (radial trajectory)")
            except Exception as e:
                print(f"   ⚠️  Could not check conservation laws: {str(e)}")
        
        # 5. Run selected validators
        print("\n5. Running physics validators...")
        
        # Get validators based on theory category
        category = getattr(theory, 'category', 'classical')
        if category == 'quantum':
            print("   Running quantum validators...")
        else:
            print("   Running classical validators...")
        
        # Import and run basic validators
        try:
            from physics_agent.validations import MercuryPrecessionValidator, LightDeflectionValidator
            
            # Mercury precession
            print("\n   Mercury Precession Test:")
            mercury_validator = MercuryPrecessionValidator(engine)
            mercury_result = mercury_validator.validate(theory, verbose=False)
            if isinstance(mercury_result, dict) and mercury_result.get('passed'):
                print(f"     ✅ PASSED (error: {mercury_result.get('error', 0):.3f} arcsec/century)")
            elif isinstance(mercury_result, dict):
                print(f"     ❌ FAILED (error: {mercury_result.get('error', 0):.3f} arcsec/century)")
            else:
                print(f"     ⚠️  Unexpected result format")
            
            # Light deflection
            print("\n   Light Deflection Test:")
            light_validator = LightDeflectionValidator(engine)
            light_result = light_validator.validate(theory, verbose=False)
            if isinstance(light_result, dict) and light_result.get('passed'):
                print(f"     ✅ PASSED (error: {light_result.get('error', 0):.3f} arcsec)")
            elif isinstance(light_result, dict):
                print(f"     ❌ FAILED (error: {light_result.get('error', 0):.3f} arcsec)")
            else:
                print(f"     ⚠️  Unexpected result format")
                
        except Exception as e:
            print(f"   ⚠️  Could not run all validators: {str(e)}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Validation Summary for {theory.name}")
        print(f"{'='*60}")
        print("\nValidation complete. Check the results above for any issues.")
        print(f"\nFor full validation with all tests, run:")
        print(f"  albert run --theory-filter {theory_name}")
        
    elif args.command == 'discover':
        # Run self-discovery
        from physics_agent.self_discovery.self_discovery import main as run_discovery
        # Build command line arguments
        sys.argv = ['albert-discover', '--self-discover']
        if args.initial:
            sys.argv.extend(['--initial-prompt', args.initial])
        if args.from_theory:
            sys.argv.extend(['--theory', args.from_theory])
        if args.self_monitor:
            sys.argv.append('--self-monitor')
        run_discovery()
        
    elif args.command == 'benchmark':
        # Run benchmark mode
        print(f"Benchmarking model: {args.model}")
        print("This feature will connect to the model API and run physics discovery tests.")
        print("Implementation coming soon...")
        
    elif args.command == 'submit':
        # Submit candidate
        from submit_candidate import main as run_submit
        sys.argv = ['albert-submit', args.candidate_dir]
        run_submit()
        
    else:
        # No command specified
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 