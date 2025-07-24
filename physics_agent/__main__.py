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
  albert run                    # Run all theories in the theories folder
  albert run --theory-filter ugm  # Run only UGM theories
  albert run --steps 1000       # Run with custom step count
  albert setup                  # Configure Albert instance
  albert discover               # Start self-discovery mode
  albert benchmark              # Run model benchmarks

For more information on each command, use: albert <command> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command - runs theories
    run_parser = subparsers.add_parser(
        'run', 
        help='Run gravitational theory simulations',
        description='Run all theories in the theories folder with comprehensive validation'
    )
    
    # Import CLI arguments from cli.py and add them to run_parser
    from physics_agent.cli import get_cli_parser
    cli_parser = get_cli_parser()
    
    # Copy all arguments from the original parser to our run subcommand
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
            # Add other attributes for store actions
            if hasattr(action, 'nargs') and action.nargs is not None:
                kwargs['nargs'] = action.nargs
            if hasattr(action, 'choices') and action.choices is not None:
                kwargs['choices'] = action.choices
            if hasattr(action, 'metavar') and action.metavar is not None:
                kwargs['metavar'] = action.metavar
        elif action.__class__.__name__ in ['StoreTrueAction', '_StoreTrueAction']:
            kwargs['action'] = 'store_true'
            # store_true doesn't accept nargs, type, etc.
        elif action.__class__.__name__ in ['StoreFalseAction', '_StoreFalseAction']:
            kwargs['action'] = 'store_false'
            # store_false doesn't accept nargs, type, etc.
        else:
            # For other action types, just use store with appropriate settings
            kwargs['action'] = 'store'
            if hasattr(action, 'type') and action.type is not None:
                kwargs['type'] = action.type
            if hasattr(action, 'nargs') and action.nargs is not None:
                kwargs['nargs'] = action.nargs
            if hasattr(action, 'choices') and action.choices is not None:
                kwargs['choices'] = action.choices
            
        run_parser.add_argument(*action.option_strings, **kwargs)
    
    # Setup command
    setup_parser = subparsers.add_parser(
        'setup',
        help='Configure your Albert instance',
        description='Set up API keys, cryptographic identity, and network participation'
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
        # Run the theory engine
        from physics_agent.theory_engine_core import main as run_theories
        from physics_agent.cli import get_cli_parser
        
        # Convert namespace to list for theory_engine_core
        # Get the default values from the parser
        cli_parser = get_cli_parser()
        defaults = {}
        for action in cli_parser._actions:
            if action.dest != 'help':
                defaults[action.dest] = action.default
        
        sys.argv = ['albert-run']  # Set program name
        # Add all the arguments back, but only if they were explicitly set
        for key, value in vars(args).items():
            if key != 'command' and value is not None:
                # Skip if it's a default value (not explicitly set by user)
                if key in defaults and value == defaults[key]:
                    continue
                    
                if isinstance(value, bool):
                    if value:
                        sys.argv.append(f'--{key.replace("_", "-")}')
                else:
                    sys.argv.append(f'--{key.replace("_", "-")}')
                    sys.argv.append(str(value))
        run_theories()
        
    elif args.command == 'setup':
        # Run the setup script
        from albert_setup import main as run_setup
        run_setup()
        
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