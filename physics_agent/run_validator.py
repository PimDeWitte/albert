#!/usr/bin/env python3
"""
Main script to validate promising gravitational theories against observational data.
Reads from promising_candidates.log files and runs all validation tests.
"""

import argparse
import os
import sys
import importlib.util
import inspect
from typing import List, Dict, Type

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_agent.theory_engine_core import TheoryEngine
from physics_agent.validations import (
    PromisingCandidateValidator,
    MercuryPrecessionValidator,
    LightDeflectionValidator,
    ObservationalValidator
)


def load_additional_validators(theory_dir: str, engine: TheoryEngine) -> Dict[str, Type[ObservationalValidator]]:
    """
    Load additional validators from a theory's additional_validations directory.
    
    Args:
        theory_dir: Path to theory directory
        engine: TheoryEngine instance
        
    Returns:
        Dict mapping validator names to validator classes
    """
    additional_validators = {}
    
    # Look for additional_validations directory
    val_dir = os.path.join(theory_dir, 'additional_validations')
    if not os.path.exists(val_dir):
        return additional_validators
        
    print(f"\nLoading additional validators from {val_dir}...")
    
    # Find all .py files in the directory
    for filename in os.listdir(val_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            filepath = os.path.join(val_dir, filename)
            
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(f"additional_{filename[:-3]}", filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find all ObservationalValidator subclasses
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, ObservationalValidator) and 
                        obj != ObservationalValidator):
                        # Use lowercase name as key
                        key = name.lower().replace('validator', '')
                        additional_validators[key] = obj
                        print(f"  Found additional validator: {name}")
                        
            except Exception as e:
                print(f"  Warning: Failed to load {filename}: {str(e)}")
                
    return additional_validators


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate promising gravitational theories against observational data"
    )
    parser.add_argument(
        'log_files',
        nargs='+',
        help='Path(s) to promising_candidates.log files'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory to save validation results (default: same as log file)'
    )
    parser.add_argument(
        '--device',
        default=None,
        help='PyTorch device (cpu, cuda, mps)'
    )
    parser.add_argument(
        '--dtype',
        default='float32',
        choices=['float32', 'float64'],
        help='PyTorch dtype'
    )
    parser.add_argument(
        '--validators',
        nargs='+',
        default=['all'],
        choices=['all', 'mercury', 'light', 'shapiro', 'frame_dragging', 'ppn', 'cmb', 'pta'],
        help='Which validators to run'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()


def get_validators(validator_names: List[str], engine: TheoryEngine, theory_dir: str = None) -> List:
    """
    Get list of validators based on names.
    
    Args:
        validator_names: List of validator names or ['all']
        engine: TheoryEngine instance
        theory_dir: Optional theory directory to load additional validators from
        
    Returns:
        List of validator instances
    """
    available_validators = {
        'mercury': MercuryPrecessionValidator,
        'light': LightDeflectionValidator,
        # Add more validators as they're implemented:
        # 'shapiro': ShapiroDelayValidator,
        # 'frame_dragging': FrameDraggingValidator,
        # 'ppn': PPNParameterValidator,
    }
    
    # Import prediction validators if available
    try:
        from physics_agent.validations import CMBPowerSpectrumValidator, PTAStochasticGWValidator
        available_validators.update({
            'cmb': CMBPowerSpectrumValidator,
            'pta': PTAStochasticGWValidator,
        })
    except ImportError:
        pass
    
    # Load additional validators from theory directory if provided
    if theory_dir:
        additional_validators = load_additional_validators(theory_dir, engine)
        available_validators.update(additional_validators)
    
    if 'all' in validator_names:
        validator_names = list(available_validators.keys())
    
    validators = []
    for name in validator_names:
        if name in available_validators:
            validator_class = available_validators[name]
            validators.append(validator_class(engine))
            print(f"  Loaded validator: {validator_class.__name__}")
        else:
            print(f"  Warning: Unknown validator '{name}'")
            
    return validators


def main():
    args = parse_args()
    
    # Set up dtype
    import torch
    if args.dtype == 'float64':
        dtype = torch.float64
    else:
        dtype = torch.float32
    
    # Initialize theory engine
    print("Initializing theory engine...")
    engine = TheoryEngine(device=args.device, dtype=dtype)
    print(f"  Device: {engine.device}")
    print(f"  Dtype: {engine.dtype}")
    
    # Get validators
    print("\nLoading validators...")
    validators = get_validators(args.validators, engine)
    
    if not validators:
        print("Error: No validators loaded!")
        return 1
    
    # Create main validator
    candidate_validator = PromisingCandidateValidator(validators, engine)
    
    # Process each log file
    for log_file in args.log_files:
        print(f"\n{'='*60}")
        print(f"Processing: {log_file}")
        print('='*60)
        
        if not os.path.exists(log_file):
            print(f"Error: Log file not found: {log_file}")
            continue
            
        # Run validations
        try:
            results = candidate_validator.validate_all_candidates(
                log_file,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"Error processing {log_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nValidation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 