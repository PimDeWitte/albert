import sys
import trace
import os
import json
from datetime import datetime
from collections import defaultdict

# <reason>chain: Create comprehensive tracer for function usage analysis</reason>
class UsageTracer:
    def __init__(self):
        self.called_functions = set()
        self.called_classes = set()
        self.function_calls = defaultdict(int)
        self.file_accesses = set()
        
    def trace_calls(self, frame, event, arg):
        if event == 'call':
            code = frame.f_code
            filename = code.co_filename
            
            # <reason>chain: Only track our project files, not standard library</reason>
            if '/albert/' in filename and 'site-packages' not in filename:
                func_name = code.co_name
                class_name = None
                
                # <reason>chain: Extract class name if this is a method</reason>
                if 'self' in frame.f_locals:
                    class_name = frame.f_locals['self'].__class__.__name__
                    self.called_classes.add(f"{filename}::{class_name}")
                
                func_id = f"{filename}::{class_name}.{func_name}" if class_name else f"{filename}::{func_name}"
                self.called_functions.add(func_id)
                self.function_calls[func_id] += 1
                self.file_accesses.add(filename)
                
        return self.trace_calls

def run_trajectory_with_trace():
    # <reason>chain: Import and setup engine for trajectory run</reason>
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    tracer = UsageTracer()
    sys.settrace(tracer.trace_calls)
    
    try:
        # <reason>chain: Run a comprehensive trajectory that exercises most functionality</reason>
        from physics_agent.theory_engine_core import TheoryEngine, process_and_evaluate_theory
        from physics_agent.theory_loader import TheoryLoader
        from physics_agent.base_theory import GravitationalTheory, QuantumMixin
        from physics_agent.constants import (
            GRAVITATIONAL_CONSTANT, SPEED_OF_LIGHT, SOLAR_MASS,
            EARTH_MASS, PLANCK_CONSTANT
        )
        import torch
        import argparse
        
        # <reason>chain: Create args to simulate full run</reason>
        args = argparse.Namespace(
            device='cpu',
            dtype='float64',
            verbose=True,
            no_cache=True,
            disable_viz=False,
            n_steps=100,
            dtau=1e-5,
            r0=20.0,
            baseline_only=False,
            theory_only=False,
            load_results=None,
            cache_only=False,
            disable_ai=True,
            sweep_alpha=False,
            sweep_r0=False,
            profile=False,
            quantum_tests=True,
            run_predictions=True,
            trajectory_cache_dir='trajectory_cache'
        )
        
        # <reason>chain: Initialize engine and run trajectory</reason>
        engine = TheoryEngine(device='cpu', dtype=torch.float64, verbose=True)
        
        # <reason>chain: Test with a quantum theory to exercise quantum code paths</reason>
        # Import a theory directly for testing
        from physics_agent.theories.quantum_corrected.theory import QuantumCorrected
        model = QuantumCorrected(
            alpha=0.01  # Alpha parameter
        )
        
        # <reason>chain: Run full trajectory with all features</reason>
        r0 = torch.tensor(args.r0, dtype=torch.float64)
        dtau = torch.tensor(args.dtau, dtype=torch.float64)
        
        # <reason>chain: Run trajectory to trace basic functions</reason>
        hist, error_msg, quantum_jumps = engine.run_trajectory(
            model, float(r0), args.n_steps, float(dtau),
            mass=1.0 * SOLAR_MASS
        )
        
        # <reason>chain: Also run multi-particle to trace more code paths</reason>
        results = engine.run_multi_particle_trajectories(
            model, float(r0), args.n_steps, float(dtau),
            theory_category='quantum'
        )
        
        # <reason>chain: Run validations to trace validation functions</reason>
        if hist is not None:
            validation_results = engine.run_all_validations(model, hist, None)
        
        # <reason>chain: Test unification assessment</reason>
        unification = engine.assess_unification_potential(model)
        
    finally:
        sys.settrace(None)
    
    return tracer

def analyze_all_functions():
    # <reason>chain: Find all defined functions and classes in the codebase</reason>
    import ast
    
    all_functions = set()
    all_classes = set()
    
    for root, dirs, files in os.walk('.'):
        # <reason>chain: Skip directories we're told to ignore</reason>
        if any(skip in root for skip in ['solver_tests/', 'theories/', 'self_discovery/', 'docs/', '.git', '__pycache__', 'runs/']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        tree = ast.parse(f.read())
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            all_functions.add(f"{os.path.abspath(filepath)}::{node.name}")
                        elif isinstance(node, ast.ClassDef):
                            all_classes.add(f"{os.path.abspath(filepath)}::{node.name}")
                            # <reason>chain: Also track methods within classes</reason>
                            for item in node.body:
                                if isinstance(item, ast.FunctionDef):
                                    all_functions.add(f"{os.path.abspath(filepath)}::{node.name}.{item.name}")
                except:
                    pass
    
    return all_functions, all_classes

def find_unused_files():
    # <reason>chain: Find PNG, PY, and MD files that might be unused</reason>
    unused_candidates = {
        'png': [],
        'py': [],
        'md': []
    }
    
    # <reason>chain: Get all files first</reason>
    for root, dirs, files in os.walk('.'):
        # Skip protected directories
        if any(skip in root for skip in ['solver_tests/', 'theories/', 'self_discovery/', 'docs/', '.git', '__pycache__', 'runs/']):
            continue
            
        for file in files:
            filepath = os.path.join(root, file)
            
            if file.endswith('.png'):
                unused_candidates['png'].append(filepath)
            elif file.endswith('.md') and file != 'README.md':
                unused_candidates['md'].append(filepath)
            elif file.endswith('.py') and file not in ['__init__.py', 'setup.py', 'albert_setup.py']:
                # <reason>chain: Check if this py file is imported anywhere</reason>
                module_name = file[:-3]
                is_imported = False
                
                # Check for imports
                for r2, d2, f2 in os.walk('.'):
                    if any(skip in r2 for skip in ['.git', '__pycache__', 'runs/']):
                        continue
                    for f in f2:
                        if f.endswith('.py'):
                            try:
                                with open(os.path.join(r2, f), 'r') as fp:
                                    content = fp.read()
                                    if module_name in content and 'import' in content:
                                        is_imported = True
                                        break
                            except:
                                pass
                    if is_imported:
                        break
                        
                if not is_imported:
                    unused_candidates['py'].append(filepath)
    
    return unused_candidates

def main():
    print("=" * 80)
    print("PHYSICS AGENT CODE USAGE ANALYSIS")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print()
    
    # <reason>chain: Run trajectory with tracing</reason>
    print("1. Running trajectory with function tracing...")
    tracer = run_trajectory_with_trace()
    
    # <reason>chain: Analyze all functions in codebase</reason>
    print("\n2. Analyzing all functions and classes in codebase...")
    all_functions, all_classes = analyze_all_functions()
    
    # <reason>chain: Find unused files</reason>
    print("\n3. Finding potentially unused files...")
    unused_files = find_unused_files()
    
    # <reason>chain: Generate report</reason>
    print("\n4. Generating usage report...")
    
    # <reason>chain: Normalize paths for comparison</reason>
    called_funcs_normalized = set()
    for func in tracer.called_functions:
        called_funcs_normalized.add(os.path.abspath(func.split('::')[0]) + '::' + func.split('::')[1])
    
    # <reason>chain: Find unused functions</reason>
    unused_functions = all_functions - called_funcs_normalized
    
    # <reason>chain: Save detailed report</reason>
    report = {
        'timestamp': str(datetime.now()),
        'summary': {
            'total_functions': len(all_functions),
            'total_classes': len(all_classes),
            'called_functions': len(tracer.called_functions),
            'called_classes': len(tracer.called_classes),
            'unused_functions': len(unused_functions),
            'accessed_files': len(tracer.file_accesses)
        },
        'called_functions': sorted(list(tracer.called_functions)),
        'called_classes': sorted(list(tracer.called_classes)),
        'function_call_counts': dict(sorted(tracer.function_calls.items(), key=lambda x: x[1], reverse=True)[:50]),
        'unused_functions': sorted(list(unused_functions)),
        'unused_files': unused_files,
        'accessed_files': sorted(list(tracer.file_accesses))
    }
    
    with open('usage_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # <reason>chain: Print summary</reason>
    print("\n" + "=" * 80)
    print("USAGE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total functions defined: {len(all_functions)}")
    print(f"Total classes defined: {len(all_classes)}")
    print(f"Functions called during trajectory: {len(tracer.called_functions)}")
    print(f"Classes used during trajectory: {len(tracer.called_classes)}")
    print(f"Potentially unused functions: {len(unused_functions)}")
    print(f"Files accessed: {len(tracer.file_accesses)}")
    print()
    print(f"Potentially unused PNG files: {len(unused_files['png'])}")
    print(f"Potentially unused PY files: {len(unused_files['py'])}")
    print(f"Potentially unused MD files: {len(unused_files['md'])}")
    print()
    print("Full report saved to: usage_analysis_report.json")
    print()
    print("Top 10 most called functions:")
    for func, count in list(sorted(tracer.function_calls.items(), key=lambda x: x[1], reverse=True))[:10]:
        print(f"  {count:5d} calls: {func}")

if __name__ == '__main__':
    main() 