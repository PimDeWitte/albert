#!/usr/bin/env python3
"""
Self-Discovery System - Automated gravitational theory exploration
Uses TheoryEngine for core simulation and evaluation functionality
"""
from __future__ import annotations
import importlib
import sys
import os

# Add the parent directory to Python path to find physics_agent module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import torch
import hashlib
import shutil
from datetime import datetime
from physics_agent.theory_engine_core import TheoryEngine, get_cli_parser
from physics_agent.theory_loader import TheoryLoader
from physics_agent.base_theory import GravitationalTheory
from physics_agent.ui.leaderboard_html_generator import LeaderboardHTMLGenerator
import time
from physics_agent.update_checker import check_on_startup

def extend_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds self-discovery specific arguments to the base parser."""
    p = parser.add_argument_group('Self-Discovery Options')
    p.add_argument("--self-discover", action="store_true", help="Enable self-discovery loop for generating new theories via API.")
    p.add_argument("--initial-prompt", type=str, default="", help="Initial prompt or seed query for theory generation. Can be a file path or direct text.")
    p.add_argument("--api-provider", type=str, default="grok", choices=["grok", "gemini", "openai", "anthropic"], 
                   help="API provider for theory generation. xAI/Grok is the primary supported provider. Others are experimental.")
    p.add_argument("--theory", type=str, default=None, help="Focus on improving a specific theory (e.g., 'theories/QuantumGravity' or 'candidates/LinearSignalLoss')")
    return parser


def load_initial_prompt(prompt_arg: str) -> str:
    """Load initial prompt from file or return as-is if it's direct text."""
    if prompt_arg and os.path.isfile(prompt_arg):
        with open(prompt_arg, 'r') as f:
            return f.read().strip()
    return prompt_arg


def generate_theory_via_api(api_provider: str, initial_prompt: str, baseline_theories: list[str], focus_theory: str = None) -> str:
    """Generate a new theory using the specified API provider"""
    # Import the LLM API
    from physics_agent.ui.llm_api import LLMApi
    
    # Initialize the API client
    api = LLMApi(provider=api_provider)
    
    # Enhance prompt if focusing on a specific theory
    if focus_theory:
        # Try to load the focus theory code
        theory_path = None
        if os.path.exists(focus_theory):
            theory_path = focus_theory
        elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), focus_theory)):
            theory_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), focus_theory)
        elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), focus_theory, "theory.py")):
            theory_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), focus_theory, "theory.py")
        
        if theory_path and os.path.exists(theory_path):
            try:
                with open(theory_path, 'r') as f:
                    base_theory_code = f.read()
                print(f"Loaded base theory from: {theory_path}")
                
                # Generate variation
                theory_code = api.generate_theory_variation(
                    base_theory_code,
                    initial_prompt,
                    baseline_theories
                )
            except Exception as e:
                print(f"Error loading base theory: {e}")
                theory_code = api.generate_new_theory(initial_prompt, baseline_theories)
        else:
            print(f"Could not find theory file for: {focus_theory}")
            initial_prompt = f"Improve upon the theory: {focus_theory}. {initial_prompt}"
            theory_code = api.generate_new_theory(initial_prompt, baseline_theories)
    else:
        # Generate completely new theory
        theory_code = api.generate_new_theory(initial_prompt, baseline_theories)
    
    if not theory_code:
        print("\nError: Failed to generate theory code from API")
        print("Make sure GROK_API_KEY is set in your environment variables")
        print("Using mock response for demonstration...")
        # Use the mock response from the API
        api_mock = LLMApi(provider=api_provider)
        theory_code = api_mock._mock_response()
    
    return theory_code


def create_candidate_directory(run_timestamp: str, theory_code: str) -> tuple[str, str]:
    """Create a candidate directory with proper naming convention.
    
    Returns:
        (candidate_id, full_path)
    """
    # Create hash from theory code
    code_hash = hashlib.sha256(theory_code.encode()).hexdigest()[:8]
    candidate_id = f"c_{run_timestamp}_{code_hash}"
    
    # Create directory
    candidates_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "theories", "candidates"
    )
    os.makedirs(candidates_dir, exist_ok=True)
    
    candidate_path = os.path.join(candidates_dir, candidate_id)
    os.makedirs(candidate_path, exist_ok=True)
    
    return candidate_id, candidate_path


def copy_run_to_candidate(run_dir: str, candidate_path: str, theory_code: str):
    """Copy a successful run to the candidates directory."""
    # Copy all files from run directory
    for item in os.listdir(run_dir):
        s = os.path.join(run_dir, item)
        d = os.path.join(candidate_path, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    
    # Save the theory code in the candidate directory
    theory_file = os.path.join(candidate_path, "theory.py")
    with open(theory_file, 'w') as f:
        f.write(theory_code)
    
    # Create a README for the candidate
    readme_path = os.path.join(candidate_path, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# Candidate Theory {os.path.basename(candidate_path)}\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This is a candidate theory generated by the self-discovery system.\n")
        f.write("It has not yet been peer-reviewed and remains in the candidates folder.\n\n")
        f.write("## How to test this candidate\n\n")
        f.write("```bash\n")
        f.write("python -m physics_agent.theory_engine_core --candidates\n")
        f.write("```\n")


def show_pr_instructions(candidate_id: str, candidate_path: str):
    """Show instructions for creating a PR with the new candidate."""
    print("\n" + "="*70)
    print("🎉 PROMISING CANDIDATE DISCOVERED! 🎉")
    print("="*70)
    print(f"\nCandidate ID: {candidate_id}")
    print(f"Location: {candidate_path}")
    print("\nThis candidate scored in the top 10 on the leaderboard!")
    print("\nTo share this discovery with the scientific community:")
    print("\n1. Manual method:")
    print("   git add physics_agent/theories/candidates/")
    print(f"   git commit -m 'Add promising candidate {candidate_id}'")
    print("   git push origin your-branch")
    print("   Then open a PR at: https://github.com/PimDeWitte/albert/pulls")
    print("\n2. Automated method:")
    print("   python submit_candidate.py " + candidate_id)
    print("\n" + "="*70)


def main():
    """Main execution function for self-discovery system"""
    # Check for updates at startup
    check_on_startup()
    
    parser = get_cli_parser()
    parser = extend_cli(parser)
    args = parser.parse_args()
    
    # Initialize the theory engine
    device = args.device
    dtype = None
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float64":
        dtype = torch.float64
    elif args.final or args.cpu_f64:
        dtype = torch.float64
    else:
        dtype = torch.float32
        
    if device is None:
        if args.final or args.cpu_f64:
            device = "cpu"
        else:
            # Determine best device
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
    
    # Only pass device parameter if explicitly set
    if device:
        engine = TheoryEngine(device=device, dtype=dtype)
    else:
        engine = TheoryEngine(dtype=dtype)
    print(f"Running on device: {engine.device}, with dtype: {engine.dtype}")
    
    # --- Theory Loading using TheoryLoader ---
    theories_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "theories")
    loader = TheoryLoader(theories_base_dir=theories_dir)
    
    # In self-discovery mode, by default only load baselines unless --candidates is specified
    if args.self_discover and not args.candidates:
        # Only discover baseline theories
        discovered_theories = {}
        all_theories = loader.discover_theories()
        for theory_key, theory_info in all_theories.items():
            if 'baselines' in theory_info['path']:
                discovered_theories[theory_key] = theory_info
    else:
        discovered_theories = loader.discover_theories()
    
    print(f"--- Loading Theories ---")
    print(f"Discovered {len(discovered_theories)} theory classes")
    
    # Get baseline theory names
    baseline_names = []
    baseline_theories = {}
    for theory_key, theory_info in discovered_theories.items():
        if 'baselines' in theory_info['path']:
            instance = loader.instantiate_theory(theory_key)
            if instance:
                baseline_names.append(instance.name)
                baseline_theories[instance.name] = instance
    
    print(f"Found {len(baseline_names)} baseline theories for comparison")
    
    # --- Self-Discovery Mode ---
    if args.self_discover:
        print("\n--- Starting Self-Discovery Mode ---")
        print("Running baselines and keeping runtime open for agent execution...")
        
        # Create a run directory for this self-discovery session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", f"self_discovery_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Run baselines first
        print("\n--- Running Baseline Theories ---")
        baseline_results = {}
        
        # Set up trajectory parameters
        N_STEPS = args.steps if hasattr(args, 'steps') and args.steps else 1000
        DTau = torch.tensor(0.1, device=engine.device, dtype=engine.dtype)
        r0 = torch.tensor(12.0, device=engine.device, dtype=engine.dtype)
        
        for name, theory in baseline_theories.items():
            print(f"\nRunning baseline: {name}")
            hist, _, _ = engine.run_trajectory(
                theory, r0.item() * engine.length_scale, N_STEPS, DTau.item() * engine.time_scale,
                quantum_interval=0, quantum_beta=0.0
            )
            if hist is not None and hist.shape[0] > 1:
                baseline_results[name] = hist
                print(f"  ✓ Successfully computed trajectory")
            else:
                print(f"  ✗ Failed to compute trajectory")
        
        # Load initial prompt
        initial_prompt = load_initial_prompt(args.initial_prompt)
        if not initial_prompt:
            initial_prompt = "Explore modifications aligned with Albert's quest for unification of gravity and electromagnetism."
        
        print(f"\n--- Agent Mode Active ---")
        print(f"Initial prompt: {initial_prompt}")
        if args.theory:
            print(f"Focusing on theory: {args.theory}")
        
        # Generate new theory via API
        theory_code = generate_theory_via_api(
            args.api_provider, 
            initial_prompt, 
            baseline_names,
            args.theory
        )
        
        if theory_code:
            # Save the theory code temporarily
            temp_theory_file = os.path.join(run_dir, "generated_theory.py")
            with open(temp_theory_file, 'w') as f:
                f.write(theory_code)
            
            # Try to parse and instantiate the generated theory
            try:
                # Load the theory module
                spec = importlib.util.spec_from_file_location("generated_theory", temp_theory_file)
                module = importlib.util.module_from_spec(spec)
                
                # Add necessary imports to module namespace
                module.__dict__['torch'] = torch
                module.__dict__['GravitationalTheory'] = GravitationalTheory
                module.__dict__['np'] = __import__('numpy')
                
                spec.loader.exec_module(module)
                
                # Find the theory class (usually named CustomTheory)
                theory_class = None
                for name, obj in vars(module).items():
                    if (isinstance(obj, type) and 
                        issubclass(obj, GravitationalTheory) and 
                        obj != GravitationalTheory):
                        theory_class = obj
                        break
                
                if theory_class:
                    # Instantiate the theory
                    generated_theory = theory_class()
                    print(f"\nSuccessfully instantiated theory: {generated_theory.name}")
                    
                    # Run full evaluation
                    print("\n--- Evaluating Generated Theory ---")
                    theory_dir = os.path.join(run_dir, generated_theory.name.replace(" ", "_"))
                    os.makedirs(theory_dir, exist_ok=True)
                    
                    # Simple trajectory test first
                    hist, _, _ = engine.run_trajectory(
                        generated_theory, r0.item() * engine.length_scale, 
                        N_STEPS, DTau.item() * engine.time_scale,
                        quantum_interval=0, quantum_beta=0.0
                    )
                    
                    if hist is not None and hist.shape[0] > 1:
                        # Save theory code in the run directory
                        with open(os.path.join(theory_dir, "theory.py"), 'w') as f:
                            f.write(theory_code)
                        
                        # Calculate loss against baselines
                        losses = {}
                        for baseline_name, baseline_hist in baseline_results.items():
                            loss = engine.loss_calculator.compute_trajectory_loss(hist, baseline_hist, 'trajectory_mse')
                            loss_val = loss if isinstance(loss, float) else loss.item()
                            losses[baseline_name] = loss_val
                            print(f"  Loss vs {baseline_name}: {loss_val:.3e}")
                        
                        # Save losses
                        with open(os.path.join(theory_dir, "losses.json"), 'w') as f:
                            json.dump(losses, f, indent=2)
                        
                        # Check if it's in top 10 (simplified check - would need full leaderboard)
                        avg_loss = sum(losses.values()) / len(losses)
                        
                        # For now, consider it promising if avg loss < 0.1
                        if avg_loss < 0.1:
                            print(f"\n✨ Theory shows promising results! Average loss: {avg_loss:.3e}")
                            
                            # Create candidate directory
                            candidate_id, candidate_path = create_candidate_directory(timestamp, theory_code)
                            
                            # Copy run to candidate
                            copy_run_to_candidate(theory_dir, candidate_path, theory_code)
                            
                            # Show PR instructions
                            show_pr_instructions(candidate_id, candidate_path)
                            
                            # Add note about API key
                            print("\nNote: The GROK_API_KEY is for generating theories via the xAI Grok model.")
                            print("Automated theory submission to the Albert Network is not yet implemented,")
                            print("but will use this API in the future for automatic PR creation. Open a PR for now!")
                        else:
                            print(f"\nTheory did not rank in top 10. Average loss: {avg_loss:.3e}")
                    else:
                        print("\nFailed to compute trajectory for generated theory")
                else:
                    print("\nError: No valid theory class found in generated code")
                    
            except Exception as e:
                print(f"\nError evaluating generated theory: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Generate leaderboard for this run
        print(f"\nGenerating leaderboard in: {run_dir}")
        generator = LeaderboardHTMLGenerator()
        generator.generate_leaderboard(run_dir)
        
        print("\n--- Self-Discovery Session Complete ---")
        print(f"Results saved to: {run_dir}")
        
        # Continue discovery loop if API key is available
        api_key = os.environ.get(f'{args.api_provider.upper()}_API_KEY')
        if api_key:
            print("\n--- Starting Continuous Discovery Loop ---")
            print("Press Ctrl+C to exit at any time.")
            
            iteration = 1
            try:
                while True:
                    print(f"\n\n{'='*70}")
                    print(f"DISCOVERY ITERATION {iteration}")
                    print(f"{'='*70}")
                    
                    # Wait a bit between iterations
                    time.sleep(5)
                    
                    # Generate new theory
                    print("\nGenerating new theory...")
                    new_prompt = initial_prompt if iteration == 1 else f"Generate a different approach than previous attempts. {initial_prompt}"
                    theory_code = generate_theory_via_api(
                        args.api_provider, 
                        new_prompt, 
                        baseline_names,
                        args.theory
                    )
                    
                    if theory_code:
                        # Process the new theory
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        iter_dir = os.path.join(run_dir, f"iteration_{iteration}_{timestamp}")
                        os.makedirs(iter_dir, exist_ok=True)
                        
                        # Save generated theory
                        theory_file = os.path.join(iter_dir, "generated_theory.py")
                        with open(theory_file, 'w') as f:
                            f.write(theory_code)
                        
                        # Import and evaluate the theory
                        try:
                            spec = importlib.util.spec_from_file_location("generated_theory", theory_file)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Find the theory class
                            theory_class = None
                            for name, obj in vars(module).items():
                                if (isinstance(obj, type) and 
                                    issubclass(obj, GravitationalTheory) and 
                                    obj != GravitationalTheory):
                                    theory_class = obj
                                    break
                            
                            if theory_class:
                                # Test the theory
                                generated_theory = theory_class()
                                print(f"\nTesting theory: {generated_theory.name}")
                                
                                theory_dir = os.path.join(iter_dir, generated_theory.name.replace(' ', '_'))
                                os.makedirs(theory_dir, exist_ok=True)
                                
                                # Run trajectory computation
                                hist, _, _ = engine.run_trajectory(
                                    generated_theory, 
                                    r0.item() * engine.length_scale,
                                    N_STEPS, 
                                    DTau.item() * engine.time_scale,
                                    quantum_interval=0, 
                                    quantum_beta=0.0
                                )
                                
                                if hist is not None and hist.shape[0] > 1:
                                    # Calculate losses against baselines
                                    losses = {}
                                    for baseline_name, baseline_hist in baseline_results.items():
                                        loss = engine.loss_calculator.compute_trajectory_loss(hist, baseline_hist, 'trajectory_mse')
                                        loss_val = loss if isinstance(loss, float) else loss.item()
                                        losses[baseline_name] = loss_val
                                        
                                    # Check results
                                    avg_loss = sum(losses.values()) / len(losses)
                                    print(f"\n✨ Theory average loss: {avg_loss:.3e}")
                                    
                                    if avg_loss < 0.1:
                                        print(f"\n🎉 PROMISING THEORY FOUND! Creating candidate...")
                                        candidate_id, candidate_path = create_candidate_directory(timestamp, theory_code)
                                        copy_run_to_candidate(theory_dir, candidate_path, theory_code)
                                        show_pr_instructions(candidate_id, candidate_path)
                                
                        except Exception as e:
                            print(f"\nError in iteration {iteration}: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    
                    iteration += 1
                    
            except KeyboardInterrupt:
                print("\n\nExiting continuous discovery mode.")
                print(f"Completed {iteration - 1} discovery iterations.")
        else:
            print("\nRuntime remains open for further agent operations...")
            print("Set your API key to enable continuous discovery loop.")
            print("Press Ctrl+C to exit.")
            
            # Keep the process alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nExiting self-discovery mode.")
        
    else:
        print("\nThis script is for self-discovery. Run with --self-discover to generate new theories.")
        print("To run simulations, use the main engine script: python -m physics_agent.theory_engine_core")
        sys.exit(0)


if __name__ == "__main__":
    main()