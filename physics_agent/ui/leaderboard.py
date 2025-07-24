"""
Leaderboard generation from run results
"""
import os
import json
import glob
from typing import List, Dict, Any

class Leaderboard:
    """Generates and manages the theory leaderboard"""
    
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
    
    def get_latest_run(self) -> str:
        """Find the most recent run directory"""
        run_dirs = glob.glob(os.path.join(self.runs_dir, "run_*"))
        if not run_dirs:
            return None
        
        # Sort by timestamp in directory name
        run_dirs.sort(key=lambda x: os.path.basename(x).split('_')[1:3])
        return run_dirs[-1]
    
    def collect_theory_results(self, run_dir: str) -> List[Dict[str, Any]]:
        """Collect results from all theories in a run"""
        results = []
        
        # Get all theory directories (excluding fail and baseline dirs)
        theory_dirs = glob.glob(os.path.join(run_dir, "*"))
        
        for theory_dir in theory_dirs:
            if not os.path.isdir(theory_dir):
                continue
                
            dir_name = os.path.basename(theory_dir)
            
            # Skip special directories
            if dir_name in ['fail', 'run_config.json'] or dir_name.startswith('baseline_'):
                continue
            
            # Load theory info
            theory_info_path = os.path.join(theory_dir, 'theory_info.json')
            if not os.path.exists(theory_info_path):
                continue
                
            with open(theory_info_path, 'r') as f:
                theory_info = json.load(f)
            
            # Load losses
            losses_path = os.path.join(theory_dir, 'losses.json')
            losses = {}
            if os.path.exists(losses_path):
                with open(losses_path, 'r') as f:
                    losses = json.load(f)
            
            # Load validation results
            validation_path = os.path.join(theory_dir, 'final_validation.json')
            validation_passed = False
            if os.path.exists(validation_path):
                with open(validation_path, 'r') as f:
                    validation_data = json.load(f)
                    validation_passed = validation_data.get('constraints_passed', False)
            
            # Check for quantum validation (for unified theories)
            quantum_validation_path = os.path.join(theory_dir, 'quantum_validation.json')
            quantum_passed = None
            if os.path.exists(quantum_validation_path):
                with open(quantum_validation_path, 'r') as f:
                    quantum_data = json.load(f)
                    quantum_validators = [v for v in quantum_data['validations'] 
                                          if any(q in v['validator'] for q in ['Interferometry', 'Decoherence', 'Clock', 'Lagrangian'])]
                    quantum_passed_count = sum(1 for v in quantum_validators if v['flags']['overall'] == 'PASS')
                    quantum_total = len(quantum_validators)
                    quantum_passed = quantum_passed_count >= 4 and quantum_total >= 5
            
            # Check for visualization
            viz_path = os.path.join(theory_dir, 'viz', 'trajectory_comparison.png')
            has_viz = os.path.exists(viz_path)
            
            # Calculate minimum loss across all baselines and loss types
            min_loss = float('inf')
            best_loss_type = None
            best_baseline = None
            
            for loss_type, baseline_losses in losses.items():
                for baseline, loss_value in baseline_losses.items():
                    if loss_value < min_loss:
                        min_loss = loss_value
                        best_loss_type = loss_type
                        best_baseline = baseline
            
            # Build result entry
            result = {
                'name': theory_info['name'],
                'category': theory_info.get('category', 'unknown'),
                'is_symmetric': theory_info.get('is_symmetric', False),
                'min_loss': min_loss if min_loss != float('inf') else None,
                'best_loss_type': best_loss_type,
                'best_baseline': best_baseline,
                'all_losses': losses,
                'validation_passed': validation_passed,
                'quantum_passed': quantum_passed,
                'has_visualization': has_viz,
                'directory': theory_dir,
                'trajectory_length': theory_info.get('trajectory_length', 0)
            }
            
            results.append(result)
        
        return results
    
    def rank_theories(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank theories by performance"""
        
        # Separate theories by status
        valid_theories = []
        failed_theories = []
        
        for theory in results:
            # Theory must pass validation and have a loss value
            if theory['validation_passed'] and theory['min_loss'] is not None:
                valid_theories.append(theory)
            else:
                failed_theories.append(theory)
        
        # Sort valid theories by minimum loss (lower is better)
        valid_theories.sort(key=lambda x: x['min_loss'])
        
        # Add ranking
        for i, theory in enumerate(valid_theories):
            theory['rank'] = i + 1
            theory['status'] = 'valid'
        
        # Failed theories don't get a rank
        for theory in failed_theories:
            theory['rank'] = None
            theory['status'] = 'failed'
        
        # Combine all theories
        all_theories = valid_theories + failed_theories
        
        return all_theories
    
    def generate_leaderboard(self) -> Dict[str, Any]:
        """Generate the complete leaderboard"""
        
        latest_run = self.get_latest_run()
        if not latest_run:
            return {
                'error': 'No runs found',
                'theories': [],
                'run_info': None
            }
        
        # Load run config
        run_config_path = os.path.join(latest_run, 'run_config.json')
        run_info = {}
        if os.path.exists(run_config_path):
            with open(run_config_path, 'r') as f:
                run_info = json.load(f)
        
        # Collect and rank theories
        results = self.collect_theory_results(latest_run)
        ranked_theories = self.rank_theories(results)
        
        # Build leaderboard
        leaderboard = {
            'run_directory': latest_run,
            'run_timestamp': run_info.get('timestamp', 'unknown'),
            'run_config': run_info,
            'total_theories': len(ranked_theories),
            'valid_theories': sum(1 for t in ranked_theories if t['status'] == 'valid'),
            'failed_theories': sum(1 for t in ranked_theories if t['status'] == 'failed'),
            'theories': ranked_theories
        }
        
        return leaderboard
    
    def get_theory_details(self, theory_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific theory"""
        
        latest_run = self.get_latest_run()
        if not latest_run:
            return None
        
        results = self.collect_theory_results(latest_run)
        
        for theory in results:
            if theory['name'] == theory_name:
                # Add more detailed information
                theory_dir = theory['directory']
                
                # Load trajectory if available
                trajectory_path = os.path.join(theory_dir, 'trajectory.pt')
                theory['has_trajectory'] = os.path.exists(trajectory_path)
                
                # Get all available visualizations
                viz_dir = os.path.join(theory_dir, 'viz')
                if os.path.exists(viz_dir):
                    theory['visualizations'] = [
                        f for f in os.listdir(viz_dir) 
                        if f.endswith(('.png', '.jpg', '.jpeg'))
                    ]
                else:
                    theory['visualizations'] = []
                
                return theory
        
        return None 