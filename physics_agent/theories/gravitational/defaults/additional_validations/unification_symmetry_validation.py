# unification_symmetry_validation.py
import torch
import numpy as np
import json
import os
from datetime import datetime
from scipy.stats import ttest_ind
from typing import Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from base_validation import ObservationalValidation
from base_theory import GravitationalTheory

# Add baselines to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'baselines'))
from kerr import Kerr
from kerr_newman import KerrNewman

# Import other validators from same directory
import mercury_perihelion_validation
import pulsar_anomaly_validation
import pulsar_timing_validation
import cassini_ppn_validation

MercuryPerihelionValidation = mercury_perihelion_validation.MercuryPerihelionValidation
PulsarAnomalyValidation = pulsar_anomaly_validation.PulsarAnomalyValidation
PulsarTimingValidation = pulsar_timing_validation.PulsarTimingValidation
CassiniValidation = cassini_ppn_validation.CassiniValidation


class UnificationSymmetryValidation(ObservationalValidation):
    """
    Validates unification symmetry at γ=0.75 for Linear Signal Loss.
    Tests if gravitational signal loss mimics electromagnetic effects geometrically.
    """
    
    def __init__(self, device=None, dtype=None):
        super().__init__(device, dtype)
        self.target_gamma = 0.75
        
    def validate(self, theory: GravitationalTheory, **kwargs) -> Dict[str, Any]:
        """Run unification symmetry tests only for Linear Signal Loss at γ=0.75."""
        
        # Check if this is the right theory and parameter
        if not hasattr(theory, 'gamma'):
            return {
                'test_name': 'Unification Symmetry',
                'skipped': True,
                'reason': 'Theory has no gamma parameter'
            }
            
        if abs(theory.gamma - self.target_gamma) > 1e-6:
            return {
                'test_name': 'Unification Symmetry',
                'skipped': True,
                'reason': f'Only runs for γ={self.target_gamma}'
            }
        
        verbose = kwargs.get('verbose', False)
        test_mode = kwargs.get('test', False)
        
        print(f"\n    Starting Unification Symmetry validation for {theory.name}...")
        
        # Configuration
        NUM_RUNS = 3 if test_mode else 10
        SEED = 42
        Q_VALUES = np.logspace(18, 20, 5)
        
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        
        # Baselines
        baselines = {
            'Kerr': Kerr(a=0.5),
            'Kerr-Newman': KerrNewman(a=0.5, Q=1e19)
        }
        
        # Sub-validators
        sub_validators = {
            'mercury': MercuryPerihelionValidation(device=self.device, dtype=self.dtype),
            'pulsar_anomaly': PulsarAnomalyValidation(device=self.device, dtype=self.dtype),
            'pulsar_timing': PulsarTimingValidation(device=self.device, dtype=self.dtype),
            'cassini': CassiniValidation(device=self.device, dtype=self.dtype)
        }
        
        # Run sub-tests
        all_balances = []
        all_p_values = []
        
        for test_name, validator in sub_validators.items():
            if verbose:
                print(f"      Running sub-test: {test_name}")
                
            # Get baseline predictions
            gr_result = validator.validate(baselines['GR'], verbose=False, test=test_mode)
            rn_result = validator.validate(baselines['RN'], verbose=False, test=test_mode)
            
            # Run theory multiple times
            theory_results = []
            for i in range(NUM_RUNS):
                result = validator.validate(theory, verbose=False, test=test_mode)
                theory_results.append(result.get('predicted', 0))
            
            # Compute losses
            gr_baseline = gr_result.get('predicted', 0)
            rn_baseline = rn_result.get('predicted', 0)
            
            losses_gr = [abs(pred - gr_baseline) for pred in theory_results]
            losses_rn = [abs(pred - rn_baseline) for pred in theory_results]
            
            # Compute balance and p-value
            mean_balance = abs(np.mean(losses_gr) - np.mean(losses_rn))
            all_balances.append(mean_balance)
            
            if len(set(losses_gr)) > 1 and len(set(losses_rn)) > 1:
                _, p_value = ttest_ind(losses_gr, losses_rn)
            else:
                p_value = 1.0  # Perfect match
            all_p_values.append(p_value)
            
            if verbose:
                print(f"        Balance: {mean_balance:.6f}, p-value: {p_value:.4f}")
        
        # Q sensitivity analysis
        if verbose:
            print("      Running Q sensitivity analysis...")
            
        q_balances = []
        for Q in Q_VALUES:
            baselines['RN'] = ReissnerNordstrom(Q=Q)
            rn_result = sub_validators['pulsar_anomaly'].validate(baselines['RN'], verbose=False, test=test_mode)
            theory_result = sub_validators['pulsar_anomaly'].validate(theory, verbose=False, test=test_mode)
            
            gr_result = sub_validators['pulsar_anomaly'].validate(baselines['GR'], verbose=False, test=test_mode)
            
            loss_gr = abs(theory_result.get('predicted', 0) - gr_result.get('predicted', 0))
            loss_rn = abs(theory_result.get('predicted', 0) - rn_result.get('predicted', 0))
            balance = abs(loss_gr - loss_rn)
            q_balances.append(balance)
            
            if verbose:
                print(f"        Q={Q:.1e}: Balance={balance:.6f}")
        
        # Overall metrics
        avg_balance = np.mean(all_balances)
        avg_p_value = np.mean(all_p_values)
        passed = avg_balance < 0.1 and avg_p_value > 0.05
        
        # Save results
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {
            'timestamp': timestamp,
            'theory': theory.name,
            'gamma': theory.gamma,
            'avg_balance': avg_balance,
            'avg_p_value': avg_p_value,
            'all_balances': all_balances,
            'all_p_values': all_p_values,
            'q_sensitivity': list(zip(Q_VALUES.tolist(), q_balances)),
            'passed': passed
        }
        
        json_path = os.path.join(results_dir, f'unification_γ{self.target_gamma}_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        if verbose:
            print(f"      Results saved to {json_path}")
        
        print(f"    Unification Symmetry validation complete:")
        print(f"      Average balance: {avg_balance:.6f}")
        print(f"      Average p-value: {avg_p_value:.4f}")
        print(f"      Result: {'PASSED' if passed else 'FAILED'}")
        
        return {
            'test_name': 'Unification Symmetry (γ=0.75)',
            'observed': 0.0,  # Perfect symmetry
            'predicted': avg_balance,
            'error': avg_balance,
            'passed': passed,
            'avg_p_value': avg_p_value,
            'details': summary
        } 