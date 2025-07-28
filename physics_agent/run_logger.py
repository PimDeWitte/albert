"""
Logging and reporting utilities for physics agent validation runs.
"""

import os
import sys
import datetime
import json
from physics_agent.theory_utils import extract_theory_name_from_dir


def generate_comprehensive_summary(main_run_dir: str, validation_results: dict, calibration_certificate: dict = None) -> dict:
    """
    Generate a comprehensive summary of all theories tested, including pass/fail status,
    failure reasons, errors thrown, and scores.
    
    Args:
        main_run_dir: The main run directory containing all results
        validation_results: Dictionary of validation results from Phase 1
        calibration_certificate: Optional calibration certificate for the run
        
    Returns:
        Dictionary containing the complete summary
    """
    print(f"\n{'='*60}")
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    # Display calibration certificate if available
    if calibration_certificate:
        print(f"\n📋 CALIBRATION CERTIFICATE")
        print(f"   ID: {calibration_certificate.get('certificate_id', 'N/A')}")
        print(f"   Status: {calibration_certificate.get('status', 'N/A')}")
        print(f"   Health Score: {calibration_certificate.get('health_score', 0):.0f}%")
        
        guarantees = calibration_certificate.get('guarantees', {})
        if guarantees:
            print("   Guarantees:")
            for guarantee_type, guarantee in guarantees.items():
                status = guarantee.get('status', 'UNKNOWN')
                symbol = '✓' if status == 'GUARANTEED' else '✗' if status == 'NOT_GUARANTEED' else '○'
                print(f"     {symbol} {guarantee_type.replace('_', ' ').title()}: {status}")
    else:
        print(f"\n⚠️  NO CALIBRATION CERTIFICATE - Run integrity not guaranteed")
    
    summary = {
        'run_directory': main_run_dir,
        'timestamp': os.path.basename(main_run_dir),
        'calibration_certificate': calibration_certificate,
        'theories_tested': {},
        'passed': [],
        'failed': [],
        'errors': [],
        'statistics': {
            'total_tested': 0,
            'total_passed': 0,
            'total_failed': 0,
            'failure_reasons': {}
        }
    }
    
    # 1. Process validation results from Phase 1
    for theory_name, val_result in validation_results.items():
        theory_info = {
            'name': theory_name,
            'phase_1_passed': val_result.get('passed', False),
            'validations': val_result.get('validations', []),
            'errors': [],
            'failure_reasons': [],
            'scores': {}
        }
        
        # Extract validation details
        for validation in val_result.get('validations', []):
            if validation['flags']['overall'] == 'ERROR':
                theory_info['errors'].append({
                    'validator': validation['validator'],
                    'error': validation['flags'].get('details', 'Unknown error')
                })
            elif validation['flags']['overall'] == 'FAIL':
                theory_info['failure_reasons'].append({
                    'validator': validation['validator'],
                    'reason': validation['flags'].get('details', 'Failed validation')
                })
        
        summary['theories_tested'][theory_name] = theory_info
    
    # 2. Check the main run directory for successful theories
    for entry in os.listdir(main_run_dir):
        theory_dir = os.path.join(main_run_dir, entry)
        if not os.path.isdir(theory_dir) or entry in ['fail', 'predictions'] or entry.startswith('baseline_'):
            continue
            
        # Extract theory name from directory - handle both underscore and special character cases
        # Examples: "Participatory_QG_ω_0_00" -> "Participatory QG (ω=0.00)"
        #          "Stochastic_Loss_Conserved_γ_0_50_σ_1_00e-04_a_0_00_Q_0_00" -> "Stochastic Loss Conserved (γ=0.50, σ=1.00e-04, a=0.00, Q=0.00)"
        theory_name = extract_theory_name_from_dir(entry)
        
        # Look for comprehensive scores
        scores_path = os.path.join(theory_dir, 'comprehensive_scores.json')
        if os.path.exists(scores_path):
            with open(scores_path, 'r') as f:
                scores = json.load(f)
                if theory_name in summary['theories_tested']:
                    summary['theories_tested'][theory_name]['scores'] = scores.get('comprehensive_score', {})
                    summary['theories_tested'][theory_name]['final_status'] = 'PASSED'
                    summary['passed'].append(theory_name)
                else:
                    # Theory wasn't in initial validation results
                    summary['theories_tested'][theory_name] = {
                        'name': theory_name,
                        'phase_1_passed': True,
                        'final_status': 'PASSED',
                        'scores': scores.get('comprehensive_score', {}),
                        'validations': [],
                        'errors': [],
                        'failure_reasons': []
                    }
                    summary['passed'].append(theory_name)
    
    # 3. Check the fail directory for failed theories
    fail_dir = os.path.join(main_run_dir, 'fail')
    if os.path.exists(fail_dir):
        for entry in os.listdir(fail_dir):
            failed_theory_dir = os.path.join(fail_dir, entry)
            if not os.path.isdir(failed_theory_dir):
                continue
                
            # Extract theory name and failure reason from directory name
            # Check for both hyphen and underscore versions for backward compatibility
            failure_suffixes = ['trajectory-failed', 'quantum-validation-failed', 
                                'constraint-validation-failed', 'non-quantum',
                                'preflight-validation-failed', 'preflight-simulation-failed',
                                # Also check underscore versions
                                'trajectory_failed', 'quantum_validation_failed',
                                'constraint_validation_failed', 'non_quantum']
            
            failure_reason = 'unknown'
            theory_name = entry
            
            # Try to find a matching suffix
            for suffix in failure_suffixes:
                if entry.endswith(f'_{suffix}'):
                    theory_name = entry[:-len(f'_{suffix}')]
                    failure_reason = suffix.replace('-', ' ').replace('_', ' ')
                    break
            
            if failure_reason == 'unknown':
                theory_name = extract_theory_name_from_dir(entry)
                
                # Try to infer failure reason from files in the directory
                failure_info_path = os.path.join(failed_theory_dir, 'failure_info.json')
                quantum_val_path = os.path.join(failed_theory_dir, 'quantum_validation.json')
                final_val_path = os.path.join(failed_theory_dir, 'final_validation.json')
                trajectory_path = os.path.join(failed_theory_dir, 'trajectory.pt')
                
                if os.path.exists(failure_info_path):
                    # Check failure_info.json for reason
                    try:
                        with open(failure_info_path, 'r') as f:
                            failure_info = json.load(f)
                            reason = failure_info.get('reason', '')
                            if 'trajectory' in reason.lower():
                                failure_reason = 'trajectory failed'
                            elif 'pre-flight' in reason.lower() or 'preflight' in reason.lower():
                                failure_reason = 'preflight simulation failed'
                    except:
                        pass
                
                # If still unknown, check which files exist
                if failure_reason == 'unknown':
                    if os.path.exists(quantum_val_path) and not os.path.exists(trajectory_path):
                        failure_reason = 'quantum validation failed'
                    elif os.path.exists(final_val_path) and not os.path.exists(trajectory_path):
                        failure_reason = 'constraint validation failed'
                    elif not os.path.exists(trajectory_path) and not os.path.exists(final_val_path):
                        failure_reason = 'validation failed'
            
            # Look for specific validation results
            quantum_val_path = os.path.join(failed_theory_dir, 'quantum_validation.json')
            final_val_path = os.path.join(failed_theory_dir, 'final_validation.json')
            
            if theory_name in summary['theories_tested']:
                summary['theories_tested'][theory_name]['final_status'] = 'FAILED'
                summary['theories_tested'][theory_name]['failure_category'] = failure_reason
            else:
                summary['theories_tested'][theory_name] = {
                    'name': theory_name,
                    'phase_1_passed': False,
                    'final_status': 'FAILED',
                    'failure_category': failure_reason,
                    'validations': [],
                    'errors': [],
                    'failure_reasons': [],
                    'scores': {}
                }
            
            # Extract quantum validation details if available
            if os.path.exists(quantum_val_path):
                with open(quantum_val_path, 'r') as f:
                    quantum_data = json.load(f)
                    for validation in quantum_data.get('validations', []):
                        if validation['flags']['overall'] == 'FAIL':
                            # Extract more detailed information from validation
                            details = validation.get('details', {})
                            validator_name = validation.get('validator', 'Unknown')
                            
                            # Format reason based on validator type
                            if 'COW' in validator_name:
                                reason = f"Predicted: {details.get('predicted', 'N/A')} radians vs Observed: {details.get('observed', 'N/A')} radians"
                            elif 'Light' in validator_name and 'deflection' in validator_name.lower():
                                reason = f"Predicted: {details.get('predicted', 'N/A')} arcsec vs Observed: {details.get('observed', 'N/A')} arcsec"
                            elif 'χ²' in str(details.get('total_chi2', '')):
                                reason = f"χ² = {details.get('total_chi2', 'N/A')} (reduced χ² = {details.get('reduced_chi2', 'N/A')})"
                            else:
                                reason = f"Predicted: {details.get('predicted', 'N/A')} vs Observed: {details.get('observed', 'N/A')}"
                            
                            summary['theories_tested'][theory_name]['failure_reasons'].append({
                                'validator': validator_name,
                                'reason': reason
                            })
            
            summary['failed'].append(theory_name)
            
            # Track failure reasons
            if failure_reason not in summary['statistics']['failure_reasons']:
                summary['statistics']['failure_reasons'][failure_reason] = 0
            summary['statistics']['failure_reasons'][failure_reason] += 1
    
    # 4. Update statistics
    summary['statistics']['total_tested'] = len(summary['theories_tested'])
    summary['statistics']['total_passed'] = len(summary['passed'])
    summary['statistics']['total_failed'] = len(summary['failed'])
    
    # 5. Print the summary
    print(f"\nTOTAL THEORIES TESTED: {summary['statistics']['total_tested']}")
    print(f"PASSED: {summary['statistics']['total_passed']}")
    print(f"FAILED: {summary['statistics']['total_failed']}")
    
    if summary['passed']:
        print(f"\n{'='*60}")
        print("THEORIES THAT PASSED ALL VALIDATIONS:")
        print(f"{'='*60}")
        for theory in sorted(summary['passed']):
            theory_data = summary['theories_tested'][theory]
            if 'final_score' in theory_data.get('scores', {}):
                print(f"  ✓ {theory} (Score: {theory_data['scores']['final_score']:.4f})")
            else:
                print(f"  ✓ {theory}")
    
    if summary['failed']:
        print(f"\n{'='*60}")
        print("THEORIES THAT FAILED:")
        print(f"{'='*60}")
        for theory in sorted(summary['failed']):
            theory_data = summary['theories_tested'][theory]
            print(f"\n  ✗ {theory}")
            
            # Print failure category
            if 'failure_category' in theory_data:
                print(f"    Failure Category: {theory_data['failure_category']}")
            
            # Print specific failed tests with details
            if 'failure_reasons' in theory_data and theory_data['failure_reasons']:
                print(f"    Failed Tests:")
                for failure in theory_data['failure_reasons']:
                    validator = failure['validator']
                    reason = failure['reason']
                    # Try to extract parameters if available
                    params = ''
                    if 'details' in failure and isinstance(failure['details'], dict):
                        param_list = [f"{k}={v}" for k, v in failure['details'].items() if k != 'error' and k != 'reason']
                        if param_list:
                            params = f" ({', '.join(param_list)})"
                    print(f"      - {validator}{params}: {reason}")
            
            # Print errors
            if 'errors' in theory_data and theory_data['errors']:
                print(f"    Errors:")
                for error in theory_data['errors']:
                    print(f"      - {error['validator']}: {error['error']}")
    
    # Print failure reason breakdown
    if summary['statistics']['failure_reasons']:
        print(f"\n{'='*60}")
        print("FAILURE REASON BREAKDOWN:")
        print(f"{'='*60}")
        for reason, count in sorted(summary['statistics']['failure_reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} theories")
    
    # <reason>chain: Import to_serializable to handle tensor conversions</reason>
    from physics_agent.functions import to_serializable
    
    # Save summary to file
    summary_path = os.path.join(main_run_dir, 'comprehensive_summary.json')
    with open(summary_path, 'w') as f:
        # <reason>chain: Convert summary to JSON-serializable format before saving</reason>
        summary_serializable = to_serializable(summary)
        json.dump(summary_serializable, f, indent=4)
    print(f"\n📊 Detailed summary saved to: {summary_path}")
    
    return summary


 


class RunLogger:
    """Captures stdout to a log file for a run"""
    
    def __init__(self, run_dir):
        self.run_dir = run_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(run_dir, f"run_log_{timestamp}.txt")
        self.original_stdout = None
        self.tee = None
        
    def start(self):
        """Start capturing stdout to log file"""
        self.original_stdout = sys.stdout
        
        class Tee:
            def __init__(self, *files):
                self.files = files
                
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
                    
            def flush(self):
                for f in self.files:
                    f.flush()
        
        # Create log file
        log_file = open(self.log_path, 'w')
        
        # Tee stdout to both console and file
        self.tee = Tee(sys.stdout, log_file)
        sys.stdout = self.tee
        
    def stop(self):
        """Stop capturing stdout"""
        if self.original_stdout:
            sys.stdout = self.original_stdout
            # Close the log file
            if hasattr(self.tee, 'files'):
                for f in self.tee.files:
                    if f != self.original_stdout:
                        f.close()
                        
    def get_log_path(self):
        """Get the path to the log file"""
        return self.log_path 