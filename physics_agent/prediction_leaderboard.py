#!/usr/bin/env python3
"""
Prediction Leaderboard Generator

Parses prediction results from latest run and generates a Markdown table leaderboard.
"""

import os
import json
import glob

import numpy as np

def get_latest_run_dir(base_dir='runs'):
    """Find the latest run directory."""
    run_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, 'run_*')) if os.path.isdir(d)])
    if not run_dirs:
        raise ValueError("No run directories found")
    return run_dirs[-1]

def load_prediction_results(predictions_dir):
    """Load all prediction JSONs."""
    results = {}
    for json_file in glob.glob(os.path.join(predictions_dir, '*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)
            theory_name = data['theory']
            results[theory_name] = data['predictions']
    return results

def load_trajectory_losses(run_dir):
    """Load trajectory losses from theory directories."""
    losses = {}
    for theory_dir in glob.glob(os.path.join(run_dir, '*')):
        if not os.path.isdir(theory_dir) or 'fail' in theory_dir:
            continue
        theory_name = os.path.basename(theory_dir).replace('_', ' ')
        scores_path = os.path.join(theory_dir, 'comprehensive_scores.json')
        if os.path.exists(scores_path):
            with open(scores_path, 'r') as f:
                scores = json.load(f)
                traj_loss = scores.get('trajectory', {}).get('average_loss', np.nan)
                losses[theory_name] = traj_loss
    return losses

def compute_overall_score(predictions):
    """Compute overall score as number of SOTA beats."""
    return sum(1 for p in predictions if p.get('beats_sota', False))

def generate_table(results, losses):
    """Generate Markdown table."""
    # Get all validators
    validators = set()
    for preds in results.values():
        for p in preds:
            validators.add(p['validator'])
    validators = sorted(validators)
    
    # Prepare data
    table_data = []
    for theory, preds in results.items():
        row = {
            'Theory': theory,
            'Trajectory Loss': losses.get(theory, np.nan)
        }
        total_beats = 0
        for v in validators:
            pred = next((p for p in preds if p['validator'] == v), None)
            if pred:
                beats = '✅' if pred['beats_sota'] else '❌'
                
                # Calculate improvement factor if possible
                improvement_str = ""
                
                # Try to get improvement from various sources
                improvement_value = None
                if 'improvement' in pred and pred['improvement'] is not None:
                    improvement_value = pred['improvement']
                elif 'prediction_data' in pred:
                    # Look for delta values in prediction_data
                    if 'delta_chi2' in pred['prediction_data']:
                        improvement_value = pred['prediction_data']['delta_chi2']
                    elif 'delta_log_likelihood' in pred['prediction_data']:
                        improvement_value = pred['prediction_data']['delta_log_likelihood']
                
                # Handle special cases for specific validators
                if v == 'CMB Power Spectrum Prediction Validator':
                    # For CMB, lower chi-squared is better
                    if 'predicted_value' in pred and pred['predicted_value'] is not None:
                        improvement_str = f"χ²={pred['predicted_value']:.1f}"
                        if improvement_value is not None:
                            if pred['beats_sota']:
                                improvement_str += f" (Δ=+{abs(improvement_value):.1f})"
                            else:
                                improvement_str += f" (Δ={improvement_value:.1f})"
                    else:
                        improvement_str = "N/A"
                elif v == 'PTA Stochastic GW Background Validator':
                    # For PTA, show log-likelihood
                    if 'predicted_value' in pred and pred['predicted_value'] is not None:
                        improvement_str = f"lnL={pred['predicted_value']:.2f}"
                        if improvement_value is not None:
                            if pred['beats_sota']:
                                improvement_str += f" (Δ=+{abs(improvement_value):.2f})"
                            else:
                                improvement_str += f" (Δ={improvement_value:.2f})"
                    else:
                        improvement_str = "N/A"
                # elif v == 'Future Detectors Validator':  # Removed - not implemented
                #     # For Future Detectors, show SNR
                #     if 'predicted_value' in pred and pred['predicted_value'] is not None:
                #         improvement_str = f"SNR={pred['predicted_value']:.0f}"
                #         if pred['beats_sota']:
                #             improvement_str += " ✓"
                #     else:
                #         improvement_str = "N/A"
                # elif v == 'Novel Signatures Validator':  # Removed - not implemented
                #     # For Novel Signatures, show Bayes Factor
                #     if 'predicted_value' in pred and pred['predicted_value'] is not None:
                #         improvement_str = f"BF={pred['predicted_value']:.1f}"
                #         if pred['beats_sota'] and pred['predicted_value'] > 10:
                #             improvement_str += " (>10 threshold)"
                #     else:
                #         improvement_str = "N/A"
                elif v == 'Primordial GWs Validator':
                    # For Primordial GWs, show r value
                    if 'predicted_value' in pred and pred['predicted_value'] is not None:
                        improvement_str = f"r={pred['predicted_value']:.3f}"
                        if 'observed_value' in pred and pred['observed_value'] is not None:
                            improvement_str += f" (<{pred['observed_value']:.3f})"
                    else:
                        improvement_str = "N/A"
                else:
                    # Generic handling for other validators
                    if 'predicted_value' in pred and pred['predicted_value'] is not None:
                        value = pred['predicted_value']
                        if abs(value) < 0.01 or abs(value) > 1000:
                            improvement_str = f"{value:.2e}"
                        else:
                            improvement_str = f"{value:.2f}"
                    else:
                        improvement_str = "N/A"
                
                row[v] = f"{beats} {improvement_str}"
                if pred['beats_sota']:
                    total_beats += 1
            else:
                row[v] = 'N/A'
        row['Total SOTA Beats'] = total_beats
        table_data.append(row)
    
    # Sort by total beats descending
    table_data.sort(key=lambda x: x['Total SOTA Beats'], reverse=True)
    
    # Generate Markdown
    md = "# Prediction Leaderboard\n\n"
    md += "| Rank | Theory | Trajectory Loss | " + ' | '.join(validators) + " | Total SOTA Beats |\n"
    md += "|------|--------|-----------------" + '|---' * len(validators) + "|------------------|\n"
    
    for i, row in enumerate(table_data, 1):
        md += f"| {i} | {row['Theory']} | {row['Trajectory Loss']:.4e} | "
        for v in validators:
            md += f"{row.get(v, 'N/A')} | "
        md += f"{row['Total SOTA Beats']} |\n"
    
    # Add legend
    md += "\n## Legend\n\n"
    md += "- ✅ = Beats state-of-the-art (SOTA)\n"
    md += "- ❌ = Does not beat SOTA\n"
    md += "- **CMB**: χ² value (lower is better), Δ shows improvement over ΛCDM\n"
    md += "- **PTA**: lnL = log-likelihood, Δ shows improvement over SMBHB model\n"
    md += "- **Future Detectors**: SNR = signal-to-noise ratio for LISA/DECIGO\n"
    md += "- **Novel Signatures**: BF = Bayes Factor vs General Relativity\n"
    md += "- **Primordial GWs**: r = tensor-to-scalar ratio\n"
    md += "- **Trajectory Loss**: Lower values indicate better match to baseline theories\n"
    
    return md

if __name__ == '__main__':
    latest_run = get_latest_run_dir()
    print(f"Latest run: {latest_run}")
    
    predictions_dir = os.path.join(latest_run, 'predictions')
    if not os.path.exists(predictions_dir):
        print("No predictions directory found")
        exit(1)
    
    results = load_prediction_results(predictions_dir)
    losses = load_trajectory_losses(latest_run)
    
    table = generate_table(results, losses)
    print(table)
    
    # Save to file
    output_file = os.path.join(latest_run, 'prediction_leaderboard.md')
    with open(output_file, 'w') as f:
        f.write(table)
    print(f"\nLeaderboard saved to: {output_file}") 