#!/usr/bin/env python3
"""
Update Docs Leaderboard Script

This script updates the leaderboard HTML in docs/latest_leaderboard.html
by parsing the latest run results from physics_agent/runs/.
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Any

def find_latest_run(runs_dir: str = "physics_agent/runs") -> str:
    """Find the most recent run directory."""
    run_dirs = glob.glob(os.path.join(runs_dir, "run_*"))
    if not run_dirs:
        print(f"No run directories found in {runs_dir}")
        return None
    
    # Sort by timestamp in directory name
    run_dirs.sort(key=lambda x: os.path.basename(x).split('_')[1:3])
    return run_dirs[-1]

def parse_validation_results(theory_dir: str) -> Dict[str, Any]:
    """Parse validation results from a theory directory."""
    validation_file = os.path.join(theory_dir, "validation_results.json")
    if not os.path.exists(validation_file):
        return None
    
    with open(validation_file, 'r') as f:
        data = json.load(f)
    
    # Count passed tests by type
    constraint_tests = []
    observational_tests = []
    
    for validation in data.get('validations', []):
        test_type = validation.get('type', 'unknown')
        validator_name = validation.get('validator', 'Unknown')
        passed = validation['flags']['overall'] == 'PASS'
        
        test_info = {
            'name': validator_name,
            'passed': passed,
            'loss': validation.get('loss', None),
            'details': validation.get('details', {})
        }
        
        if test_type == 'constraint':
            constraint_tests.append(test_info)
        elif test_type == 'observational':
            observational_tests.append(test_info)
    
    return {
        'theory_name': data.get('theory_name', 'Unknown'),
        'category': data.get('category', 'unknown'),
        'passed': data.get('passed', False),
        'constraint_tests': constraint_tests,
        'observational_tests': observational_tests,
        'constraints_passed': sum(1 for t in constraint_tests if t['passed']),
        'constraints_total': len(constraint_tests),
        'observational_passed': sum(1 for t in observational_tests if t['passed']),
        'observational_total': len(observational_tests)
    }

def collect_run_results(run_dir: str) -> List[Dict[str, Any]]:
    """Collect all theory results from a run directory."""
    results = []
    
    # Get all theory directories
    for entry in os.listdir(run_dir):
        theory_dir = os.path.join(run_dir, entry)
        
        # Skip non-directories and special directories
        if not os.path.isdir(theory_dir) or entry in ['fail', 'predictions']:
            continue
        
        # Parse validation results
        theory_results = parse_validation_results(theory_dir)
        if theory_results:
            theory_results['directory'] = theory_dir
            theory_results['is_baseline'] = entry.startswith('baseline_')
            results.append(theory_results)
    
    # Sort by quantum theories first, then by name
    results.sort(key=lambda x: (x['category'] != 'quantum', x['theory_name']))
    
    return results

def generate_test_card_html(title: str, tests: List[Dict[str, Any]]) -> str:
    """Generate HTML for a test card."""
    html = f'<div class="test-card">\n'
    html += f'    <h3>{title}</h3>\n'
    
    for test in tests:
        status_class = 'pass' if test['passed'] else 'fail'
        status_text = 'PASS' if test['passed'] else 'FAIL'
        
        # Add error percentage if available
        details = test.get('details', {})
        error_percent = details.get('error_percent', None)
        if error_percent is not None and test['passed']:
            status_text += f' ({error_percent:.1f}%)'
        
        html += f'    <div class="test-result">\n'
        html += f'        <span class="test-name">{test["name"]}</span>\n'
        html += f'        <span class="{status_class}">{status_text}</span>\n'
        html += f'    </div>\n'
    
    html += '</div>\n'
    return html

def generate_leaderboard_html(run_dir: str, results: List[Dict[str, Any]]) -> str:
    """Generate the complete leaderboard HTML."""
    run_name = os.path.basename(run_dir)
    timestamp = run_name.split('_')[1:3]
    run_date = f"{timestamp[0][:4]}/{timestamp[0][4:6]}/{timestamp[0][6:8]} {timestamp[1][:2]}:{timestamp[1][2:4]}:{timestamp[1][4:6]}"
    
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Albert Framework - Latest Leaderboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #1976d2;
            margin-bottom: 10px;
        }
        
        .run-info {
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }
        
        .leaderboard {
            margin-bottom: 40px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        th {
            background-color: #1976d2;
            color: white;
            font-weight: 600;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        .rank-1 {
            background-color: #fff3cd;
        }
        
        .rank-1:hover {
            background-color: #ffeaa7;
        }
        
        .category-quantum {
            color: #4caf50;
            font-weight: 600;
        }
        
        .category-classical {
            color: #ff9800;
            font-weight: 600;
        }
        
        .pass {
            color: #4caf50;
            font-weight: 600;
        }
        
        .fail {
            color: #f44336;
            font-weight: 600;
        }
        
        .detail-section {
            margin-top: 40px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        
        .detail-section h2 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .test-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .test-card {
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .test-card h3 {
            color: #1976d2;
            font-size: 16px;
            margin-bottom: 10px;
        }
        
        .test-result {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .test-name {
            font-weight: 500;
        }
        
        .no-data {
            color: #666;
            font-style: italic;
        }
        
        .update-info {
            margin-top: 40px;
            padding: 10px;
            background-color: #e3f2fd;
            border-radius: 4px;
            font-size: 12px;
            color: #666;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Albert Framework - Latest Leaderboard</h1>
        <div class="run-info">
'''
    
    html += f'            Run: {run_name} | Date: {run_date} | Total Theories: {len(results)}\n'
    html += '        </div>\n\n'
    
    # Leaderboard table
    html += '''        <div class="leaderboard">
            <h2>Theory Rankings</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Theory</th>
                        <th>Category</th>
                        <th>Validation Status</th>
                        <th>Constraint Tests</th>
                        <th>Observational Tests</th>
                        <th>Overall Result</th>
                    </tr>
                </thead>
                <tbody>
'''
    
    # Add theory rows
    rank = 1
    for i, theory in enumerate(results):
        if theory['is_baseline']:
            rank_text = '-'
            rank_class = ''
        else:
            rank_text = str(rank)
            rank_class = f'rank-{rank}' if rank <= 3 else ''
            rank += 1
        
        category_class = f'category-{theory["category"]}'
        baseline_text = ' (baseline)' if theory['is_baseline'] else ''
        
        validation_status = 'VALIDATED' if theory['passed'] else 'FAILED'
        validation_class = 'pass' if theory['passed'] else 'fail'
        
        constraint_text = f"{theory['constraints_passed']}/{theory['constraints_total']} PASSED"
        constraint_class = 'pass' if theory['constraints_passed'] == theory['constraints_total'] else 'fail'
        
        obs_text = f"{theory['observational_passed']}/{theory['observational_total']} PASSED"
        obs_class = 'pass' if theory['observational_passed'] > theory['observational_total'] * 0.5 else 'fail'
        
        overall_text = 'PASSED' if theory['passed'] else 'FAILED'
        overall_class = 'pass' if theory['passed'] else 'fail'
        
        html += f'                    <tr class="{rank_class}">\n'
        html += f'                        <td>{rank_text}</td>\n'
        html += f'                        <td>{theory["theory_name"]}</td>\n'
        html += f'                        <td class="{category_class}">{theory["category"]}{baseline_text}</td>\n'
        html += f'                        <td class="{validation_class}">{validation_status}</td>\n'
        html += f'                        <td class="{constraint_class}">{constraint_text}</td>\n'
        html += f'                        <td class="{obs_class}">{obs_text}</td>\n'
        html += f'                        <td class="{overall_class}">{overall_text}</td>\n'
        html += f'                    </tr>\n'
    
    html += '''                </tbody>
            </table>
        </div>
'''
    
    # Detailed test results for each theory
    for theory in results:
        if theory['is_baseline']:
            continue
            
        html += f'''
        <div class="detail-section">
            <h2>{theory["theory_name"]} - Detailed Test Results</h2>
            
            <div class="test-grid">
'''
        
        # Constraint tests
        if theory['constraint_tests']:
            html += generate_test_card_html("Constraint Tests", theory['constraint_tests'])
        
        # Observational tests - split into quantum and classical
        quantum_tests = []
        classical_tests = []
        
        for test in theory['observational_tests']:
            if any(keyword in test['name'].lower() for keyword in ['cow', 'atom', 'quantum', 'clock', 'decoherence']):
                quantum_tests.append(test)
            else:
                classical_tests.append(test)
        
        if quantum_tests:
            html += generate_test_card_html("Quantum Observational Tests", quantum_tests)
        
        if classical_tests:
            html += generate_test_card_html("Classical Observational Tests", classical_tests)
        
        html += '''            </div>
        </div>
'''
    
    # Update info
    html += f'''
        <div class="update-info">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Generated from: {run_dir}
        </div>
    </div>
</body>
</html>'''
    
    return html

def main():
    """Main function to update the leaderboard."""
    # Find latest run
    latest_run = find_latest_run()
    if not latest_run:
        print("No run directories found!")
        return
    
    print(f"Found latest run: {latest_run}")
    
    # Collect results
    results = collect_run_results(latest_run)
    print(f"Found {len(results)} theories in run")
    
    # Generate HTML
    html = generate_leaderboard_html(latest_run, results)
    
    # Save to docs
    output_path = "docs/latest_leaderboard.html"
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Leaderboard updated: {output_path}")
    print(f"View at: documentation.html?tab=leaderboard")

if __name__ == "__main__":
    main() 