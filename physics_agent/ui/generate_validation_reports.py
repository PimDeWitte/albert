#!/usr/bin/env python3
"""
Generate validation report pages for the leaderboard.

This script creates:
1. solver_tests.html - Latest solver test results
2. validator_performance.html - Validator performance metrics
"""

import os
import json
from datetime import datetime
from pathlib import Path


def generate_solver_tests_page(docs_dir: str):
    """Generate the solver tests page from latest report."""
    # Find the latest solver test report
    solver_reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'solver_tests', 'reports'
    )
    
    latest_report = os.path.join(solver_reports_dir, 'solver_test_report_latest.html')
    
    if not os.path.exists(latest_report):
        # Generate a placeholder page
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solver Tests - Albert Framework</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Solver Tests</h1>
            <nav>
                <a href="index.html">← Back to Leaderboard</a>
            </nav>
        </header>
        
        <main>
            <div class="notice">
                <h2>No Solver Test Results Available</h2>
                <p>Run <code>python physics_agent/solver_tests/test_geodesic_validator_comparison.py</code> to generate test results.</p>
            </div>
        </main>
    </div>
</body>
</html>"""
    else:
        # Read the latest report and wrap it with navigation
        with open(latest_report, 'r') as f:
            content = f.read()
        
        # Extract the body content
        body_start = content.find('<div class="container">')
        body_end = content.find('</div>\n</body>')
        
        if body_start != -1 and body_end != -1:
            body_content = content[body_start:body_end+6]
        else:
            body_content = '<div class="container"><p>Error loading report content.</p></div>'
        
        # Create wrapped page
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solver Tests - Albert Framework</title>
    <link rel="stylesheet" href="style.css">
    <style>
        /* Import styles from the report */
        {content[content.find('<style>'):content.find('</style>')+8] if '<style>' in content else ''}
        
        /* Additional navigation styles */
        .nav-header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        .nav-header a {{
            color: #3498db;
            text-decoration: none;
        }}
        .nav-header a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="nav-header">
        <h1>🔬 Solver Tests - Albert Framework</h1>
        <nav>
            <a href="index.html">← Back to Leaderboard</a> | 
            <a href="validator_performance.html">Validator Performance →</a>
        </nav>
    </div>
    {body_content}
</body>
</html>"""
    
    # Save the page
    output_path = os.path.join(docs_dir, 'solver_tests.html')
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Generated: {output_path}")


def generate_validator_performance_page(docs_dir: str):
    """Generate the validator performance page from latest report."""
    # Find the latest performance report
    perf_reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'cache', 'validator_performance', 'reports'
    )
    
    latest_report = os.path.join(perf_reports_dir, 'validator_performance_latest.html')
    
    if not os.path.exists(latest_report):
        # Generate a placeholder page
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validator Performance - Albert Framework</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Validator Performance</h1>
            <nav>
                <a href="index.html">← Back to Leaderboard</a>
            </nav>
        </header>
        
        <main>
            <div class="notice">
                <h2>No Performance Data Available</h2>
                <p>Performance metrics will be collected as theories are validated.</p>
            </div>
        </main>
    </div>
</body>
</html>"""
    else:
        # Read the latest report and wrap it with navigation
        with open(latest_report, 'r') as f:
            content = f.read()
        
        # Extract the body content
        body_start = content.find('<div class="container">')
        body_end = content.find('</div>\n</body>')
        
        if body_start != -1 and body_end != -1:
            body_content = content[body_start:body_end+6]
        else:
            body_content = '<div class="container"><p>Error loading report content.</p></div>'
        
        # Create wrapped page
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validator Performance - Albert Framework</title>
    <link rel="stylesheet" href="style.css">
    <style>
        /* Import styles from the report */
        {content[content.find('<style>'):content.find('</style>')+8] if '<style>' in content else ''}
        
        /* Additional navigation styles */
        .nav-header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        .nav-header a {{
            color: #3498db;
            text-decoration: none;
        }}
        .nav-header a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="nav-header">
        <h1>📊 Validator Performance - Albert Framework</h1>
        <nav>
            <a href="index.html">← Back to Leaderboard</a> | 
            <a href="solver_tests.html">← Solver Tests</a>
        </nav>
    </div>
    {body_content}
</body>
</html>"""
    
    # Save the page
    output_path = os.path.join(docs_dir, 'validator_performance.html')
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Generated: {output_path}")


def generate_validator_registry_summary(docs_dir: str):
    """Generate a summary of the validator registry."""
    registry_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'solver_tests', 'reports', 'validator_registry_latest.json'
    )
    
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry_data = json.load(f)
        
        # Create a summary file for the leaderboard
        summary = {
            'total_validators': registry_data['total_validators'],
            'tested_validators': registry_data['tested_validators'],
            'untested_validators': registry_data['untested_validators'],
            'last_updated': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(docs_dir, 'validator_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Generated: {summary_path}")


def generate_calibration_page(docs_dir: str):
    """Generate the calibration report page from latest report."""
    # Find the latest calibration report
    calibration_reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'cache', 'calibration', 'reports'
    )
    
    latest_report = os.path.join(calibration_reports_dir, 'calibration_report_latest.html')
    
    if not os.path.exists(latest_report):
        # Generate a placeholder page
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solver Calibration - Albert Framework</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>⚙️ Solver Calibration</h1>
            <nav>
                <a href="index.html">← Back to Leaderboard</a>
            </nav>
        </header>
        
        <main>
            <div class="notice">
                <h2>No Calibration Data Available</h2>
                <p>Calibration tests run automatically before each physics agent run.</p>
                <p>Run <code>python -m physics_agent</code> to generate calibration data.</p>
            </div>
        </main>
    </div>
</body>
</html>"""
    else:
        # Read the latest report and wrap it with navigation
        with open(latest_report, 'r') as f:
            content = f.read()
        
        # Extract the body content
        body_start = content.find('<div class="container">')
        body_end = content.find('</div>\n</body>')
        
        if body_start != -1 and body_end != -1:
            body_content = content[body_start:body_end+6]
        else:
            body_content = '<div class="container"><p>Error loading report content.</p></div>'
        
        # Create wrapped page
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solver Calibration - Albert Framework</title>
    <link rel="stylesheet" href="style.css">
    <style>
        /* Import styles from the report */
        {content[content.find('<style>'):content.find('</style>')+8] if '<style>' in content else ''}
        
        /* Additional navigation styles */
        .nav-header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        .nav-header a {{
            color: #3498db;
            text-decoration: none;
        }}
        .nav-header a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="nav-header">
        <h1>⚙️ Solver Calibration - Albert Framework</h1>
        <nav>
            <a href="index.html">← Back to Leaderboard</a> | 
            <a href="solver_tests.html">Solver Tests →</a>
        </nav>
    </div>
    {body_content}
</body>
</html>"""
    
    # Save the page
    output_path = os.path.join(docs_dir, 'calibration.html')
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Generated: {output_path}")


def main():
    """Generate all validation report pages."""
    # Get docs directory
    docs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'docs'
    )
    
    print("Generating validation report pages...")
    
    # Generate pages
    generate_solver_tests_page(docs_dir)
    generate_validator_performance_page(docs_dir)
    generate_validator_registry_summary(docs_dir)
    generate_calibration_page(docs_dir)
    
    print("\nDone! Validation report pages have been generated.")
    print("Add links to these pages in your leaderboard navigation:")
    print("  - calibration.html")
    print("  - solver_tests.html")
    print("  - validator_performance.html")


if __name__ == '__main__':
    main() 