#!/usr/bin/env python3
"""
Solver Test Report Generator

Generates comprehensive HTML reports for solver test results, including:
1. Test execution status and timing
2. Performance comparisons
3. Validator coverage matrix
4. Historical trends
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import time


class SolverTestReportGenerator:
    """Generates HTML reports for solver test results."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the report generator."""
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(__file__),
                'reports'
            )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.test_results = {}
        self.timing_results = {}
        self.validator_coverage = {}
    
    def add_test_result(self, test_name: str, passed: bool, execution_time: float,
                       details: Optional[Dict[str, Any]] = None):
        """Add a test result to the report."""
        self.test_results[test_name] = {
            'passed': passed,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
    
    def add_timing_result(self, test_name: str, cached_time: float, uncached_time: float):
        """Add timing comparison results."""
        self.timing_results[test_name] = {
            'cached': cached_time,
            'uncached': uncached_time,
            'speedup': uncached_time / cached_time if cached_time > 0 else 0
        }
    
    def add_validator_coverage(self, validator_name: str, test_function: str, tested: bool):
        """Track which validators are covered by which tests."""
        if validator_name not in self.validator_coverage:
            self.validator_coverage[validator_name] = {}
        self.validator_coverage[validator_name][test_function] = tested
    
    def generate_html_report(self, report_name: str = "solver_test_report") -> str:
        """Generate HTML report and return the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_name}_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        html = self._generate_html()
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        # Also save as latest
        latest_path = os.path.join(self.output_dir, f"{report_name}_latest.html")
        with open(latest_path, 'w') as f:
            f.write(html)
        
        # Save JSON data
        json_path = os.path.join(self.output_dir, f"{report_name}_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump({
                'test_results': self.test_results,
                'timing_results': self.timing_results,
                'validator_coverage': self.validator_coverage,
                'timestamp': timestamp
            }, f, indent=2)
        
        return filepath
    
    def _generate_html(self) -> str:
        """Generate the HTML content."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['passed'])
        total_time = sum(r['execution_time'] for r in self.test_results.values())
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solver Test Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 30px;
            text-align: center;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 36px;
            font-weight: bold;
            color: #3498db;
        }}
        .summary-card.success .value {{
            color: #27ae60;
        }}
        .summary-card.warning .value {{
            color: #f39c12;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .pass {{
            color: #27ae60;
            font-weight: bold;
        }}
        .fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .speedup {{
            color: #3498db;
            font-weight: bold;
        }}
        .coverage-matrix {{
            overflow-x: auto;
        }}
        .coverage-cell {{
            text-align: center;
            padding: 8px;
        }}
        .covered {{
            background-color: #27ae60;
            color: white;
        }}
        .not-covered {{
            background-color: #e74c3c;
            color: white;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 40px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Solver Test Report</h1>
        
        <div class="summary">
            <div class="summary-card success">
                <h3>Tests Passed</h3>
                <div class="value">{passed_tests}/{total_tests}</div>
            </div>
            <div class="summary-card">
                <h3>Success Rate</h3>
                <div class="value">{(passed_tests/total_tests*100 if total_tests > 0 else 0):.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Total Time</h3>
                <div class="value">{total_time:.2f}s</div>
            </div>
            <div class="summary-card warning">
                <h3>Validators Tested</h3>
                <div class="value">{len(self.validator_coverage)}</div>
            </div>
        </div>
        
        <h2>📊 Test Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Execution Time</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for test_name, result in self.test_results.items():
            status_class = 'pass' if result['passed'] else 'fail'
            status_text = 'PASSED' if result['passed'] else 'FAILED'
            details = result.get('details', {})
            details_str = ', '.join(f"{k}: {v}" for k, v in details.items() if k != 'error_trace')
            
            html += f"""
                <tr>
                    <td>{test_name}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result['execution_time']:.3f}s</td>
                    <td>{details_str[:100]}...</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <h2>⚡ Performance Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Name</th>
                    <th>Cached Time (ms)</th>
                    <th>Uncached Time (ms)</th>
                    <th>Speedup</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for test_name, timing in self.timing_results.items():
            html += f"""
                <tr>
                    <td>{test_name}</td>
                    <td>{timing['cached']*1000:.1f}</td>
                    <td>{timing['uncached']*1000:.1f}</td>
                    <td class="speedup">{timing['speedup']:.1f}x</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <h2>✅ Validator Coverage Matrix</h2>
        <div class="coverage-matrix">
            <table>
                <thead>
                    <tr>
                        <th>Validator</th>
                        <th>Test Function</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for validator, tests in self.validator_coverage.items():
            for test_func, tested in tests.items():
                status_class = 'covered' if tested else 'not-covered'
                status_text = '✓ Tested' if tested else '✗ Not Tested'
                html += f"""
                    <tr>
                        <td>{validator}</td>
                        <td>{test_func}</td>
                        <td class="coverage-cell {status_class}">{status_text}</td>
                    </tr>
"""
        
        html += f"""
                </tbody>
            </table>
        </div>
        
        <div class="timestamp">
            Report generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>
"""
        
        return html 