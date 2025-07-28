#!/usr/bin/env python3
"""
Calibration Report Generator

Generates HTML reports for solver calibration results.
Tracks calibration history and identifies trends.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class CalibrationReportGenerator:
    """Generates HTML reports for calibration results."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the report generator."""
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'cache', 'calibration'
            )
        self.cache_dir = cache_dir
        self.reports_dir = os.path.join(cache_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def load_calibration_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Load recent calibration results."""
        history = []
        
        # Find all calibration files
        calibration_files = []
        for file in os.listdir(self.cache_dir):
            if file.startswith('calibration_') and file.endswith('.json') and file != 'calibration_latest.json':
                calibration_files.append(file)
        
        # Sort by timestamp (newest first)
        calibration_files.sort(reverse=True)
        
        # Load limited number of files
        for file in calibration_files[:limit]:
            filepath = os.path.join(self.cache_dir, file)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    history.append(data)
            except:
                continue
        
        return history
    
    def generate_html_report(self) -> str:
        """Generate HTML report and return the file path."""
        # Load latest calibration
        latest_file = os.path.join(self.cache_dir, 'calibration_latest.json')
        if not os.path.exists(latest_file):
            return self._generate_no_data_report()
        
        with open(latest_file, 'r') as f:
            latest = json.load(f)
        
        # Load history
        history = self.load_calibration_history()
        
        # Generate HTML
        html = self._generate_html(latest, history)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_report_{timestamp}.html"
        filepath = os.path.join(self.reports_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        # Also save as latest
        latest_path = os.path.join(self.reports_dir, "calibration_report_latest.html")
        with open(latest_path, 'w') as f:
            f.write(html)
        
        return filepath
    
    def _generate_html(self, latest: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
        """Generate the HTML content."""
        # Calculate statistics
        total_runs = len(history) + 1
        successful_runs = sum(1 for h in history if h.get('success', False)) + (1 if latest.get('success', False) else 0)
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        # Average duration
        durations = [h.get('duration', 0) for h in history if h.get('duration')]
        if latest.get('duration'):
            durations.append(latest['duration'])
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solver Calibration Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</title>
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
        .test-result {{
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            background: #f8f9fa;
        }}
        .test-result.passed {{
            border-left: 4px solid #27ae60;
        }}
        .test-result.failed {{
            border-left: 4px solid #e74c3c;
        }}
        .test-name {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .test-status {{
            color: #27ae60;
        }}
        .test-status.failed {{
            color: #e74c3c;
        }}
        .history-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .history-table th, .history-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .history-table th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        .history-table tr:hover {{
            background-color: #f5f5f5;
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
        <h1>⚙️ Solver Calibration Report</h1>
        
        <div class="summary">
            <div class="summary-card {'success' if latest.get('success', False) else ''}">
                <h3>Latest Status</h3>
                <div class="value">{'PASS' if latest.get('success', False) else 'FAIL'}</div>
            </div>
            <div class="summary-card">
                <h3>Success Rate</h3>
                <div class="value">{success_rate:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Avg Duration</h3>
                <div class="value">{avg_duration:.2f}s</div>
            </div>
            <div class="summary-card">
                <h3>Total Runs</h3>
                <div class="value">{total_runs}</div>
            </div>
        </div>
        
        <h2>Latest Calibration Results</h2>
        <p><strong>Timestamp:</strong> {latest.get('timestamp', 'Unknown')}</p>
        <p><strong>Duration:</strong> {latest.get('duration', 0):.3f} seconds</p>
"""
        
        # Add test results
        if 'results' in latest:
            for test_name, result in latest['results'].items():
                passed = result.get('passed', False)
                error = result.get('error')
                
                html += f"""
        <div class="test-result {'passed' if passed else 'failed'}">
            <div class="test-name">{test_name}</div>
            <div class="test-status {'failed' if not passed else ''}">
                Status: {'PASSED' if passed else 'FAILED'}
                {f' - {error}' if error else ''}
            </div>
        </div>
"""
        
        # Add history table
        html += """
        <h2>Calibration History</h2>
        <table class="history-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Tests Passed</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for h in history[:20]:  # Show last 20 runs
            timestamp = h.get('timestamp', 'Unknown')
            success = h.get('success', False)
            duration = h.get('duration', 0)
            results = h.get('results', {})
            passed_tests = sum(1 for r in results.values() if r.get('passed', False))
            total_tests = len(results)
            
            html += f"""
                <tr>
                    <td>{timestamp}</td>
                    <td style="color: {'#27ae60' if success else '#e74c3c'}">{'PASS' if success else 'FAIL'}</td>
                    <td>{duration:.3f}s</td>
                    <td>{passed_tests}/{total_tests}</td>
                </tr>
"""
        
        html += f"""
            </tbody>
        </table>
        
        <div class="timestamp">
            Report generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_no_data_report(self) -> str:
        """Generate a report when no calibration data is available."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solver Calibration Report - No Data</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        h1 {
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚙️ Solver Calibration Report</h1>
        <p>No calibration data available yet.</p>
        <p>Run <code>python -m physics_agent</code> to generate calibration data.</p>
    </div>
</body>
</html>
"""
        
        filepath = os.path.join(self.reports_dir, "calibration_report_latest.html")
        with open(filepath, 'w') as f:
            f.write(html)
        
        return filepath


# Convenience function
def generate_calibration_report() -> str:
    """Generate a calibration report and return the filepath."""
    generator = CalibrationReportGenerator()
    return generator.generate_html_report()


if __name__ == '__main__':
    # Test report generation
    report_path = generate_calibration_report()
    print(f"Calibration report generated: {report_path}") 