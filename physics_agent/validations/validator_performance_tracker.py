#!/usr/bin/env python3
"""
Validator Performance Tracker

Tracks and reports on validator performance across different theories.
Generates performance reports showing:
1. Execution time per validator per theory
2. Success rates
3. Performance trends
4. Comparative analysis
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# Try to import torch for tensor conversion
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ValidatorPerformanceTracker:
    """Tracks validator performance metrics across theories."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the performance tracker."""
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'cache', 'validator_performance'
            )
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.current_run_metrics = {}
        self.historical_metrics = self._load_historical_metrics()
    
    def _load_historical_metrics(self) -> Dict[str, Any]:
        """Load historical performance metrics."""
        history_file = os.path.join(self.cache_dir, 'performance_history.json')
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load performance history: {e}")
                # Rename corrupted file
                corrupted_file = history_file + '.corrupted'
                try:
                    os.rename(history_file, corrupted_file)
                    print(f"Renamed corrupted file to: {corrupted_file}")
                except:
                    pass
                return {}
        return {}
    
    def _to_json_serializable(self, value):
        """Convert a value to a JSON-serializable format."""
        if HAS_TORCH and isinstance(value, torch.Tensor):
            # Convert tensor to Python scalar or list
            if value.numel() == 1:
                return value.item()
            else:
                return value.tolist()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif isinstance(value, dict):
            return {k: self._to_json_serializable(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._to_json_serializable(v) for v in value]
        else:
            return value

    def _save_historical_metrics(self):
        """Save historical performance metrics."""
        history_file = os.path.join(self.cache_dir, 'performance_history.json')
        # Convert entire metrics to ensure everything is JSON serializable
        serializable_metrics = self._to_json_serializable(self.historical_metrics)
        with open(history_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def start_validator_timing(self, validator_name: str, theory_name: str) -> float:
        """Start timing a validator execution."""
        start_time = time.time()
        key = f"{validator_name}_{theory_name}"
        self.current_run_metrics[key] = {
            'start_time': start_time,
            'validator': validator_name,
            'theory': theory_name
        }
        return start_time
    
    def end_validator_timing(self, validator_name: str, theory_name: str, 
                           result: Dict[str, Any]) -> float:
        """End timing and record results."""
        end_time = time.time()
        key = f"{validator_name}_{theory_name}"
        
        if key not in self.current_run_metrics:
            return 0.0
        
        start_time = self.current_run_metrics[key]['start_time']
        execution_time = end_time - start_time
        
        # Update current run metrics
        self.current_run_metrics[key].update({
            'end_time': end_time,
            'execution_time': execution_time,
            'status': result.get('flags', {}).get('overall', 'UNKNOWN'),
            'loss': result.get('loss', float('inf')),
            'details': result.get('details', {})
        })
        
        # Add to historical metrics
        if validator_name not in self.historical_metrics:
            self.historical_metrics[validator_name] = {}
        if theory_name not in self.historical_metrics[validator_name]:
            self.historical_metrics[validator_name][theory_name] = []
        
        # Convert loss value to ensure it's JSON serializable
        loss_value = result.get('loss', float('inf'))
        loss_value = self._to_json_serializable(loss_value)
        
        self.historical_metrics[validator_name][theory_name].append({
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'status': result.get('flags', {}).get('overall', 'UNKNOWN'),
            'loss': loss_value
        })
        
        self._save_historical_metrics()
        return execution_time
    
    def get_validator_stats(self, validator_name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific validator."""
        if validator_name not in self.historical_metrics:
            return {}
        
        all_times = []
        all_statuses = []
        theory_stats = {}
        
        for theory, runs in self.historical_metrics[validator_name].items():
            times = [r['execution_time'] for r in runs]
            statuses = [r['status'] for r in runs]
            
            theory_stats[theory] = {
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'success_rate': sum(1 for s in statuses if s in ['PASS', 'WARNING']) / len(statuses),
                'run_count': len(runs)
            }
            
            all_times.extend(times)
            all_statuses.extend(statuses)
        
        return {
            'validator': validator_name,
            'overall_avg_time': np.mean(all_times) if all_times else 0,
            'overall_success_rate': sum(1 for s in all_statuses if s in ['PASS', 'WARNING']) / len(all_statuses) if all_statuses else 0,
            'total_runs': len(all_times),
            'theories_tested': len(theory_stats),
            'theory_breakdown': theory_stats
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validators': {},
            'summary': {
                'total_validators': 0,
                'total_executions': 0,
                'average_execution_time': 0,
                'fastest_validator': None,
                'slowest_validator': None
            }
        }
        
        total_time = 0
        total_runs = 0
        validator_avg_times = {}
        
        for validator in self.historical_metrics:
            stats = self.get_validator_stats(validator)
            report['validators'][validator] = stats
            
            if stats['overall_avg_time'] > 0:
                validator_avg_times[validator] = stats['overall_avg_time']
                total_time += stats['overall_avg_time'] * stats['total_runs']
                total_runs += stats['total_runs']
        
        if validator_avg_times:
            report['summary']['fastest_validator'] = min(validator_avg_times, key=validator_avg_times.get)
            report['summary']['slowest_validator'] = max(validator_avg_times, key=validator_avg_times.get)
        
        report['summary']['total_validators'] = len(self.historical_metrics)
        report['summary']['total_executions'] = total_runs
        report['summary']['average_execution_time'] = total_time / total_runs if total_runs > 0 else 0
        
        return report
    
    def generate_html_report(self, output_dir: Optional[str] = None) -> str:
        """Generate HTML performance report."""
        if output_dir is None:
            output_dir = os.path.join(self.cache_dir, 'reports')
        os.makedirs(output_dir, exist_ok=True)
        
        report = self.generate_performance_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validator_performance_{timestamp}.html"
        filepath = os.path.join(output_dir, filename)
        
        html = self._generate_html(report)
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        # Also save as latest
        latest_path = os.path.join(output_dir, "validator_performance_latest.html")
        with open(latest_path, 'w') as f:
            f.write(html)
        
        # Save JSON report
        json_path = os.path.join(output_dir, f"validator_performance_{timestamp}.json")
        # Convert report to ensure everything is JSON serializable
        serializable_report = self._to_json_serializable(report)
        with open(json_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        return filepath
    
    def _generate_html(self, report: Dict[str, Any]) -> str:
        """Generate HTML content for the performance report."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validator Performance Report - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
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
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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
            font-size: 28px;
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
        .performance-bar {{
            background-color: #e9ecef;
            height: 20px;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }}
        .performance-fill {{
            background-color: #3498db;
            height: 100%;
            transition: width 0.3s ease;
        }}
        .fast {{ background-color: #27ae60; }}
        .medium {{ background-color: #f39c12; }}
        .slow {{ background-color: #e74c3c; }}
        .validator-details {{
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Validator Performance Report</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Total Validators</h3>
                <div class="value">{report['summary']['total_validators']}</div>
            </div>
            <div class="summary-card">
                <h3>Total Executions</h3>
                <div class="value">{report['summary']['total_executions']}</div>
            </div>
            <div class="summary-card">
                <h3>Avg Execution Time</h3>
                <div class="value">{report['summary']['average_execution_time']:.3f}s</div>
            </div>
            <div class="summary-card success">
                <h3>Fastest Validator</h3>
                <div class="value" style="font-size: 16px;">{report['summary']['fastest_validator'] or 'N/A'}</div>
            </div>
            <div class="summary-card warning">
                <h3>Slowest Validator</h3>
                <div class="value" style="font-size: 16px;">{report['summary']['slowest_validator'] or 'N/A'}</div>
            </div>
        </div>
        
        <h2>🚀 Performance Overview</h2>
        <table>
            <thead>
                <tr>
                    <th>Validator</th>
                    <th>Avg Time (s)</th>
                    <th>Success Rate</th>
                    <th>Total Runs</th>
                    <th>Theories Tested</th>
                    <th>Performance</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Sort validators by average execution time
        sorted_validators = sorted(
            report['validators'].items(),
            key=lambda x: x[1].get('overall_avg_time', float('inf'))
        )
        
        max_time = max((v['overall_avg_time'] for _, v in sorted_validators if v.get('overall_avg_time', 0) > 0), default=1)
        
        for validator_name, stats in sorted_validators:
            avg_time = stats.get('overall_avg_time', 0)
            success_rate = stats.get('overall_success_rate', 0)
            total_runs = stats.get('total_runs', 0)
            theories_tested = stats.get('theories_tested', 0)
            
            # Determine performance class
            if avg_time < 0.1:
                perf_class = 'fast'
            elif avg_time < 1.0:
                perf_class = 'medium'
            else:
                perf_class = 'slow'
            
            # Calculate bar width
            bar_width = (avg_time / max_time * 100) if max_time > 0 else 0
            
            html += f"""
                <tr>
                    <td><strong>{validator_name}</strong></td>
                    <td>{avg_time:.3f}</td>
                    <td>{success_rate*100:.1f}%</td>
                    <td>{total_runs}</td>
                    <td>{theories_tested}</td>
                    <td>
                        <div class="performance-bar">
                            <div class="performance-fill {perf_class}" style="width: {bar_width}%"></div>
                        </div>
                    </td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
        
        <h2>📈 Detailed Performance by Theory</h2>
"""
        
        # Add detailed breakdown for each validator
        for validator_name, stats in sorted_validators:
            if stats.get('theory_breakdown'):
                html += f"""
        <div class="validator-details">
            <h3>{validator_name}</h3>
            <table>
                <thead>
                    <tr>
                        <th>Theory</th>
                        <th>Avg Time (s)</th>
                        <th>Min Time (s)</th>
                        <th>Max Time (s)</th>
                        <th>Success Rate</th>
                        <th>Runs</th>
                    </tr>
                </thead>
                <tbody>
"""
                
                for theory, theory_stats in stats['theory_breakdown'].items():
                    html += f"""
                    <tr>
                        <td>{theory}</td>
                        <td>{theory_stats['avg_time']:.3f}</td>
                        <td>{theory_stats['min_time']:.3f}</td>
                        <td>{theory_stats['max_time']:.3f}</td>
                        <td>{theory_stats['success_rate']*100:.1f}%</td>
                        <td>{theory_stats['run_count']}</td>
                    </tr>
"""
                
                html += """
                </tbody>
            </table>
        </div>
"""
        
        html += f"""
        <div class="timestamp" style="text-align: center; color: #7f8c8d; margin-top: 40px; font-size: 14px;">
            Report generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>
"""
        
        return html


# Global performance tracker instance
performance_tracker = ValidatorPerformanceTracker() 