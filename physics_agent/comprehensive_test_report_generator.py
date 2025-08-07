#!/usr/bin/env python3
"""
Generate comprehensive HTML report for theory validation tests.
This combines analytical and solver-based test results into a scientific scorecard.
Version 2: Properly handles run directories and multi-particle visualizations.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import torch
import numpy as np
import html

class ComprehensiveTestReportGenerator:
    """Generate HTML reports for comprehensive theory validation."""
    
    def __init__(self):
        self.theory_categories = {
            'baseline': '#2c3e50',   # Dark blue-gray
            'classical': '#3498db',  # Blue
            'quantum': '#9b59b6'     # Purple
        }
        
    def generate_report(self, results: List[Dict[str, Any]], output_file: str, run_dir: str = None) -> str:
        """Generate comprehensive HTML report from test results."""
        html_lines = []
        
        # Add HTML header and CSS
        html_lines.extend(self._generate_header(run_dir))
        
        # Add summary section
        html_lines.extend(self._generate_summary(results))
        
        # Add detailed rankings
        html_lines.extend(self._generate_rankings(results))
        
        # Add individual theory details
        html_lines.extend(self._generate_theory_details(results))
        
        # Add test descriptions
        html_lines.extend(self._generate_test_descriptions())
        
        # Add log section if available
        if run_dir:
            html_lines.extend(self._generate_log_section(results, run_dir))
        
        # Add footer
        html_lines.extend(self._generate_footer())
        
        # Write to file
        html_content = '\n'.join(html_lines)
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        # Generate trajectory viewers if in run directory
        if run_dir and os.path.exists(run_dir):
            self._generate_trajectory_viewers(results, run_dir)
        
        return output_file
    
    def _generate_header(self, run_dir: str = None) -> List[str]:
        """Generate HTML header with styles and scripts."""
        # Determine if we're in a run directory to adjust paths
        in_run_dir = run_dir is not None
        viz_path = "trajectory_visualizations" if in_run_dir else "../trajectory_visualizations"
        
        return [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '    <meta charset="UTF-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '    <title>Comprehensive Theory Validation Report</title>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }',
            '        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '        h1 { color: #2c3e50; text-align: center; margin-bottom: 10px; }',
            '        .timestamp { text-align: center; color: #7f8c8d; margin-bottom: 30px; }',
            '        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }',
            '        .summary-card { background: #ecf0f1; padding: 20px; border-radius: 6px; text-align: center; }',
            '        .summary-card h3 { margin: 0 0 10px 0; color: #34495e; font-size: 0.9em; text-transform: uppercase; }',
            '        .summary-card .value { font-size: 2.5em; font-weight: bold; color: #3498db; }',
            '        .summary-card .label { color: #7f8c8d; font-size: 0.9em; margin-top: 5px; }',
            '        .sota-card { background: #fff3cd; border: 2px solid #ffc107; }',
            '        .sota-card h3 { color: #856404; }',
            '        .sota-card .value { color: #856404; }',
            '        table { width: auto; max-width: 100%; border-collapse: collapse; margin-bottom: 30px; margin-left: auto; margin-right: auto; }',
            '        th { background-color: #34495e; color: white; padding: 12px; text-align: left; position: sticky; top: 0; z-index: 10; white-space: nowrap; }',
            '        td { padding: 10px; border-bottom: 1px solid #ecf0f1; vertical-align: middle; }',
            '        td.rank, td.score, td.timing-info { white-space: nowrap; }',
            '        tr:hover { background-color: #f8f9fa; }',
            '        .rank { font-weight: bold; text-align: center; }',
            '        .category-baseline { color: #2c3e50; font-weight: bold; }',
            '        .category-classical { color: #3498db; font-weight: bold; }',
            '        .category-quantum { color: #9b59b6; font-weight: bold; }',
            '        .score { text-align: center; font-weight: 500; }',
            '        .pass { color: #27ae60; }',
            '        .fail { color: #e74c3c; }',
            '        .warning { color: #f39c12; }',
            '        .details-section { margin-top: 40px; }',
            '        .theory-detail { background: #f8f9fa; padding: 20px; margin-bottom: 20px; border-radius: 8px; }',
            '        .theory-detail h3 { margin-top: 0; color: #2c3e50; }',
            '        .test-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 15px; }',
            '        .test-result { background: white; padding: 15px; border-radius: 6px; border: 1px solid #ecf0f1; }',
            '        .test-result.pass { border-left: 4px solid #27ae60; }',
            '        .test-result.fail { border-left: 4px solid #e74c3c; }',
            '        .test-result.warning { border-left: 4px solid #f39c12; }',
            '        .test-name { font-weight: 500; margin-bottom: 5px; }',
            '        .test-details { font-size: 0.9em; color: #7f8c8d; }',
            '        .sota-beater { color: #f39c12; font-weight: bold; font-size: 1em; margin-top: 5px; }',
            '        .solver-info { background: #e8f4f8; padding: 8px 12px; border-radius: 4px; margin-top: 5px; font-size: 0.85em; }',
    
            '        .timing-info { color: #7f8c8d; font-size: 0.85em; margin-top: 5px; }',
        '        .not-available { color: #7f8c8d; font-style: italic; text-align: center; }',
            '        .note-box { background: #fff9c4; padding: 15px; border-radius: 6px; margin: 20px 0; border-left: 4px solid #fbc02d; }',
            '        .legend { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-top: 30px; }',
            '        .legend h4 { margin-top: 0; color: #34495e; }',
            '        .legend-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }',
            '        @media (max-width: 768px) {',
            '            .container { padding: 15px; }',
            '            table { font-size: 0.9em; }',
            '            th, td { padding: 8px; }',
            '        }',
            '        .view-trajectory-btn { background: #4a9eff; color: white; border: none; padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; }',
            '        .view-trajectory-btn:hover { background: #357abd; }',
            '        .trajectory-popup { display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); z-index: 1000; max-width: 90vw; max-height: 90vh; overflow: auto; }',
            '        .trajectory-popup.show { display: block; }',
            '        .popup-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 999; }',
            '        .popup-overlay.show { display: block; }',
            '        /* Tooltip styles */',
            '        .tooltip { position: relative; display: inline-block; cursor: help; }',
            '        .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }',
            '        .tooltiptext { visibility: hidden; width: 250px; background-color: #555; color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s; font-size: 12px; line-height: 1.4; }',
            '        .tooltiptext::after { content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: #555 transparent transparent transparent; }',
                    '        .distance-divergence { font-size: 0.85em; line-height: 1.3; }',
        '        .distance-divergence strong { color: #34495e; font-size: 0.9em; }',
            '        .progressive-loss-label { color: #666; font-size: 0.85em; font-weight: normal; }',
            '        /* Log viewer styles */',
            '        .log-section { margin-top: 40px; }',
            '        .log-viewer { background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 8px; font-family: monospace; font-size: 12px; line-height: 1.5; max-height: 400px; overflow-y: auto; white-space: pre-wrap; }',
            '        .log-toggle { background: #666; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-bottom: 10px; }',
            '        .log-toggle:hover { background: #777; }',
            '        .popup-close { position: absolute; top: 10px; right: 10px; background: #e74c3c; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; }',
            '        .trajectory-image { max-width: 100%; height: auto; }',
            '        .trajectory-links { margin-top: 15px; }',
            '        .trajectory-links a { margin-right: 15px; }',
            '        /* Collapsible styles */',
            '        .collapsible { background-color: #f39c12; color: white; cursor: pointer; padding: 10px 18px; width: auto; border: none; text-align: center; outline: none; font-size: 14px; border-radius: 4px; margin-bottom: 10px; transition: background-color 0.3s; }',
            '        .collapsible:hover { background-color: #e67e22; }',
            '        .collapsible:after { content: "\\002B"; color: white; font-weight: bold; float: right; margin-left: 10px; }',
            '        .collapsible.active:after { content: "\\2212"; }',
            '        .collapsible-content { padding: 0; max-height: 0; overflow: hidden; transition: max-height 0.2s ease-out; }',
            '        .collapsible-content.active { max-height: 800px; transition: max-height 0.3s ease-in; }',
            '    </style>',
            '    <script>',
            '        function viewTrajectory(theoryName) {',
            '            const safeName = theoryName.replace(/ /g, "_").replace(/[(),.]/g, "");',
            '            ',
            '            const popup = document.getElementById("trajectory-popup");',
            '            const overlay = document.getElementById("popup-overlay");',
            '            const imageContainer = document.getElementById("trajectory-image-container");',
            '            const titleElement = document.getElementById("trajectory-title");',
            '            ',
            '            titleElement.textContent = theoryName + " - Multi-Particle Trajectories";',
            '            ',
            '            // Generate links for all particles',
            '            const particles = ["electron", "neutrino", "photon", "proton"];',
            '            let particleLinks = \'<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;">\';',
            '            ',
            '            particles.forEach(particle => {',
            f'                const trajectoryPath = `{viz_path}/${{safeName}}_${{particle}}_trajectory.png`;',
            f'                const orbitPath = `{viz_path}/${{safeName}}_${{particle}}_orbit.png`;',
            '                ',
            '                particleLinks += `',
            '                    <div style="background: #f8f9fa; padding: 15px; border-radius: 6px;">',
            '                        <h4 style="margin-top: 0; color: #2c3e50;">${particle.charAt(0).toUpperCase() + particle.slice(1)}</h4>',
            '                        <div style="margin-top: 10px;">',
            '                            <a href="${trajectoryPath}" target="_blank" style="margin-right: 10px;">📊 Analysis</a>',
            '                            <a href="${orbitPath}" target="_blank">🌐 Orbit</a>',
            '                        </div>',
            '                    </div>',
            '                `;',
            '            });',
            '            ',
            '            particleLinks += "</div>";',
            '            ',
            '            imageContainer.innerHTML = `',
            '                ${particleLinks}',
            '                <div class="trajectory-links" style="margin-top: 20px; text-align: center;">',
            '                    <a href="trajectory_viewers/unified_multi_particle_viewer_advanced.html" target="_blank" style="font-size: 1.1em;">🌐 Interactive 3D Viewer (All Theories)</a>',
            '                </div>',
            '            `;',
            '            ',
            '            popup.classList.add("show");',
            '            overlay.classList.add("show");',
            '        }',
            '        ',
            '        function closeTrajectoryPopup() {',
            '            document.getElementById("trajectory-popup").classList.remove("show");',
            '            document.getElementById("popup-overlay").classList.remove("show");',
            '        }',
            '        ',
            '        function toggleCollapsible(button) {',
            '            button.classList.toggle("active");',
            '            const content = button.nextElementSibling;',
            '            content.classList.toggle("active");',
            '        }',
            '    </script>',
            '</head>',
            '<body>',
            '    <div class="container">',
            f'        <h1>Comprehensive Theory Validation Report</h1>',
            f'        <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>'
        ]
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate summary statistics section."""
        total_theories = len(results)
        
        # Calculate aggregate statistics
        total_analytical_tests = sum(r['analytical_summary']['total'] for r in results)
        total_analytical_passed = sum(r['analytical_summary']['passed'] for r in results)
        total_solver_tests = sum(r['solver_summary']['total'] for r in results)
        total_solver_passed = sum(r['solver_summary']['passed'] for r in results)
        
        # Count theories by performance
        perfect_analytical = sum(1 for r in results if r['analytical_summary']['success_rate'] == 1.0)
        perfect_combined = sum(1 for r in results if r['combined_summary']['success_rate'] == 1.0)
        
        # Count theories that beat SOTA
        theories_beating_sota = set()
        total_sota_beats = 0
        
        for result in results:
            theory_name = result['theory']
            theory_beats_sota = False
            
            # Check analytical tests
            for test in result.get('analytical_tests', []):
                if test.get('beats_sota', False):
                    theory_beats_sota = True
                    total_sota_beats += 1
            
            # Check solver tests
            for test in result.get('solver_tests', []):
                if test.get('beats_sota', False):
                    theory_beats_sota = True
                    total_sota_beats += 1
            
            if theory_beats_sota:
                theories_beating_sota.add(theory_name)
        
        num_theories_beating_sota = len(theories_beating_sota)
        
        lines = [
            '        <div class="summary-grid">',
            '            <div class="summary-card">',
            '                <h3>Total Theories</h3>',
            f'                <div class="value">{total_theories}</div>',
            '                <div class="label">Tested</div>',
            '            </div>',
            '            <div class="summary-card">',
            '                <h3>Analytical Tests</h3>',
            f'                <div class="value">{total_analytical_passed}/{total_analytical_tests}</div>',
            '                <div class="label">Passed</div>',
            '            </div>',
            '            <div class="summary-card">',
            '                <h3>Solver Tests</h3>',
            f'                <div class="value">{total_solver_passed}/{total_solver_tests}</div>',
            '                <div class="label">Passed</div>',
            '            </div>',
            '            <div class="summary-card">',
            '                <h3>Perfect Scores</h3>',
            f'                <div class="value">{perfect_analytical}/{perfect_combined}</div>',
            '                <div class="label">Analytical/Combined</div>',
            '            </div>',
            '            <div class="summary-card sota-card">',
            '                <h3>⭐ Beat SOTA</h3>',
            f'                <div class="value">{num_theories_beating_sota}</div>',
            f'                <div class="label">Theories ({total_sota_beats} tests)</div>',
            '            </div>',
            '        </div>'
        ]
        
        # Add details about SOTA beats if any
        if num_theories_beating_sota > 0:
            lines.extend([
                '        <div style="margin-top: 20px;">',
                '            <button class="collapsible" onclick="toggleCollapsible(this)">⭐ View Theories Beating State-of-the-Art ({} theories, {} tests)</button>'.format(num_theories_beating_sota, total_sota_beats),
                '            <div class="collapsible-content">',
                '                <div class="note-box">',
                '                    <ul style="margin-bottom: 0;">',
            ])
            
            # Collect SOTA beats by theory
            sota_details = {}
            for result in results:
                theory_name = result['theory']
                sota_tests = []
                
                # Check analytical tests
                for test in result.get('analytical_tests', []):
                    if test.get('beats_sota', False):
                        sota_tests.append(test['name'])
                
                # Check solver tests
                for test in result.get('solver_tests', []):
                    if test.get('beats_sota', False):
                        sota_tests.append(test['name'])
                
                if sota_tests:
                    sota_details[theory_name] = sota_tests
            
            # Display SOTA beats
            for theory_name in sorted(sota_details.keys()):
                tests = sota_details[theory_name]
                tests_str = ', '.join(tests)
                lines.append(f'                <li><strong>{theory_name}:</strong> {tests_str}</li>')
            
            lines.extend([
                '                    </ul>',
                '                </div>',
                '            </div>',
                '        </div>'
            ])
        
        return lines
    
    def _generate_rankings(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate the ranking table."""
        lines = ['        <h2>Theory Rankings - Combined Score</h2>']
        
        # Sort by combined score
        sorted_results = sorted(results, key=lambda x: -x['combined_summary']['success_rate'])
        
        lines.extend([
            '        <table>',
            '            <thead>',
            '                <tr>',
            '                    <th>Rank</th>',
            '                    <th>Theory</th>',
            '                    <th>Category</th>',
            '                    <th>Combined Score</th>',
            '                    <th>Analytical</th>',
            '                    <th>Solver</th>',
            '                   <th>Distance/Divergence (P1/50/99)</th>',
            '                    <th>Solver Time</th>',
            '                    <th>Actions</th>',
            '                </tr>',
            '            </thead>',
            '            <tbody>'
        ])
        
        for i, result in enumerate(sorted_results, 1):
            # Check if this theory beats any SOTA
            beats_any_sota = False
            for test in result.get('analytical_tests', []) + result.get('solver_tests', []):
                if test.get('beats_sota', False):
                    beats_any_sota = True
                    break
            
            # Extract solver test details
            trajectory_loss = None
            progressive_losses = None
            distance_traveled = None
            kerr_distance = None
            total_solver_time = 0
            total_solver_steps = 0
            cached_trajectory = False
            failed_solver_tests = []
            
            for test in result.get('solver_tests', []):
                if test['status'] not in ['SKIP', 'N/A'] and not test['passed']:
                    # Shorten test names
                    test_name = test['name']
                    if test_name == "Circular Orbit":
                        test_name = "CO"
                    elif test_name == "CMB Power Spectrum":
                        test_name = "CMB"
                    elif test_name == "Primordial GWs":
                        test_name = "PGW"
                    elif test_name == "Quantum Geodesic Sim":
                        test_name = "QGS"
                    elif test_name == "Trajectory vs Kerr":
                        test_name = "TvK"
                    failed_solver_tests.append(test_name)
                
                # Get trajectory loss and distance specifically
                if test['name'] == 'Trajectory vs Kerr':
                    trajectory_loss = test.get('loss')
                    progressive_losses = test.get('progressive_losses')
                    distance_traveled = test.get('distance_traveled')
                    kerr_distance = test.get('kerr_distance')
                    if 'cached' in test.get('solver_type', '').lower():
                        cached_trajectory = True
                
                # Accumulate solver timing (including cached trajectories with metrics)
                # <reason>chain: Include cached trajectories with metrics in timing calculations</reason>
                if (test.get('num_steps', 0) > 0 and test.get('solver_time', 0) > 0 
                    and test['name'] in ['Trajectory vs Kerr', 'Circular Orbit']):
                    # Include cached trajectories if they have timing data
                    if test.get('solver_type', '').endswith('(cached)'):
                        # This is a cached trajectory with metrics
                        total_solver_time += test['solver_time']
                        total_solver_steps += test['num_steps']
                    elif 'cached' not in test.get('solver_type', '').lower():
                        # Regular non-cached trajectory
                        total_solver_time += test['solver_time']
                        total_solver_steps += test['num_steps']
            
            # Format loss - special case for Kerr showing exactly 0.00
            if trajectory_loss is not None:
                if result['theory'] == 'Kerr' and trajectory_loss < 1e-10:
                    loss_str = '0.00'
                else:
                    loss_str = f'{trajectory_loss:.2e}'
            else:
                loss_str = 'N/A'
            

            
            # Format solver info
            solver_str = f"{result['solver_summary']['passed']}/{result['solver_summary']['total']}"
            if failed_solver_tests:
                solver_str += f" ({','.join(failed_solver_tests)})"
            
            # Format timing
            # <reason>chain: Use accumulated timing which now includes cached trajectories with metrics</reason>
            if total_solver_steps > 0 and total_solver_time > 0:
                ms_per_step = total_solver_time / total_solver_steps * 1000
                time_str = f'{total_solver_time:.3f}s ({ms_per_step:.1f}ms/step)'
            elif cached_trajectory:
                # Only show "Cached" for old-style caches without timing metadata
                time_str = 'Cached (no metrics)'
            else:
                time_str = 'N/A'
            
            # Combine distance and progressive losses
            if distance_traveled is not None and kerr_distance is not None:
                # Distances are in geometric units (M = GM/c², black hole radii)
                combined_str = f'<strong>Distance:</strong> <span class="tooltip">{distance_traveled:.1f} <small>M</small><span class="tooltiptext">Distance in black hole radii<br>(M = GM/c²)</span></span>'
                combined_str += f' / {kerr_distance:.1f} <small>M</small>'
                if kerr_distance > 0:
                    ratio = distance_traveled / kerr_distance
                    if abs(ratio - 1.0) > 0.01:  # More than 1% difference
                        if ratio > 1.0:
                            combined_str += f' <span style="color: #e74c3c;">({ratio:.2f}× longer)</span>'
                        else:
                            combined_str += f' <span style="color: #f39c12;">({ratio:.2f}× shorter)</span>'
                
                # Add progressive losses if available
                if progressive_losses:
                    # Calculate average of the three values
                    avg_loss = (progressive_losses["1%"] + progressive_losses["50%"] + progressive_losses["99%"]) / 3
                    
                    combined_str += '<br><strong>Divergence:</strong> '
                    combined_str += f'{progressive_losses["1%"]:.1e} → {progressive_losses["50%"]:.1e} → {progressive_losses["99%"]:.1e}'
                    combined_str += f' <span style="color: #666; font-size: 0.85em;">(avg: {avg_loss:.1e})</span>'
            else:
                combined_str = '<span class="not-available">N/A</span>'
            
            # Add SOTA indicator to theory name if it beats any
            theory_display = result["theory"]
            if beats_any_sota:
                theory_display += ' ⭐'
            
            lines.extend([
                '                <tr>',
                f'                    <td class="rank">{i}</td>',
                f'                    <td class="theory-name">{theory_display}</td>',
                f'                    <td class="category-{result["category"]}">{result["category"]}</td>',
                f'                    <td class="score">{result["combined_summary"]["success_rate"]*100:.1f}%</td>',
                f'                    <td>{result["analytical_summary"]["passed"]}/{result["analytical_summary"]["total"]}</td>',
                f'                    <td>{solver_str}</td>',
                f'                    <td class="distance-divergence">{combined_str}</td>',
                f'                    <td class="timing-info">{time_str}</td>',
                f'                    <td><button class="view-trajectory-btn" onclick="viewTrajectory(\'{result["theory"]}\')">View Trajectory</button></td>',
                '                </tr>'
            ])
        
        lines.extend([
            '            </tbody>',
            '        </table>'
        ])
        
        return lines
    
    def _generate_theory_details(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate detailed results for each theory."""
        lines = ['        <div class="details-section">',
                '            <h2>Detailed Theory Results</h2>']
        
        # Sort by combined score for consistent ordering
        sorted_results = sorted(results, key=lambda x: (-x['combined_summary']['success_rate'], 
                                                        x['combined_summary'].get('complexity_score', float('inf'))))
        
        for result in sorted_results:
            lines.extend([
                f'            <div class="theory-detail">',
                f'                <h3>{result["theory"]} <span class="category-{result["category"]}">({result["category"]})</span></h3>',
                f'                <p>Combined Score: <strong>{result["combined_summary"]["success_rate"]*100:.1f}%</strong> ',
                f'                ({result["combined_summary"]["passed"]}/{result["combined_summary"]["total"]} tests passed)</p>',
                f'                <p><button onclick="viewTrajectory(\'{result["theory"]}\')">🚀 View Multi-Particle Trajectories</button></p>',
                '                <h4>Analytical Tests</h4>',
                '                <div class="test-grid">'
            ])
            
            # Analytical test results
            for test in result['analytical_tests']:
                status_class = 'pass' if test['passed'] else 'fail'
                if test['status'] == 'WARNING':
                    status_class = 'warning'
                
                lines.extend([
                    f'                    <div class="test-result {status_class}">',
                    f'                        <div class="test-name">{test["name"]}</div>',
                    f'                        <div class="test-details">Status: {test["status"]}</div>'
                ])
                
                if test.get('loss') is not None:
                    lines.append(f'                        <div class="test-details">Loss: {test["loss"]:.4f}</div>')
                if test.get('error_percent') is not None:
                    lines.append(f'                        <div class="test-details">Error: {test["error_percent"]:.2f}%</div>')
                
                # Add prediction values and SOTA information
                if test.get('predicted_value') is not None:
                    pred_val = test['predicted_value']
                    units = test.get('units', '')
                    
                    # <reason>chain: Use higher precision for g-2 values to show actual differences</reason>
                    # Check if this is a g-2 test
                    is_g2_test = 'g-2' in test.get('name', '') or units == '(g-2)/2'
                    
                    if is_g2_test:
                        # Show 8 significant figures for g-2 to see the actual anomaly
                        pred_str = f'{pred_val:.8g}'
                    else:
                        # Use standard precision for other tests
                        pred_str = f'{pred_val:.3g}'
                    
                    lines.append(f'                        <div class="test-details">Predicted: {pred_str} {units}</div>')
                    
                    # <reason>chain: Show experimental value for g-2 to explain why test fails</reason>
                    # Add experimental/observed value if different from SOTA
                    if test.get('observed_value') is not None and test.get('observed_value') != test.get('sota_value'):
                        obs_val = test['observed_value']
                        if is_g2_test:
                            obs_str = f'{obs_val:.8g}'
                        else:
                            obs_str = f'{obs_val:.3g}'
                        lines.append(f'                        <div class="test-details">Experimental: {obs_str} {units}</div>')
                    
                    if test.get('sota_value') is not None:
                        sota_val = test['sota_value']
                        
                        if is_g2_test:
                            sota_str = f'{sota_val:.8g}'
                        else:
                            sota_str = f'{sota_val:.3g}'
                        
                        lines.append(f'                        <div class="test-details">SOTA: {sota_str} {units}</div>')
                        
                        # <reason>chain: Only show BEATS SOTA if values are actually different when displayed</reason>
                        # Check if beats_sota is true AND the displayed values are different
                        if test.get('beats_sota'):
                            if pred_str != sota_str:  # Only show if visually different
                                lines.append(f'                        <div class="test-details sota-beater">⭐ BEATS SOTA!</div>')
                            else:
                                # If they appear equal but beats_sota is true, show the actual difference
                                diff = abs(pred_val - sota_val)
                                if diff > 0:
                                    lines.append(f'                        <div class="test-details sota-beater">⭐ BEATS SOTA! (Δ = {diff:.2e})</div>')
                                # If diff is exactly 0, don't show BEATS SOTA at all
                
                # <reason>chain: Add explanatory notes for tests that fail despite matching SOTA</reason>
                # For g-2 tests that fail, explain why
                if is_g2_test and not test.get('passed', True) and test.get('observed_value'):
                    obs_val = test.get('observed_value')
                    if abs(pred_val - sota_val) < 1e-10:  # Prediction matches SOTA
                        lines.append(f'                        <div class="test-details" style="color: #666; font-style: italic; font-size: 0.9em;">Note: Fails because SM (SOTA) doesn\'t explain the experimental anomaly</div>')
                
                # For CMB tests that fail despite matching SOTA
                if 'CMB' in test.get('name', '') and not test.get('passed', True):
                    if abs(pred_val - sota_val) < 0.1:  # Prediction matches SOTA
                        # Check if it's a spatial curvature issue
                        if test.get('notes', '').find('lacks spatial curvature') >= 0:
                            lines.append(f'                        <div class="test-details" style="color: #666; font-style: italic; font-size: 0.9em;">Note: Fails because theory lacks spatial curvature needed for cosmology</div>')
                
                lines.append('                    </div>')
            
            lines.append('                </div>')
            
            # Solver test results
            if result.get('solver_tests'):
                lines.extend([
                    '                <h4>Solver-Based Tests</h4>',
                    '                <div class="test-grid">'
                ])
                
                for test in result['solver_tests']:
                    if test['status'] == 'SKIP':
                        continue
                        
                    status_class = 'pass' if test['passed'] else 'fail'
                    if test['status'] == 'WARNING':
                        status_class = 'warning'
                    
                    lines.extend([
                        f'                    <div class="test-result {status_class}">',
                        f'                        <div class="test-name">{test["name"]}</div>',
                        f'                        <div class="test-details">Status: {test["status"]}</div>'
                    ])
                    
                    if test.get('solver_type'):
                        lines.append(f'                        <div class="solver-info">Solver: {test["solver_type"]}</div>')
                    
                    if test.get('loss') is not None:
                        lines.append(f'                        <div class="test-details">Loss: {test["loss"]:.4e}</div>')
                    
                    # Add prediction values and SOTA information for solver tests
                    if test.get('predicted_value') is not None:
                        pred_val = test['predicted_value']
                        units = test.get('units', '')
                        
                        # <reason>chain: Use higher precision for g-2 values in solver tests too</reason>
                        # Check if this is a g-2 test
                        is_g2_test = 'g-2' in test.get('name', '') or units == '(g-2)/2'
                        
                        if is_g2_test:
                            # Show 8 significant figures for g-2 to see the actual anomaly
                            pred_str = f'{pred_val:.8g}'
                        else:
                            # Use standard precision for other tests
                            pred_str = f'{pred_val:.3g}'
                        
                        lines.append(f'                        <div class="test-details">Predicted: {pred_str} {units}</div>')
                        
                        # <reason>chain: Show experimental value for g-2 in solver tests too</reason>
                        # Add experimental/observed value if different from SOTA
                        if test.get('observed_value') is not None and test.get('observed_value') != test.get('sota_value'):
                            obs_val = test['observed_value']
                            if is_g2_test:
                                obs_str = f'{obs_val:.8g}'
                            else:
                                obs_str = f'{obs_val:.3g}'
                            lines.append(f'                        <div class="test-details">Experimental: {obs_str} {units}</div>')
                        
                        if test.get('sota_value') is not None:
                            sota_val = test['sota_value']
                            
                            if is_g2_test:
                                sota_str = f'{sota_val:.8g}'
                            else:
                                sota_str = f'{sota_val:.3g}'
                            
                            lines.append(f'                        <div class="test-details">SOTA: {sota_str} {units}</div>')
                            
                            # <reason>chain: Only show BEATS SOTA if values are actually different when displayed</reason>
                            # Check if beats_sota is true AND the displayed values are different
                            if test.get('beats_sota'):
                                if pred_str != sota_str:  # Only show if visually different
                                    lines.append(f'                        <div class="test-details sota-beater">⭐ BEATS SOTA!</div>')
                                else:
                                    # If they appear equal but beats_sota is true, show the actual difference
                                    diff = abs(pred_val - sota_val)
                                    if diff > 0:
                                        lines.append(f'                        <div class="test-details sota-beater">⭐ BEATS SOTA! (Δ = {diff:.2e})</div>')
                                    # If diff is exactly 0, don't show BEATS SOTA at all
                    
                    # <reason>chain: Add explanatory notes for solver tests that fail despite matching SOTA</reason>
                    # For g-2 tests that fail, explain why
                    if is_g2_test and not test.get('passed', True) and test.get('observed_value'):
                        obs_val = test.get('observed_value')
                        if abs(pred_val - sota_val) < 1e-10:  # Prediction matches SOTA
                            lines.append(f'                        <div class="test-details" style="color: #666; font-style: italic; font-size: 0.9em;">Note: Fails because SM (SOTA) doesn\'t explain the experimental anomaly</div>')
                    
                    # For CMB solver tests that fail despite matching SOTA
                    if 'CMB' in test.get('name', '') and not test.get('passed', True):
                        if abs(pred_val - sota_val) < 0.1:  # Prediction matches SOTA
                            # Check if it's a spatial curvature issue
                            if test.get('notes', '').find('lacks spatial curvature') >= 0:
                                lines.append(f'                        <div class="test-details" style="color: #666; font-style: italic; font-size: 0.9em;">Note: Fails because theory lacks spatial curvature needed for cosmology</div>')
                    
                    if test.get('exec_time') is not None and test['status'] not in ['SKIP', 'N/A']:
                        exec_time = test['exec_time']
                        solver_time = test.get('solver_time', 0)
                        if 'cached' in test.get('solver_type', '').lower():
                            lines.append(f'                        <div class="timing-info">Using cached trajectory</div>')
                        else:
                            lines.append(f'                        <div class="timing-info">Time: {exec_time:.3f}s (solver: {solver_time:.3f}s)</div>')
                    
                    if test.get('num_steps', 0) > 0:
                        lines.append(f'                        <div class="test-details">Steps: {test["num_steps"]}</div>')
                    
                    lines.append('                    </div>')
                
                lines.append('                </div>')
            
            lines.append('            </div>')
        
        return lines
    
    def _generate_test_descriptions(self) -> List[str]:
        """Generate test methodology descriptions."""
        return [
            '        <div class="legend">',
            '            <h4>Test Methodology</h4>',
            '            <div class="legend-grid">',
            '                <div>',
            '                    <strong>Analytical Tests:</strong>',
            '                    <ul>',
            '                        <li>Mercury Precession</li>',
            '                        <li>Light Deflection</li>',
            '                        <li>Photon Sphere</li>',
            '                        <li>PPN Parameters</li>',
            '                        <li>COW Interferometry</li>',
            '                        <li>Gravitational Waves</li>',
            '                        <li>PSR J0740+6620</li>',
            '                    </ul>',
            '                </div>',
            '                <div>',
            '                    <strong>Solver-Based Tests:</strong>',
            '                    <ul>',
            '                        <li>Trajectory vs Kerr (10k steps)</li>',
            '                        <li>Circular Orbit Conservation</li>',
            '                        <li>Quantum Geodesic Simulation</li>',
            '                        <li>g-2 Muon Anomaly</li>',
            '                        <li>Scattering Amplitudes</li>',
            '                        <li>CMB Power Spectrum</li>',
            '                        <li>Primordial GWs</li>',
            '                    </ul>',
            '                </div>',
            '            </div>',
            '        </div>',
            '        <div class="note-box">',
            '            <strong>Note on Solver Timings:</strong> Cached trajectory results show as "Using cached trajectory" ',
            '            instead of misleading timing values. Actual solver performance is only measured for freshly computed trajectories.',
            '        </div>'
        ]
    
    def _generate_log_section(self, results: List[Dict[str, Any]], run_dir: str) -> List[str]:
        """Generate log section with test output."""
        lines = [
            '        <div class="log-section">',
            '            <h2>Test Execution Logs</h2>',
        ]
        
        # Try to find the main test log
        test_log_path = None
        for filename in os.listdir(run_dir):
            if filename.endswith('.log') or filename.startswith('test_output'):
                test_log_path = os.path.join(run_dir, filename)
                break
        
        if test_log_path and os.path.exists(test_log_path):
            try:
                with open(test_log_path, 'r') as f:
                    log_content = f.read()
                
                lines.extend([
                    '            <button class="log-toggle" onclick="toggleLog(\'main-log\')">Toggle Full Test Log</button>',
                    '            <div id="main-log" class="log-viewer" style="display: none;">',
                    f'{html.escape(log_content)}',
                    '            </div>'
                ])
            except Exception as e:
                lines.append(f'            <p>Could not load test log: {e}</p>')
        
        # Add per-theory logs if available
        lines.append('            <h3>Individual Theory Logs</h3>')
        
        for result in results:
            theory_name = result['theory']
            safe_name = theory_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
            
            # Look for theory-specific log files
            theory_log_found = False
            for test in result.get('solver_tests', []):
                if 'log' in test:
                    theory_log_found = True
                    lines.extend([
                        f'            <h4>{theory_name}</h4>',
                        f'            <button class="log-toggle" onclick="toggleLog(\'{safe_name}-log\')">Toggle {theory_name} Log</button>',
                        f'            <div id="{safe_name}-log" class="log-viewer" style="display: none;">',
                        f'{html.escape(test["log"])}',
                        '            </div>'
                    ])
                    break
        
        lines.extend([
            '            <script>',
            '            function toggleLog(logId) {',
            '                var log = document.getElementById(logId);',
            '                if (log.style.display === "none") {',
            '                    log.style.display = "block";',
            '                } else {',
            '                    log.style.display = "none";',
            '                }',
            '            }',
            '            </script>',
            '        </div>'
        ])
        
        return lines
    
    def _generate_footer(self) -> List[str]:
        """Generate HTML footer."""
        return [
            '    </div>',
            '    <!-- Trajectory Popup -->',
            '    <div id="popup-overlay" class="popup-overlay" onclick="closeTrajectoryPopup()"></div>',
            '    <div id="trajectory-popup" class="trajectory-popup">',
            '        <button class="popup-close" onclick="closeTrajectoryPopup()">✕</button>',
            '        <h2 id="trajectory-title">Trajectory Analysis</h2>',
            '        <div id="trajectory-image-container"></div>',
            '    </div>',
            '</body>',
            '</html>'
        ]
    
    def _generate_trajectory_viewers(self, results: List[Dict[str, Any]], output_dir: str):
        """Generate individual trajectory viewer HTML files for each theory."""
        # Skip individual viewer generation - unified viewer handles this better
        return
        
        viewers_dir = os.path.join(output_dir, 'trajectory_viewers')
        os.makedirs(viewers_dir, exist_ok=True)
        
        try:
            from physics_agent.ui.multi_particle_trajectory_viewer_generator import (
                generate_multi_particle_viewer_from_run,
                generate_multi_particle_trajectory_viewer
            )
        except ImportError:
            print("Warning: Could not import multi-particle trajectory viewer generator")
            return
        
        for result in results:
            theory_name = result['theory']
            
            # Clean theory name for file naming
            clean_name = theory_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            
            # Generate multi-particle viewer
            if output_dir:
                try:
                    viewer_path = os.path.join(viewers_dir, f'{clean_name}_multi_particle_viewer.html')
                    generate_multi_particle_viewer_from_run(
                        theory_name=theory_name,
                        run_dir=output_dir,
                        output_path=viewer_path,
                        black_hole_mass=9.945e13  # Primordial mini BH in kg
                    )
                    continue  # Skip single-particle viewer if multi-particle succeeds
                except Exception as e:
                    print(f"Warning: Could not generate multi-particle viewer for {theory_name}: {e}")
            
            # Fallback to single particle viewer
            trajectory_data = None
            kerr_data = None
            
            for test in result.get('solver_tests', []):
                if test['name'] == 'Trajectory vs Kerr':
                    trajectory_data = test.get('trajectory_data')
                    kerr_data = test.get('kerr_trajectory')
                    break
            
            if trajectory_data is not None:
                try:
                    viewer_path = os.path.join(viewers_dir, f'{clean_name}_trajectory_viewer.html')
                    generate_multi_particle_trajectory_viewer(
                        theory_trajectory=trajectory_data,
                        kerr_trajectory=kerr_data,
                        theory_name=theory_name,
                        output_path=viewer_path,
                        black_hole_mass=9.945e13  # Primordial mini BH
                    )
                except Exception as e:
                    print(f"Warning: Could not generate trajectory viewer for {theory_name}: {e}")