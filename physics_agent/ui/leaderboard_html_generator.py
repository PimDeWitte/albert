#!/usr/bin/env python3
"""
Leaderboard HTML Generator

Generates a comprehensive leaderboard HTML file that:
1. Ranks all theories by their unification score
2. Embeds HTML files for every candidate
3. Makes all code completely auditable
4. Creates leaderboard.html that references everything correctly
"""

import os
import json
import html
from typing import Dict, Any, List, Tuple
import numpy as np

class LeaderboardHTMLGenerator:
    """Generates comprehensive leaderboard HTML from run results."""
    
    def __init__(self):
        """Initialize the leaderboard generator."""
        
    def generate_leaderboard(self, run_dir: str) -> str:
        """
        Generate a comprehensive leaderboard HTML from all theory results in a run.
        
        Args:
            run_dir: Path to the run directory containing all theory results
            
        Returns:
            Path to the generated leaderboard HTML file
        """
        # Collect all theory results
        theory_results = self._collect_theory_results(run_dir)
        
        if not theory_results:
            print("No theory results found in run directory.")
            return None
            
        # Sort by unified score
        theory_results.sort(key=lambda x: x['unified_score'], reverse=True)
        
        # Generate HTML
        html_content = self._generate_html(theory_results, run_dir)
        
        # Save to file
        output_path = os.path.join(run_dir, 'leaderboard.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"Leaderboard generated: {output_path}")
        
        return output_path
        
    def _collect_theory_results(self, run_dir: str) -> List[Dict[str, Any]]:
        """Collect all theory results from the run directory."""
        results = []
        
        # Iterate through all subdirectories
        for entry in os.listdir(run_dir):
            theory_dir = os.path.join(run_dir, entry)
            
            # Skip non-directories and special directories
            if not os.path.isdir(theory_dir) or entry in ['fail', 'predictions'] or entry.startswith('baseline_'):
                continue
                
            # Load comprehensive scores
            scores_path = os.path.join(theory_dir, 'comprehensive_scores.json')
            if not os.path.exists(scores_path):
                continue
                
            with open(scores_path, 'r') as f:
                scores = json.load(f)
            
            # <reason>chain: Only quantum and UGM theories should be on the leaderboard per project requirements</reason>
            # Filter out classical and test theories - leaderboard is only for quantum/UGM theories
            category = scores.get('category', 'unknown').lower()
            if category not in ['quantum', 'ugm']:
                print(f"  Skipping {scores.get('theory_name', entry)} - category '{category}' not allowed on leaderboard (only quantum/ugm)")
                continue
                
            # Load theory info
            theory_info_path = os.path.join(theory_dir, 'theory_info.json')
            theory_info = {}
            if os.path.exists(theory_info_path):
                with open(theory_info_path, 'r') as f:
                    theory_info = json.load(f)
                    
            # Check for results.html
            has_results_html = os.path.exists(os.path.join(theory_dir, 'results.html'))
            
            # Check for visualizations
            viz_dir = os.path.join(theory_dir, 'viz')
            visualizations = []
            if os.path.exists(viz_dir):
                visualizations = [f for f in os.listdir(viz_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                
            # Extract unified score
            unified_score = 0.0
            if 'comprehensive_score' in scores:
                unified_score = scores['comprehensive_score'].get('final_score', 0.0)
            elif 'overall_scores' in scores:
                unified_score = scores['overall_scores'].get('unified_score', 0.0)
                
            # Create result entry
            result = {
                'theory_name': scores.get('theory_name', 'Unknown'),
                'category': scores.get('category', 'unknown'),
                'unified_score': unified_score,
                'directory': theory_dir,
                'dir_name': entry,
                # <reason>chain: Create JavaScript-safe ID for onclick handlers</reason>
                'js_safe_id': entry.replace("'", "_").replace('"', "_").replace('(', '_').replace(')', '_').replace(' ', '_').replace('=', '_').replace(',', '_').replace('.', '_'),
                'has_results_html': has_results_html,
                'visualizations': visualizations,
                'theory_info': theory_info,
                'comprehensive_scores': scores,
                'component_scores': self._extract_component_scores(scores),
                'validators_passed': self._count_validators_passed(scores),
                'beats_sota_count': self._count_beats_sota(scores)
            }
            
            # Extract particle-specific Kerr and Kerr-Newman losses
            # <reason>chain: Extract losses for charged (electron) and uncharged (photon) particles separately</reason>
            particle_losses = scores.get('particle_trajectory_losses', {})
            
            # Initialize the four loss values we need
            charged_kerr_loss = float('inf')
            charged_kn_loss = float('inf')
            uncharged_kerr_loss = float('inf')
            uncharged_kn_loss = float('inf')
            
            # Process particle-specific losses
            if particle_losses:
                # Look for electron (charged) losses
                if 'electron' in particle_losses:
                    electron_losses = particle_losses['electron']
                    # <reason>chain: Prefer trajectory-based losses over Ricci (which often fails)</reason>
                    # Priority order: trajectory_mse, fft, endpoint_mse, cosine, ricci (fallback)
                    loss_type_priority = ['trajectory_mse', 'fft', 'endpoint_mse', 'cosine', 'ricci']
                    
                    for loss_type in loss_type_priority:
                        if loss_type in electron_losses:
                            baselines = electron_losses[loss_type]
                            if isinstance(baselines, dict):
                                for baseline_name, loss_val in baselines.items():
                                    # <reason>chain: Only use finite values, skip infinity</reason>
                                    if loss_val != float('inf'):
                                        if 'Kerr' in baseline_name and 'Newman' not in baseline_name:
                                            charged_kerr_loss = min(charged_kerr_loss, loss_val)
                                        elif 'Kerr-Newman' in baseline_name or 'Kerr Newman' in baseline_name:
                                            charged_kn_loss = min(charged_kn_loss, loss_val)
                                # <reason>chain: Stop at first loss type with valid values</reason>
                                if charged_kerr_loss != float('inf') or charged_kn_loss != float('inf'):
                                    break
                
                # Look for photon (uncharged) losses  
                if 'photon' in particle_losses:
                    photon_losses = particle_losses['photon']
                    # <reason>chain: Prefer trajectory-based losses over Ricci (which often fails)</reason>
                    # Priority order: trajectory_mse, fft, endpoint_mse, cosine, ricci (fallback)
                    loss_type_priority = ['trajectory_mse', 'fft', 'endpoint_mse', 'cosine', 'ricci']
                    
                    for loss_type in loss_type_priority:
                        if loss_type in photon_losses:
                            baselines = photon_losses[loss_type]
                            if isinstance(baselines, dict):
                                for baseline_name, loss_val in baselines.items():
                                    # <reason>chain: Only use finite values, skip infinity</reason>
                                    if loss_val != float('inf'):
                                        if 'Kerr' in baseline_name and 'Newman' not in baseline_name:
                                            uncharged_kerr_loss = min(uncharged_kerr_loss, loss_val)
                                        elif 'Kerr-Newman' in baseline_name or 'Kerr Newman' in baseline_name:
                                            uncharged_kn_loss = min(uncharged_kn_loss, loss_val)
                                # <reason>chain: Stop at first loss type with valid values</reason>
                                if uncharged_kerr_loss != float('inf') or uncharged_kn_loss != float('inf'):
                                    break
            
            # Store all four loss values (preserve infinity values to show as FAIL)
            result['charged_kerr_loss'] = charged_kerr_loss
            result['charged_kn_loss'] = charged_kn_loss
            result['uncharged_kerr_loss'] = uncharged_kerr_loss
            result['uncharged_kn_loss'] = uncharged_kn_loss
            
            results.append(result)
            
        return results
        
    def _extract_component_scores(self, scores: Dict[str, Any]) -> Dict[str, float]:
        """Extract component scores from comprehensive scores."""
        if 'comprehensive_score' in scores:
            comp_score = scores['comprehensive_score']
            component_scores = comp_score.get('component_scores', {})
            return {
                'constraints': component_scores.get('constraints', 0),
                'observational': component_scores.get('observational', 0),
                'predictions': component_scores.get('predictions', 0),
                'trajectory': component_scores.get('trajectory', 0),
                'unification': component_scores.get('unification', 0)
            }
        else:
            # Legacy format
            return {
                'constraints': scores.get('overall_scores', {}).get('constraint_pass_rate', 0),
                'observational': scores.get('overall_scores', {}).get('observational_pass_rate', 0),
                'predictions': 0,  # Not available in legacy format
                'trajectory': 0,  # Not available in legacy format
                'unification': 0   # Not available in legacy format
            }
            
    def _count_validators_passed(self, scores: Dict[str, Any]) -> Tuple[int, int]:
        """Count how many validators passed. Returns (passed, total)."""
        passed = 0
        total = 0
        
        # Count constraints
        for validator, result in scores.get('constraints', {}).items():
            total += 1
            if result.get('passed', False):
                passed += 1
                
        # Count observational
        for validator, result in scores.get('observational', {}).items():
            total += 1
            if result.get('passed', False):
                passed += 1
                
        # Count predictions
        for validator, result in scores.get('predictions', {}).items():
            total += 1
            if result.get('beats_sota', False):
                passed += 1
                
        return passed, total
        
    def _count_beats_sota(self, scores: Dict[str, Any]) -> int:
        """Count how many predictions beat SOTA."""
        count = 0
        # <reason>chain: Check predictions for beats_sota flag, handling both dict format and legacy format</reason>
        predictions = scores.get('predictions', {})
        for validator, result in predictions.items():
            if isinstance(result, dict):
                # Check if this result beats SOTA
                if result.get('beats_sota', False):
                    count += 1
                # Also check in details if not found at top level
                elif result.get('details', {}).get('beats_sota', False):
                    count += 1
        return count
        
    def _generate_html(self, theory_results: List[Dict[str, Any]], run_dir: str) -> str:
        """Generate the complete HTML content for the leaderboard."""
        # Get run timestamp
        run_timestamp = os.path.basename(run_dir)
        
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '    <meta charset="UTF-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '    <title>Gravity Compression - Quantum/UGM Theory Leaderboard</title>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }',
            '        .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }',
            '        .header h1 { margin: 0; font-size: 2.5em; }',
            '        .header p { margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }',
            '        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }',
            '        .summary { background: white; padding: 20px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '        .summary h2 { margin-top: 0; color: #2c3e50; }',
            '        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }',
            '        .stat-box { background: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }',
            '        .stat-box .value { font-size: 2em; font-weight: bold; color: #3498db; }',
            '        .stat-box .label { color: #7f8c8d; margin-top: 5px; }',
            '        .leaderboard { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '        .leaderboard h2 { margin-top: 0; color: #2c3e50; }',
            '        table { width: 100%; border-collapse: collapse; margin-top: 20px; }',
            '        th { background-color: #34495e; color: white; padding: 12px; text-align: left; position: sticky; top: 0; z-index: 10; }',
            '        td { padding: 12px; border-bottom: 1px solid #ecf0f1; }',
            '        tr:hover { background-color: #f8f9fa; }',
            '        .rank-1 { background-color: #ffd700; font-weight: bold; }',
            '        .rank-2 { background-color: #c0c0c0; }',
            '        .rank-3 { background-color: #cd7f32; }',
            '        .rank-1 td, .rank-2 td, .rank-3 td { color: #2c3e50; }',
            '        .score-bar { background: #ecf0f1; height: 20px; border-radius: 10px; overflow: hidden; position: relative; }',
            '        .score-fill { background: #3498db; height: 100%; transition: width 0.3s; }',
            '        .category-quantum { color: #9b59b6; font-weight: bold; }',
            '        .category-unified { color: #e74c3c; font-weight: bold; }',
            '        .category-classical { color: #95a5a6; font-weight: bold; }',
            '        .expandable { cursor: pointer; user-select: none; }',
            '        .expandable:hover { text-decoration: underline; }',
            '        .details { display: none; background: #f8f9fa; padding: 20px; margin-top: 10px; border-radius: 5px; }',
            '        .details.show { display: block; }',
            '        .btn { background: #3498db; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; display: inline-block; margin: 5px; }',
            '        .btn:hover { background: #2980b9; }',
            '        .btn-theory { background: #e74c3c; }',
            '        .btn-theory:hover { background: #c0392b; }',
            '        .component-scores { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 15px; }',
            '        .component-score { text-align: center; padding: 10px; background: white; border-radius: 5px; }',
            '        .component-score .label { font-size: 0.9em; color: #7f8c8d; }',
            '        .component-score .value { font-size: 1.5em; font-weight: bold; margin-top: 5px; }',
            '        .viz-preview { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 15px; }',
            '        .viz-preview img { width: 100%; height: 150px; object-fit: cover; border-radius: 5px; }',
            '        iframe { width: 100%; height: 800px; border: 1px solid #ddd; border-radius: 5px; margin-top: 15px; }',
            '        .timestamp { color: #7f8c8d; font-style: italic; }',
            '        .loss-good { color: #27ae60; font-weight: bold; }',
            '        .loss-warning { color: #f39c12; font-weight: bold; }',
            '        .loss-bad { color: #e74c3c; font-weight: bold; }',
            '        .loss-na { color: #95a5a6; font-style: italic; }',
            '    </style>',
            '    <script>',
            '        function toggleDetails(id) {',
            '            const details = document.getElementById(id);',
            '            details.classList.toggle("show");',
            '            const button = event.target;',
            '            button.textContent = details.classList.contains("show") ? "Hide Details ▲" : "Show Details ▼";',
            '        }',
                    '        function loadTheoryResult(dirName, safeId) {',
        '            const iframe = document.getElementById("result-iframe-" + safeId);',
        '            iframe.src = dirName + "/results.html";',
        '            iframe.style.display = "block";',
        '        }',
            '    </script>',
            '</head>',
            '<body>',
            '    <div class="header">',
            f'        <h1><a href="./" style="color: white; text-decoration: none;">Run: {run_timestamp}</a></h1>',
            f'        <p>Click to view run directory</p>',
            '    </div>',
            '    <div class="container">',
        ]
        
        # Summary section
        html_parts.extend(self._generate_summary_section(theory_results))
        
        # Leaderboard table
        html_parts.extend([
            '        <div class="leaderboard">',
            '            <h2>Theory Rankings</h2>',
            '            <table>',
            '                <thead>',
            '                    <tr>',
            '                        <th>Rank</th>',
            '                        <th>Theory</th>',
            '                        <th>Category</th>',
            '                        <th>Quantum Score</th>',
            '                        <th>Validators Passed</th>',
            '                        <th>Beats SOTA</th>',
            '                        <th title="Electron (charged) trajectory loss vs Kerr">e⁻ Kerr Loss</th>',
            '                        <th title="Electron (charged) trajectory loss vs Kerr-Newman">e⁻ KN Loss</th>',
            '                        <th title="Photon (uncharged) trajectory loss vs Kerr">γ Kerr Loss</th>',
            '                        <th title="Photon (uncharged) trajectory loss vs Kerr-Newman">γ KN Loss</th>',
            '                        <th>Actions</th>',
            '                    </tr>',
            '                </thead>',
            '                <tbody>',
        ])
        
        # Add theory rows
        for rank, theory in enumerate(theory_results, 1):
            rank_class = f'rank-{rank}' if rank <= 3 else ''
            category_class = f'category-{theory["category"]}'
            
            passed, total = theory['validators_passed']
            validators_text = f"{passed}/{total}"
            
            # Format loss values for charged particle (electron)
            charged_kerr_text = "N/A"
            charged_kerr_class = "loss-na"
            charged_kerr_loss = theory['charged_kerr_loss']
            if charged_kerr_loss is not None and charged_kerr_loss != float('inf'):
                if charged_kerr_loss < 0.001:
                    charged_kerr_text = f"{charged_kerr_loss:.2e}"
                    charged_kerr_class = "loss-good"
                elif charged_kerr_loss < 0.01:
                    charged_kerr_text = f"{charged_kerr_loss:.4f}"
                    charged_kerr_class = "loss-warning"
                else:
                    charged_kerr_text = f"{charged_kerr_loss:.4f}"
                    charged_kerr_class = "loss-bad"
            elif charged_kerr_loss == float('inf'):
                # <reason>chain: Show FAIL when Ricci computation failed (infinity), not N/A</reason>
                charged_kerr_text = "FAIL"
                charged_kerr_class = "loss-bad"
            
            charged_kn_text = "N/A"
            charged_kn_class = "loss-na"
            charged_kn_loss = theory['charged_kn_loss']
            if charged_kn_loss is not None and charged_kn_loss != float('inf'):
                if charged_kn_loss < 0.001:
                    charged_kn_text = f"{charged_kn_loss:.2e}"
                    charged_kn_class = "loss-good"
                elif charged_kn_loss < 0.01:
                    charged_kn_text = f"{charged_kn_loss:.4f}"
                    charged_kn_class = "loss-warning"
                else:
                    charged_kn_text = f"{charged_kn_loss:.4f}"
                    charged_kn_class = "loss-bad"
            elif charged_kn_loss == float('inf'):
                # <reason>chain: Show FAIL when Ricci computation failed (infinity), not N/A</reason>
                charged_kn_text = "FAIL"
                charged_kn_class = "loss-bad"
            
            # Format loss values for uncharged particle (photon)
            uncharged_kerr_text = "N/A"
            uncharged_kerr_class = "loss-na"
            uncharged_kerr_loss = theory['uncharged_kerr_loss']
            if uncharged_kerr_loss is not None and uncharged_kerr_loss != float('inf'):
                if uncharged_kerr_loss < 0.001:
                    uncharged_kerr_text = f"{uncharged_kerr_loss:.2e}"
                    uncharged_kerr_class = "loss-good"
                elif uncharged_kerr_loss < 0.01:
                    uncharged_kerr_text = f"{uncharged_kerr_loss:.4f}"
                    uncharged_kerr_class = "loss-warning"
                else:
                    uncharged_kerr_text = f"{uncharged_kerr_loss:.4f}"
                    uncharged_kerr_class = "loss-bad"
            elif uncharged_kerr_loss == float('inf'):
                # <reason>chain: Show FAIL when Ricci computation failed (infinity), not N/A</reason>
                uncharged_kerr_text = "FAIL"
                uncharged_kerr_class = "loss-bad"
            
            uncharged_kn_text = "N/A"
            uncharged_kn_class = "loss-na"
            uncharged_kn_loss = theory['uncharged_kn_loss']
            if uncharged_kn_loss is not None and uncharged_kn_loss != float('inf'):
                if uncharged_kn_loss < 0.001:
                    uncharged_kn_text = f"{uncharged_kn_loss:.2e}"
                    uncharged_kn_class = "loss-good"
                elif uncharged_kn_loss < 0.01:
                    uncharged_kn_text = f"{uncharged_kn_loss:.4f}"
                    uncharged_kn_class = "loss-warning"
                else:
                    uncharged_kn_text = f"{uncharged_kn_loss:.4f}"
                    uncharged_kn_class = "loss-bad"
            elif uncharged_kn_loss == float('inf'):
                # <reason>chain: Show FAIL when Ricci computation failed (infinity), not N/A</reason>
                uncharged_kn_text = "FAIL"
                uncharged_kn_class = "loss-bad"
            
            html_parts.extend([
                f'                    <tr class="{rank_class}">',
                f'                        <td>{rank}</td>',
                f'                        <td class="expandable" onclick="toggleDetails(\'details-{theory["js_safe_id"]}\')">{html.escape(theory["theory_name"])}</td>',
                f'                        <td class="{category_class}">{theory["category"].capitalize()}</td>',
                f'                        <td>',
                f'                            <div class="score-bar">',
                f'                                <div class="score-fill" style="width: {theory["unified_score"]*100:.1f}%"></div>',
                f'                            </div>',
                f'                            {theory["unified_score"]:.3f}',
                f'                        </td>',
                f'                        <td>{validators_text}</td>',
                f'                        <td>{theory["beats_sota_count"]}</td>',
                f'                        <td><span class="{charged_kerr_class}">{charged_kerr_text}</span></td>',
                f'                        <td><span class="{charged_kn_class}">{charged_kn_text}</span></td>',
                f'                        <td><span class="{uncharged_kerr_class}">{uncharged_kerr_text}</span></td>',
                f'                        <td><span class="{uncharged_kn_class}">{uncharged_kn_text}</span></td>',
                f'                        <td>',
                f'                            <button class="btn expandable" onclick="toggleDetails(\'details-{theory["js_safe_id"]}\')">Show Details ▼</button>',
                f'                        </td>',
                f'                    </tr>',
            ])
            
            html_parts.extend([
                f'                    <tr>',
                f'                        <td colspan="11" style="padding: 0;">',
                f'                            <div class="details" id="details-{theory["js_safe_id"]}">',
            ])
            
            # Add theory details
            html_parts.extend(self._generate_theory_details(theory, run_dir))
            
            html_parts.extend([
                f'                            </div>',
                f'                        </td>',
                f'                    </tr>',
            ])
            
        html_parts.extend([
            '                </tbody>',
            '            </table>',
            '        </div>',
        ])
        
        # Add run log section
        html_parts.extend(self._generate_run_log_section(run_dir))
        
        # Add Scoring Methodology section at the bottom
        html_parts.extend([
            '        <div class="summary" style="margin-top: 30px;">',
            '            <h2>Scoring Methodology</h2>',
            '            <p>The Quantum Score ranks only quantum and UGM (Unified Gravity Model) theories. Classical theories are excluded from the leaderboard.</p>',
            '            <p>The score is calculated as a weighted sum of component scores, adjusted by multipliers:</p>',
            '            <table>',
            '                <thead>',
            '                    <tr>',
            '                        <th>Component</th>',
            '                        <th>Weight</th>',
            '                        <th>Brief Description</th>',
            '                    </tr>',
            '                </thead>',
            '                <tbody>',
            '                    <tr><td>Constraints</td><td>0.20</td><td>Theoretical consistency checks</td></tr>',
            '                    <tr><td>Observational</td><td>0.25</td><td>Matches to quantum experiments</td></tr>',
            '                    <tr><td>Predictions</td><td>0.30</td><td>Novel predictions beating SOTA</td></tr>',
            '                    <tr><td>Trajectory</td><td>0.05</td><td>Particle trajectory matching</td></tr>',
            '                    <tr><td>Unification</td><td>0.20</td><td>Unification potential</td></tr>',
            '                </tbody>',
            '            </table>',
            '            <p>Bonuses/Penalties applied as multipliers. Final score capped at 1.0.</p>',
            '        </div>',
        ])
        
        html_parts.extend([
            '    </div>',
            '</body>',
            '</html>',
        ])
        
        return '\n'.join(html_parts)
        
    def _generate_summary_section(self, theory_results: List[Dict[str, Any]]) -> List[str]:
        """Generate the summary statistics section."""
        total_theories = len(theory_results)
        
        # Category counts
        category_counts = {}
        for theory in theory_results:
            cat = theory['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
        # Best scores
        best_score = theory_results[0]['unified_score'] if theory_results else 0
        avg_score = np.mean([t['unified_score'] for t in theory_results]) if theory_results else 0
        
        # Theories beating SOTA
        theories_with_sota_wins = sum(1 for t in theory_results if t['beats_sota_count'] > 0)
        
        parts = [
            '        <div class="summary">',
            '            <h2>Summary Statistics</h2>',
            '            <div class="stats-grid">',
            '                <div class="stat-box">',
            f'                    <div class="value">{total_theories}</div>',
            '                    <div class="label">Total Theories</div>',
            '                </div>',
            '                <div class="stat-box">',
            f'                    <div class="value">{best_score:.3f}</div>',
            '                    <div class="label">Best Score</div>',
            '                </div>',
            '                <div class="stat-box">',
            f'                    <div class="value">{avg_score:.3f}</div>',
            '                    <div class="label">Average Score</div>',
            '                </div>',
            '                <div class="stat-box">',
            f'                    <div class="value">{theories_with_sota_wins}</div>',
            '                    <div class="label">Beat SOTA</div>',
            '                </div>',
        ]
        
        # Add category counts
        for cat, count in sorted(category_counts.items()):
            # <reason>chain: Only show quantum and UGM categories on the leaderboard</reason>
            if cat.lower() in ['quantum', 'ugm']:
                parts.extend([
                    '                <div class="stat-box">',
                    f'                    <div class="value">{count}</div>',
                    f'                    <div class="label">{cat.upper() if cat == "ugm" else cat.capitalize()} Theories</div>',
                    '                </div>',
                ])
            
        parts.extend([
            '            </div>',
            '        </div>',
        ])
        
        return parts
        
    def _generate_theory_details(self, theory: Dict[str, Any], run_dir: str) -> List[str]:
        """Generate the detailed view for a theory."""
        parts = []
        
        # Component scores
        component_scores = theory['component_scores']
        parts.extend([
            '                                <h3>Component Scores</h3>',
            '                                <div class="component-scores">',
        ])
        
        for component, score in component_scores.items():
            parts.extend([
                '                                    <div class="component-score">',
                f'                                        <div class="label">{component.capitalize()}</div>',
                f'                                        <div class="value">{score:.3f}</div>',
                '                                    </div>',
            ])
            
        parts.append('                                </div>')
        
        # Theory info
        if theory['theory_info']:
            theory_class = theory['theory_info'].get('class_name', 'Unknown')
            theory_module = theory['theory_info'].get('module_name', 'Unknown')
            trajectory_length = theory['theory_info'].get('trajectory_length', 0)
            
            parts.extend([
                '                                <h3>Theory Information</h3>',
                '                                <p>',
                f'                                    <strong>Class:</strong> <code>{theory_class}</code><br>',
                f'                                    <strong>Module:</strong> <code>{theory_module}</code><br>',
                f'                                    <strong>Trajectory Steps:</strong> {trajectory_length}',
            ])
            
            if trajectory_length < 1000:
                parts.append('                                    <span style="color: #e74c3c;"> ⚠️ Low step count!</span>')
                
            parts.append('                                </p>')
            
        # Visualizations preview
        if theory['visualizations']:
            parts.extend([
                '                                <h3>Visualizations</h3>',
                '                                <div class="viz-preview">',
            ])
            
            for viz in theory['visualizations'][:4]:  # Show first 4
                viz_path = f"{theory['dir_name']}/viz/{viz}"
                parts.append(f'                                    <img src="{viz_path}" alt="{viz}">')
                
            parts.append('                                </div>')
            
        # Links
        parts.extend([
            '                                <h3>View Full Results</h3>',
            '                                <p>',
        ])
        
        if theory['has_results_html']:
            parts.append(f'                                    <a href="{theory["dir_name"]}/results.html" target="_blank" class="btn">Open Full Report</a>')
            # <reason>chain: Escape single quotes in dir_name for JavaScript</reason>
            escaped_dir_name = theory["dir_name"].replace("'", "\\'")
            parts.append(f'                                    <button class="btn btn-theory" onclick="loadTheoryResult(\'{escaped_dir_name}\', \'{theory["js_safe_id"]}\')">Load Below</button>')
            
        # Theory source code link
        if theory['theory_info']:
            theory_module = theory['theory_info'].get('module_name', '')
            if theory_module:
                # Check for local code copy first
                local_code = os.path.join(run_dir, theory['dir_name'], 'code', 'theory_source.py')
                if os.path.exists(local_code):
                    parts.append(f'                                    <a href="{theory["dir_name"]}/code/theory_source.py" class="btn btn-theory">View Theory Code</a>')
                else:
                    # Fallback to original location
                    theory_file = theory_module.replace('.', '/') + '.py'
                    parts.append(f'                                    <a href="file://{os.path.abspath(theory_file)}" class="btn btn-theory">View Theory Code</a>')
                
        parts.extend([
            '                                </p>',
            f'                                <iframe id="result-iframe-{theory["js_safe_id"]}" style="display:none;"></iframe>',
        ])
        
        return parts
        
    def _generate_run_log_section(self, run_dir: str) -> List[str]:
        """Generate the run log section with an iframe."""
        parts = []
        
        # Check for log file in run config
        run_config_path = os.path.join(run_dir, 'run_config.json')
        log_file_path = None
        
        if os.path.exists(run_config_path):
            with open(run_config_path, 'r') as f:
                run_config = json.load(f)
                log_file_path = run_config.get('log_file')
                
        # If not in config, look for any run_log*.txt file
        if not log_file_path or not os.path.exists(log_file_path):
            log_files = [f for f in os.listdir(run_dir) if f.startswith('run_log_') and f.endswith('.txt')]
            if log_files:
                # Use the most recent one
                log_files.sort()
                log_file_path = os.path.join(run_dir, log_files[-1])
                
        if log_file_path and os.path.exists(log_file_path):
            # Get relative path for the iframe
            log_filename = os.path.basename(log_file_path)
            
            parts.extend([
                '',
                '        <div class="leaderboard" style="margin-top: 30px;">',
                '            <h2>📝 Full Run Log</h2>',
                '            <p style="margin-bottom: 15px;">',
                '                Complete output log from the entire run, including all theory evaluations, validator outputs, and debugging information.',
                '                <a href="' + log_filename + '" target="_blank" class="btn" style="margin-left: 10px;">Open in New Tab</a>',
                '            </p>',
                '            <iframe src="' + log_filename + '" style="width: 100%; height: 600px; border: 1px solid #ddd; background: #1e1e1e; color: #d4d4d4;"></iframe>',
                '        </div>',
            ])
            
        return parts
        
def main():
    """Main function to generate leaderboard from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate leaderboard HTML from run results')
    parser.add_argument('run_dir', help='Path to the run directory')
    
    args = parser.parse_args()
    
    generator = LeaderboardHTMLGenerator()
    output_path = generator.generate_leaderboard(args.run_dir)
    
    if output_path:
        print(f"Successfully generated: {output_path}")
    else:
        print("Failed to generate leaderboard.")
        
if __name__ == '__main__':
    main() 