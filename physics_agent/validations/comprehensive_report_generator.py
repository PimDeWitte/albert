#!/usr/bin/env python3
"""
Comprehensive HTML Report Generator

Generates detailed HTML reports for each theory with complete validation results,
SOTA comparisons, references, and code links in the exact format required for 
Einstein's legacy research paper.
"""

import os
import json
import html
from typing import Dict, Any, List, Optional
from datetime import datetime

class ComprehensiveReportGenerator:
    """Generates comprehensive HTML reports for theory validation results."""
    
    def __init__(self):
        """Initialize the report generator with validation metadata."""
        self.validation_metadata = self._get_validation_metadata()
        
    def _get_validation_metadata(self) -> Dict[str, Dict[str, Any]]:
        """<reason>chain: Define metadata only for validators that have been tested in solver_tests</reason>"""
        return {
            # Constraint Validators (tested)
            'Conservation Validator': {
                'description': 'Checks energy & angular momentum conservation in trajectories over 1000 steps. Tolerances: energy 1e-12, angular momentum 1e-12, constraint 1e-10 (relative).',
                'purpose': 'Tests numerical stability and physical conservation laws in simulated orbits/trajectories.',
                'reference_links': '[Will (2014) review on numerical GR](https://link.springer.com/article/10.12942/lrr-2014-4)',
                'dataset_links': 'N/A (simulation-based; tolerances from CODATA/numerical standards)',
                'sota_value': 'Drift < 1e-12 (relative)',
                'sota_theory': 'General Relativity (exact conservation)',
                'web_reference': '[Numerical relativity textbook (Baumgarte 2010)](https://global.oup.com/academic/product/numerical-relativity-9780199691531)',
                'validator_file': 'physics_agent/validations/conservation_validator.py'
            },
            'Metric Properties Validator': {
                'description': 'Verifies metric signature (-+++), positive-definiteness of spatial parts, and asymptotic flatness at large r.',
                'purpose': 'Ensures metric is physically valid (Lorentzian, no tachyons, Minkowski limit).',
                'reference_links': '[Wald (1984) GR textbook](https://press.uchicago.edu/ucp/books/book/chicago/G/bo5978156.html)',
                'dataset_links': 'N/A (analytical properties)',
                'sota_value': 'Signature (-,+,+,+), flat at infinity',
                'sota_theory': 'General Relativity (Schwarzschild metric)',
                'web_reference': '[Metric properties in GR](https://en.wikipedia.org/wiki/Metric_tensor_(general_relativity))',
                'validator_file': 'physics_agent/validations/metric_properties_validator.py'
            },
            # Lagrangian Validator removed - not tested
            
            # Classical Observational Validators (tested)
            'Mercury Precession Validator': {
                'description': 'Tests perihelion precession of Mercury. Integrates orbit for one century, measures excess precession vs Newtonian prediction.',
                'purpose': 'Classic test of GR in weak field regime. Validates spacetime curvature effects on planetary orbits.',
                'reference_links': '[Will (2014) Theory and Experiment in Gravitational Physics](https://doi.org/10.1017/CBO9780511564246)',
                'dataset_links': 'N/A (Mercury orbital elements from NASA JPL)',
                'sota_value': '42.98 ± 0.04 arcsec/century',
                'sota_theory': 'General Relativity',
                'web_reference': '[Tests of GR - Mercury](https://en.wikipedia.org/wiki/Tests_of_general_relativity#Perihelion_precession_of_Mercury)',
                'validator_file': 'physics_agent/validations/mercury_precession_validator.py'
            },
            'Light Deflection Validator': {
                'description': 'Calculates deflection of light passing near the Sun. Integrates null geodesics, measures deflection angle at solar limb.',
                'purpose': 'Tests spacetime curvature effect on light propagation. Key prediction of GR.',
                'reference_links': '[Dyson et al. (1920) Eclipse Expedition](https://doi.org/10.1098/rsta.1920.0009)',
                'dataset_links': 'N/A (solar parameters from IAU)',
                'sota_value': '1.7509 ± 0.0003 arcsec',
                'sota_theory': 'General Relativity',
                'web_reference': '[1919 Eclipse - light bending](https://en.wikipedia.org/wiki/Eddington_experiment)',
                'validator_file': 'physics_agent/validations/light_deflection_validator.py'
            },
            'PPN Parameters Validator': {
                'description': 'Computes Parameterized Post-Newtonian parameters (γ, β, etc.) and compares to Solar System constraints.',
                'purpose': 'Comprehensive weak-field test framework. Tests all deviations from GR systematically.',
                'reference_links': '[Will (2018) PPN review](https://link.springer.com/article/10.12942/lrr-2014-4)',
                'dataset_links': 'N/A (Cassini, lunar ranging constraints)',
                'sota_value': 'γ = 1.000 ± 0.002, β = 1.000 ± 0.003',
                'sota_theory': 'General Relativity (γ=β=1)',
                'web_reference': '[PPN formalism](https://en.wikipedia.org/wiki/Parameterized_post-Newtonian_formalism)',
                'validator_file': 'physics_agent/validations/ppn_validator.py'
            },
            'Photon Sphere Validator': {
                'description': 'Calculates photon sphere radius and black hole shadow size. Tests strong-field light behavior.',
                'purpose': 'Tests extreme gravity regime predictions. Directly observable by Event Horizon Telescope.',
                'reference_links': '[EHT Collaboration (2019)](https://doi.org/10.3847/2041-8213/ab0ec7)',
                'dataset_links': 'N/A (M87* and Sgr A* parameters)',
                'sota_value': 'r_ph = 3GM/c² (Schwarzschild)',
                'sota_theory': 'General Relativity',
                'web_reference': '[Black hole shadow](https://en.wikipedia.org/wiki/Black_hole#Photon_sphere)',
                'validator_file': 'physics_agent/validations/photon_sphere_validator.py'
            },
            'GW Waveform Validator': {
                'description': 'Generates gravitational wave inspiral waveforms and cross-correlates with GR templates.',
                'purpose': 'Tests dynamic strong-field gravity. Validates theory against LIGO/Virgo observations.',
                'reference_links': '[Abbott et al. (2016) GW150914](https://doi.org/10.1103/PhysRevLett.116.061102)',
                'dataset_links': '[GWOSC strain data](https://gwosc.org)',
                'sota_value': 'Correlation > 0.95 with GR',
                'sota_theory': 'General Relativity',
                'web_reference': '[LIGO detections](https://en.wikipedia.org/wiki/List_of_gravitational_wave_observations)',
                'validator_file': 'physics_agent/validations/gw_validator.py'
            },
            # Hawking Radiation Validator removed - not tested
            # Cosmology Validator removed - not tested
            # PsrJ0740Validator removed - test exists but not in main suite
            
            # Quantum Observational Validators (tested)
            'COW Neutron Interferometry Validator': {
                'description': 'Measures phase shift in neutron interferometry due to gravitational potential difference. Uses Colella-Overhauser-Werner (COW) experiment setup with neutron wavelength 2.2 Å, enclosed area 0.3 cm², height difference 0.1 m. Compares predicted phase shift against observed 2.70 ± 0.21 radians.',
                'purpose': 'Tests quantum effects in gravity, specifically gravitational phase shift in interferometry. Validates semiclassical gravity predictions.',
                'reference_links': '[Original COW paper (1975)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.34.1472); [Review on quantum gravity tests (2023)](https://arxiv.org/abs/2305.10478)',
                'dataset_links': 'N/A (theoretical prediction vs. historical measurement; no external dataset file, values hardcoded from paper)',
                'sota_value': '2.70 radians',
                'sota_theory': 'General Relativity (semiclassical limit)',
                'web_reference': '[COW experiment Wikipedia with references](https://en.wikipedia.org/wiki/Colella%E2%80%93Overhauser%E2%80%93Werner_experiment)',
                'validator_file': 'physics_agent/validations/cow_interferometry_validator.py'
            },
            # Atom Interferometry Validator removed - not tested
            # Gravitational Decoherence Validator removed - not tested
            # Quantum Clock Validator removed - not tested
            # Quantum Lagrangian Grounding Validator removed - not tested
            
            # Unification Validators removed - not tested
            
            # Prediction Validators (tested)
            'CMB Power Spectrum Prediction Validator': {
                'description': 'Computes χ²/dof for TT spectrum (l=2-30) vs. Planck 2018 data. Compares to ΛCDM SOTA (χ²/dof ~53.08).',
                'purpose': 'Tests theory\'s prediction for primordial fluctuations and cosmology.',
                'reference_links': '[Planck 2018 results](https://www.aanda.org/articles/aa/abs/2020/09/aa35332-19/aa35332-19.html)',
                'dataset_links': '[Planck TT spectrum data](https://pla.esac.esa.int/pla/#cosmology) (COM_PowerSpect_CMB-TT-full_R3.01.txt)',
                'sota_value': 'χ²/dof ≈53.08',
                'sota_theory': 'ΛCDM cosmology',
                'web_reference': '[Planck 2018 cosmology paper](https://arxiv.org/abs/1807.06209)',
                'validator_file': 'physics_agent/validations/cmb_power_spectrum_validator.py'
            },
            # PTA Stochastic GW Background Validator removed - not tested
            'Primordial GWs Validator': {
                'description': 'Predicts tensor-to-scalar ratio r and tilt n_t vs. standard inflation (r<0.032 upper limit from BICEP/Keck).',
                'purpose': 'Tests inflationary predictions and tensor modes.',
                'reference_links': '[BICEP/Keck 2023](https://arxiv.org/abs/2310.05224)',
                'dataset_links': 'N/A (upper limits; no full dataset, derived constraints)',
                'sota_value': 'r < 0.032',
                'sota_theory': 'Single-field slow-roll inflation',
                'web_reference': '[BICEP/Keck collaboration paper (2023)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.131.041001)',
                'validator_file': 'physics_agent/validations/primordial_gws_validator.py'
            },
            # Future Detectors Validator removed - not implemented
            # Novel Signatures Validator removed - not implemented
            
            # Trajectory Matching (not a validator but included in reports)
            'Trajectory Matching': {
                'description': 'Compares simulated trajectories (e.g., orbits) to baselines like Kerr, with visualizations (checkpoints, multi-particle grids). Checks deviation over 1000 steps.',
                'purpose': 'Tests geodesic accuracy and stability against known solutions.',
                'reference_links': '[Teukolsky (2015) numerical GR review](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.87.1083)',
                'dataset_links': 'N/A (simulation-based)',
                'sota_value': 'Deviation <1e-10 (relative)',
                'sota_theory': 'Kerr metric (exact solution)',
                'web_reference': '[Kerr metric Wikipedia](https://en.wikipedia.org/wiki/Kerr_metric)',
                'validator_file': 'N/A (part of trajectory calculation)'
            }
        }
        
    def generate_theory_report(self, theory_name: str, theory_results: Dict[str, Any], 
                             output_dir: str, logs: Optional[str] = None) -> str:
        """
        <reason>chain: Generate comprehensive HTML report for a single theory</reason>
        
        Args:
            theory_name: Name of the theory
            theory_results: Dictionary containing all validation results
            output_dir: Directory to save the report
            logs: Optional logs to include
            
        Returns:
            Path to generated HTML file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'results.html')
        
        # Generate HTML content
        html_content = self._generate_html(theory_name, theory_results, output_dir, logs)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return output_file
        
    def _generate_html(self, theory_name: str, theory_results: Dict[str, Any], 
                      output_dir: str, logs: Optional[str] = None) -> str:
        """<reason>chain: Generate the complete HTML content with table and logs</reason>"""
        
        # Start HTML document
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '    <meta charset="UTF-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'    <title>{html.escape(theory_name)} - Comprehensive Validation Results</title>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }',
            '        h1 { color: #333; text-align: center; }',
            '        h2 { color: #666; margin-top: 30px; }',
            '        table { border-collapse: collapse; width: 100%; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '        th { background-color: #4CAF50; color: white; padding: 12px; text-align: left; position: sticky; top: 0; }',
            '        td { padding: 12px; border-bottom: 1px solid #ddd; }',
            '        tr:hover { background-color: #f5f5f5; }',
            '        .code-button { background-color: #008CBA; color: white; padding: 5px 10px; text-decoration: none; border-radius: 3px; }',
            '        .code-button:hover { background-color: #005f79; }',
            '        .pass { color: green; font-weight: bold; }',
            '        .fail { color: red; font-weight: bold; }',
            '        .na { color: #999; font-style: italic; }',
            '        .logs { background-color: #1e1e1e; color: #d4d4d4; padding: 20px; font-family: monospace; white-space: pre-wrap; overflow-x: auto; margin-top: 30px; }',
            '        .meta-info { background-color: #e8f4f8; padding: 15px; margin-bottom: 20px; border-radius: 5px; }',
            '        .sota-match { background-color: #c8e6c9; }',
            '        .sota-beat { background-color: #81c784; font-weight: bold; }',
            '        .timestamp { color: #666; font-style: italic; }',
            '        .viz-container { background-color: white; padding: 20px; margin-top: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '        .viz-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }',
            '        .viz-item { text-align: center; }',
            '        .viz-item img { max-width: 100%; height: auto; border: 1px solid #ddd; max-height: 600px; }',
            '        .warning { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin: 10px 0; }',
            '        .error { background-color: #ffebee; color: #c62828; padding: 2px 6px; border-radius: 3px; }',
            '        details { margin: 10px 0; }',
            '        details summary { cursor: pointer; color: #1976d2; font-weight: 500; }',
            '        details summary:hover { text-decoration: underline; }',
            '    </style>',
            '</head>',
            '<body>',
            f'    <h1>Comprehensive Validation Results: {html.escape(theory_name)}</h1>',
            f'    <div class="meta-info">',
            f'        <p><strong>Theory:</strong> {html.escape(theory_name)}</p>',
            f'        <p><strong>Generated:</strong> <span class="timestamp">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span></p>',
            f'        <p><strong>Total Validators Run:</strong> {len(self.validation_metadata)}</p>',
        ]
        
        # Add visualization section first (moved to top)
        viz_section = self._generate_visualization_section(output_dir, theory_results)
        html_parts.extend(viz_section)
        
        # Add theory code reference
        theory_info_path = os.path.join(output_dir, 'theory_info.json')
        if os.path.exists(theory_info_path):
            with open(theory_info_path, 'r') as f:
                theory_info = json.load(f)
            theory_class = theory_info.get('class_name', 'Unknown')
            theory_module = theory_info.get('module_name', 'Unknown')
            
            # Check for local theory code copy
            local_theory_code = os.path.join(output_dir, 'code', 'theory_source.py')
            if os.path.exists(local_theory_code):
                html_parts.append(f'        <p><strong>Theory Implementation:</strong> <code>{theory_class}</code> in <a href="code/theory_source.py" class="code-button">Show Theory Code</a> (<a href="code/theory_instance.py">Instance</a>)</p>')
            else:
                # Fallback to original location
                theory_file = theory_module.replace('.', '/') + '.py' if theory_module != 'Unknown' else 'Unknown'
                html_parts.append(f'        <p><strong>Theory Implementation:</strong> <code>{theory_class}</code> in <a href="file://{os.path.abspath(theory_file)}" class="code-button">Show Theory Code</a></p>')
        
        # Add command line info if available
        run_config_path = os.path.join(os.path.dirname(output_dir), 'run_config.json')
        if os.path.exists(run_config_path):
            with open(run_config_path, 'r') as f:
                run_config = json.load(f)
            command_line = run_config.get('command_line', 'N/A')
            html_parts.append(f'        <p><strong>Command Line:</strong> <code>{html.escape(command_line)}</code></p>')
        
        html_parts.extend([
            '    </div>',
            '    <table>',
            '        <thead>',
            '            <tr>',
            '                <th>Test Name</th>',
            '                <th>Description</th>',
            '                <th>Purpose/What it Tests</th>',
            '                <th>Reference Links</th>',
            '                <th>Dataset Links</th>',
            '                <th>SOTA Value</th>',
            '                <th>SOTA Theory</th>',
            '                <th>Web Reference for SOTA</th>',
            '                <th>Our System\'s Score</th>',
            '                <th>Python File</th>',
            '            </tr>',
            '        </thead>',
            '        <tbody>'
        ])
        
        # Add rows for each validator
        for validator_name, metadata in self.validation_metadata.items():
            # Get the result for this validator
            result = self._get_validator_result(validator_name, theory_results)
            
            # Format our score
            our_score = self._format_our_score(validator_name, result)
            
            # Determine if we beat SOTA
            beats_sota = self._check_beats_sota(validator_name, result)
            score_class = 'sota-beat' if beats_sota else ('sota-match' if result.get('passed', False) else '')
            
            # Generate table row
            html_parts.extend([
                '            <tr>',
                f'                <td>{html.escape(validator_name)}</td>',
                f'                <td>{html.escape(metadata["description"])}</td>',
                f'                <td>{html.escape(metadata["purpose"])}</td>',
                f'                <td>{metadata["reference_links"]}</td>',
                f'                <td>{metadata["dataset_links"]}</td>',
                f'                <td>{html.escape(metadata["sota_value"])}</td>',
                f'                <td>{html.escape(metadata["sota_theory"])}</td>',
                f'                <td>{metadata["web_reference"]}</td>',
                f'                <td class="{score_class}">{our_score}</td>',
                f'                <td><a href="file://{os.path.abspath(metadata["validator_file"])}" class="code-button">Show Code</a></td>',
                '            </tr>'
            ])
            
        html_parts.extend([
            '        </tbody>',
            '    </table>'
        ])
        
        # Add summary statistics
        summary = self._generate_summary(theory_results)
        html_parts.extend([
            '    <h2>Summary Statistics</h2>',
            '    <div class="meta-info">',
            f'        {summary}',
            '    </div>'
        ])
        
        # <reason>chain: Add dedicated error section for validation exceptions</reason>
        error_section = self._generate_error_section(theory_results)
        if error_section:
            html_parts.extend([
                '    <h2>Validation Errors</h2>',
                '    <div class="warning" style="background-color: #ffebee; border-left: 4px solid #f44336;">',
                f'        {error_section}',
                '    </div>'
            ])
            
        # Visualization section already added at top
        
        # Add logs if provided
        if logs:
            html_parts.extend([
                '    <h2>Execution Logs</h2>',
                '    <div class="logs">',
                html.escape(logs),
                '    </div>'
            ])
            
        # Close HTML
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)
        
    def _get_validator_result(self, validator_name: str, theory_results: Dict[str, Any]) -> Dict[str, Any]:
        """<reason>chain: Extract result for specific validator from theory results - now includes all validators</reason>"""
        # <reason>chain: Only include validators that have been tested in solver_tests</reason>
        # Map validator names to result keys and categories
        validator_map = {
            # Constraint validators (tested)
            'Conservation Validator': ('Conservation Validator', 'constraints'),
            'Metric Properties Validator': ('Metric Properties Validator', 'constraints'),
            # 'Lagrangian Validator': removed - not tested
            
            # Classical observational validators (tested)
            'Mercury Precession Validator': ('MercuryPrecessionValidator', 'observational'),
            'Light Deflection Validator': ('LightDeflectionValidator', 'observational'),
            'PPN Parameters Validator': ('PpnValidator', 'observational'),
            'Photon Sphere Validator': ('PhotonSphereValidator', 'observational'),
            'GW Waveform Validator': ('GwValidator', 'observational'),
            # 'Hawking Radiation Validator': removed - not tested
            # 'Cosmology Validator': removed - not tested
            # 'PsrJ0740Validator': removed - test exists but not in main suite
            
            # Quantum observational validators (tested)
            'COW Neutron Interferometry Validator': ('COWInterferometryValidator', 'observational'),
            # 'Atom Interferometry Validator': removed - not tested
            # 'Gravitational Decoherence Validator': removed - not tested
            # 'Quantum Clock Validator': removed - not tested
            # 'Quantum Lagrangian Grounding Validator': removed - not tested
            
            # Unification validators
            # 'Renormalizability Validator': removed - not tested
            # 'Unification Scale Validator': removed - not tested
            
            # Prediction validators (tested)
            'CMB Power Spectrum Prediction Validator': ('CMB Power Spectrum Prediction Validator', 'predictions'),
            # 'PTA Stochastic GW Background Validator': removed - not tested
            'Primordial GWs Validator': ('Primordial GWs Validator', 'predictions'),
            # 'Future Detectors Validator': ('Future Detectors Validator', 'predictions'),  # Removed - not implemented
            # 'Novel Signatures Validator': ('Novel Signatures Validator', 'predictions'),  # Removed - not implemented
            
            # Special case
            'Trajectory Matching': ('trajectory_matching', 'trajectory')
        }
        
        # Get result from comprehensive scores structure
        if validator_name not in validator_map:
            return {}
            
        result_key, category = validator_map[validator_name]
        
        # Handle trajectory matching specially
        if category == 'trajectory':
            if 'trajectory_losses' in theory_results:
                trajectory_losses = theory_results['trajectory_losses']
                # Find best loss across all types
                best_loss = float('inf')
                best_loss_type = None
                for loss_type, baselines in trajectory_losses.items():
                    for baseline, loss_val in baselines.items():
                        if loss_val < best_loss:
                            best_loss = loss_val
                            best_loss_type = f"{loss_type} vs {baseline}"
                
                # Always return a result, even if best_loss is inf
                return {
                    'best_loss': best_loss,
                    'best_loss_type': best_loss_type if best_loss_type else 'all baselines',
                    'passed': best_loss < 1e-3  # Threshold for trajectory matching
                }
            elif 'overall_scores' in theory_results:
                if 'best_loss' in theory_results['overall_scores']:
                    return {
                        'best_loss': theory_results['overall_scores']['best_loss'],
                        'best_loss_type': theory_results['overall_scores'].get('best_loss_type', 'unknown'),
                        'passed': theory_results['overall_scores']['best_loss'] < 1e-3
                    }
            return {}
            
        # For other validators, check in category sections
        if category in theory_results:
            if result_key in theory_results[category]:
                result = theory_results[category][result_key]
                # Convert 'details' to top-level fields for consistency
                if 'details' in result:
                    merged = {**result, **result['details']}
                    return merged
                return result
                
        # Also check in validation results sections
        validation_sections = [
            'quantum_validation_results',
            'constraint_validation_results', 
            'prediction_validation_results',
            'observational_validation_results'
        ]
        
        for section in validation_sections:
            if section in theory_results:
                if result_key in theory_results[section]:
                    result = theory_results[section][result_key]
                    if 'details' in result:
                        merged = {**result, **result['details']}
                        return merged
                    return result
        
        # <reason>chain: Check in the raw validation list as well</reason>
        # Sometimes results are stored in a 'validations' array
        if 'validations' in theory_results:
            # First try exact match with result_key
            for val in theory_results['validations']:
                if val.get('validator') == result_key:
                    if 'details' in val:
                        merged = {**val, **val['details']}
                        return merged
                    return val
            
            # <reason>chain: Also check for direct class name match</reason>
            # For cases where validator name doesn't go through the mapping
            # e.g., PpnValidator, PhotonSphereValidator, GwValidator
            class_names = {
                'PPN Parameters Validator': 'PpnValidator',
                'Photon Sphere Validator': 'PhotonSphereValidator',
                'GW Waveform Validator': 'GwValidator',
                'CMB Power Spectrum Prediction Validator': 'CMBPowerSpectrumValidator',
                'Primordial GWs Validator': 'PrimordialGWsValidator'
            }
            
            if validator_name in class_names:
                class_name = class_names[validator_name]
                for val in theory_results['validations']:
                    if val.get('validator') == class_name:
                        if 'details' in val:
                            merged = {**val, **val['details']}
                            return merged
                        return val
                    
        return {}
        
    def _format_our_score(self, validator_name: str, result: Dict[str, Any]) -> str:
        """<reason>chain: Format our score for display in table - always show score, not just pass/fail</reason>"""
        if not result:
            return '<span class="na">N/A - Not Run</span>'
            
        # <reason>chain: Check for ERROR status from exceptions first</reason>
        if result.get('flags', {}).get('overall') == 'ERROR':
            error_msg = result.get('flags', {}).get('details', 'Unknown error')
            error_type = result.get('details', {}).get('error_type', 'Exception')
            # Show error with red background and details
            return f'<span class="error" style="background-color: #ffebee; color: #c62828; padding: 2px 6px; border-radius: 3px;">ERROR - {error_type}: {html.escape(str(error_msg)[:100])}{"..." if len(str(error_msg)) > 100 else ""}</span>'
            
        # Special handling for different validators
        if 'Trajectory' in validator_name:
            if 'best_loss' in result:
                loss = result['best_loss']
                if loss == float('inf'):
                    return '<span class="fail">FAIL - SCORE: ∞ (diverged)</span>'
                passed = result.get('passed', False)
                status = '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'
                return f"{status} - SCORE: {loss:.2e} ({result.get('best_loss_type', 'unknown')})"
            return '<span class="na">N/A - Not Run</span>'
            
        # For prediction validators with specific values
        if validator_name == 'CMB Power Spectrum Prediction Validator':
            chi2_dof = result.get('theory_value', result.get('chi2_dof', result.get('predicted_value')))
            if chi2_dof is not None:
                passed = chi2_dof <= 53.08  # SOTA value (≤ to include ties)
                # <reason>chain: Check beats_sota flag or calculate from score</reason>
                beats_sota = result.get('beats_sota', False) or chi2_dof < 53.08
                status = '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'
                sota_indicator = ' ⭐' if beats_sota else ''
                return f"{status} - SCORE: χ²/dof = {chi2_dof:.2f}{sota_indicator}"
                
        elif validator_name == 'PTA Stochastic GW Background Validator':
            lnL = result.get('theory_value', result.get('log_likelihood'))
            if lnL is not None:
                passed = lnL > -0.08  # SOTA value
                # <reason>chain: Check beats_sota flag or calculate from score</reason>
                beats_sota = result.get('beats_sota', False) or lnL > -0.08
                status = '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'
                sota_indicator = ' ⭐' if beats_sota else ''
                return f"{status} - SCORE: lnL = {lnL:.2f}{sota_indicator}"
                
        elif validator_name == 'Primordial GWs Validator':
            r = result.get('theory_value', result.get('predicted_value'))
            if r is not None:
                passed = 0 < r < 0.032  # Below upper limit
                # <reason>chain: For primordial GWs, beating SOTA means having better likelihood</reason>
                beats_sota = result.get('beats_sota', False)
                status = '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'
                sota_indicator = ' ⭐' if beats_sota else ''
                return f"{status} - SCORE: r = {r:.3f}{sota_indicator}"
                
        # elif validator_name == 'Future Detectors Validator':  # Removed - not implemented
        #     snr = result.get('theory_value', result.get('predicted_value', 0))
        #     if snr is not None:
        #         passed = snr > 1000  # SOTA value  
        #         beats_sota = result.get('beats_sota', False)
        #         status = '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'
        #         sota_indicator = ' ⭐' if beats_sota else ''
        #         return f"{status} - SCORE: SNR = {snr:.1f}{sota_indicator}"
                
        # elif validator_name == 'Novel Signatures Validator':  # Removed - not implemented
        #     bf = result.get('theory_value', result.get('bayes_factor', 0))
        #     if bf is not None:
        #         passed = bf > 5  # SOTA value
        #         beats_sota = result.get('beats_sota', False)
        #         status = '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'
        #         sota_indicator = ' ⭐' if beats_sota else ''
        #         return f"{status} - SCORE: BF = {bf:.1f}{sota_indicator}"
            
        # Standard result format for observational validators
        if 'predicted_value' in result or 'predicted' in result:
            value = result.get('predicted_value', result.get('predicted'))
            result.get('observed_value', result.get('observed'))
            error_pct = result.get('error_percent')
            units = result.get('units', '')
            passed = result.get('passed', False)
            
            # Format value
            if isinstance(value, (int, float)):
                if abs(value) < 1e-3 or abs(value) > 1e3:
                    formatted = f"{value:.2e}"
                else:
                    formatted = f"{value:.4f}"
            else:
                formatted = str(value)
                
            # Add units
            if units:
                formatted += f" {units}"
                
            # Add pass/fail
            status = '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'
            
            # Add error if available
            error_str = ""
            if error_pct is not None:
                error_str = f" (Error: {error_pct:.1f}%)"
                
            return f"{status} - SCORE: {formatted}{error_str}"
            
        # If we have a simple pass/fail result with loss
        if 'loss' in result:
            passed = result.get('passed', result.get('loss', 1.0) < 0.01)
            status = '<span class="pass">PASS</span>' if passed else '<span class="fail">FAIL</span>'
            return f"{status} - SCORE: {result['loss']:.2e}"
            
        # If we only have pass/fail
        if 'passed' in result:
            status = '<span class="pass">PASS</span>' if result['passed'] else '<span class="fail">FAIL</span>'
            return f"{status} - SCORE: Not Available"
            
        return '<span class="na">N/A - Incomplete Data</span>'
        
    def _check_beats_sota(self, validator_name: str, result: Dict[str, Any]) -> bool:
        """<reason>chain: Check if result beats SOTA</reason>"""
        if not result:
            return False
            
        # Direct check
        if 'beats_sota' in result:
            return result['beats_sota']
            
        # Validator-specific checks
        if validator_name == 'CMB Power Spectrum Prediction Validator':
            chi2_dof = result.get('theory_value', result.get('chi2_dof', result.get('predicted_value', float('inf'))))
            return chi2_dof < 53.08  # SOTA value
            
        if validator_name == 'PTA Stochastic GW Background Validator':
            lnL = result.get('theory_value', result.get('log_likelihood', float('-inf')))
            return lnL > -0.08  # SOTA value
            
        if validator_name == 'Primordial GWs Validator':
            r = result.get('theory_value', result.get('predicted_value', float('inf')))
            return 0 < r < 0.032  # Below upper limit
            
        # if validator_name == 'Future Detectors Validator':  # Removed - not implemented
        #     snr = result.get('theory_value', result.get('predicted_value', 0))
        #     return snr > 1000  # SOTA value
            
        # if validator_name == 'Novel Signatures Validator':  # Removed - not implemented
        #     bf = result.get('theory_value', result.get('bayes_factor', 0))
        #     return bf > 5  # SOTA value
            
        return False
        
    def _generate_summary(self, theory_results: Dict[str, Any]) -> str:
        """<reason>chain: Generate summary statistics for the theory</reason>"""
        parts = []
        
        # Overall scores
        if 'comprehensive_score' in theory_results:
            comp_score = theory_results['comprehensive_score']
            final_score = comp_score.get('final_score', 0)
            component_scores = comp_score.get('component_scores', {})
            
            parts.append(f"<p><strong>Unified Score:</strong> {final_score:.3f}</p>")
            parts.append(f"<p><strong>Constraint Score:</strong> {component_scores.get('constraints', 0):.3f}</p>")
            parts.append(f"<p><strong>Observational Score:</strong> {component_scores.get('observational', 0):.3f}</p>")
            parts.append(f"<p><strong>Prediction Score:</strong> {component_scores.get('predictions', 0):.3f}</p>")
            parts.append(f"<p><strong>Trajectory Score:</strong> {component_scores.get('trajectory', 0):.3f}</p>")
        elif 'overall_scores' in theory_results:
            # Legacy format fallback
            scores = theory_results['overall_scores']
            parts.append(f"<p><strong>Unified Score:</strong> {scores.get('unified_score', 0):.3f}</p>")
            
        # Validation counts - count actual passed validators
        total_validators = len(self.validation_metadata)
        passed = 0
        beats_sota = 0
        
        for validator_name in self.validation_metadata:
            result = self._get_validator_result(validator_name, theory_results)
            if result and result.get('passed', False):
                passed += 1
            if self._check_beats_sota(validator_name, result):
                beats_sota += 1
        
        parts.append(f"<p><strong>Validators Passed:</strong> {passed}/{total_validators}</p>")
        parts.append(f"<p><strong>Beats SOTA:</strong> {beats_sota}/{total_validators}</p>")
        
        return '\n'.join(parts)
        
    def _generate_visualization_section(self, output_dir: str, theory_results: Dict[str, Any]) -> List[str]:
        """<reason>chain: Generate visualization section with warnings about step counts</reason>"""
        parts = ['    <h2>Visualizations</h2>']
        
        # Check for visualization directory
        viz_dir = os.path.join(output_dir, 'viz')
        if os.path.exists(viz_dir):
            viz_files = [f for f in os.listdir(viz_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if viz_files:
                parts.append('    <div class="viz-container">')
                parts.append('        <div class="viz-grid">')
                
                for viz_file in sorted(viz_files):
                    parts.append('            <div class="viz-item">')
                    parts.append(f'                <h3>{viz_file.replace("_", " ").replace(".png", "").title()}</h3>')
                    parts.append(f'                <img src="viz/{viz_file}" alt="{viz_file}">')
                    parts.append('            </div>')
                
                parts.append('        </div>')
                parts.append('    </div>')
        
        # Add particle information if available
        particle_dir = os.path.join(output_dir, 'particles')
        if os.path.exists(particle_dir):
            parts.append('    <h3>Particle Trajectories</h3>')
            parts.append('    <div class="meta-info">')
            
            particle_files = [f for f in os.listdir(particle_dir) if f.endswith('_info.json')]
            parts.append(f'        <p><strong>Particles Simulated:</strong> {len(particle_files)}</p>')
            
            # Load trajectory info to get step count and cache info
            trajectory_info_path = os.path.join(output_dir, 'theory_info.json')
            if os.path.exists(trajectory_info_path):
                with open(trajectory_info_path, 'r') as f:
                    trajectory_info = json.load(f)
                    trajectory_length = trajectory_info.get('trajectory_length', 0)
                    was_cached = trajectory_info.get('trajectory_was_cached', False)
                    cached_particles = trajectory_info.get('cached_particles', [])
                    
                    # Add warning for low step counts
                    if trajectory_length < 1000:
                        parts.append(f'        <div class="warning"><strong>⚠️ Warning:</strong> Low trajectory steps ({trajectory_length}). Consider re-running with more steps for better accuracy.</div>')
                    else:
                        parts.append(f'        <p><strong>Trajectory Steps:</strong> {trajectory_length}</p>')
                        
                    # Show cache information if trajectories were cached
                    if was_cached:
                        parts.append(f'        <p><strong>📦 Cached Trajectories Used:</strong> {", ".join(cached_particles)}</p>')
            
            # List particles used with cache info
            parts.append('        <ul>')
            for pfile in sorted(particle_files):
                if not pfile.endswith('_info.json'):
                    continue
                particle_name = pfile.replace('_info.json', '')
                
                # Check if cache info exists for this particle
                cache_info_path = os.path.join(particle_dir, f'{particle_name}_cache_info.json')
                if os.path.exists(cache_info_path):
                    with open(cache_info_path, 'r') as f:
                        cache_info = json.load(f)
                    cache_size_mb = cache_info['cache_file_size'] / (1024 * 1024)
                    parts.append(f'            <li>{particle_name.capitalize()} (cached, {cache_size_mb:.1f} MB)</li>')
                else:
                    parts.append(f'            <li>{particle_name.capitalize()}</li>')
            parts.append('        </ul>')
            
            # Add links to cached trajectory files
            cached_files = [f for f in os.listdir(particle_dir) if f.endswith('_cached_source.pt')]
            if cached_files:
                parts.append('        <p><strong>Cached Trajectory Files:</strong></p>')
                parts.append('        <ul>')
                for cached_file in sorted(cached_files):
                    parts.append(f'            <li><a href="particles/{cached_file}">{cached_file}</a></li>')
                parts.append('        </ul>')
            
            parts.append('    </div>')
        
        return parts 

    def _generate_error_section(self, theory_results: Dict[str, Any]) -> str:
        """<reason>chain: Generate a dedicated section for validation exceptions</reason>"""
        error_parts = []
        
        # Check all validations for errors
        if 'validations' in theory_results:
            for val in theory_results['validations']:
                if val.get('flags', {}).get('overall') == 'ERROR':
                    validator_name = val.get('validator', 'Unknown Validator')
                    error_msg = val.get('flags', {}).get('details', 'Unknown error')
                    
                    # Extract additional error details if available
                    error_details = val.get('details', {})
                    error_type = error_details.get('error_type', 'Exception')
                    
                    error_parts.append(f'<div style="margin-bottom: 15px;">')
                    error_parts.append(f'<strong>{html.escape(validator_name)}</strong> - <code>{html.escape(error_type)}</code><br>')
                    error_parts.append(f'<pre style="background: #f5f5f5; padding: 10px; margin-top: 5px; overflow-x: auto; font-size: 12px;">{html.escape(str(error_msg))}</pre>')
                    
                    # Add traceback if available
                    if 'traceback' in error_details:
                        error_parts.append(f'<details style="margin-top: 5px;">')
                        error_parts.append(f'<summary style="cursor: pointer;">Show Traceback</summary>')
                        error_parts.append(f'<pre style="background: #f5f5f5; padding: 10px; margin-top: 5px; overflow-x: auto; font-size: 11px;">{html.escape(error_details["traceback"])}</pre>')
                        error_parts.append(f'</details>')
                    
                    error_parts.append(f'</div>')
        
        # Also check individual category results for errors
        for category in ['constraints', 'observational', 'predictions']:
            if category in theory_results:
                for validator_name, result in theory_results[category].items():
                    if not result.get('passed', True) and 'error' in result.get('details', {}):
                        error_parts.append(f'<div style="margin-bottom: 15px;">')
                        error_parts.append(f'<strong>{html.escape(validator_name)}</strong> (from {category})<br>')
                        error_msg = result['details'].get('error', 'Unknown error')
                        error_parts.append(f'<pre style="background: #f5f5f5; padding: 10px; margin-top: 5px; overflow-x: auto; font-size: 12px;">{html.escape(str(error_msg))}</pre>')
                        error_parts.append(f'</div>')
        
        if error_parts:
            return '\n'.join(error_parts)
        return "" 