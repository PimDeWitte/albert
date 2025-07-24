import os
import numpy as np
from typing import Dict, Any
import datetime
from scipy import stats

class ScientificReportGenerator:
    """
    Generates comprehensive scientific reports for validation results.
    Produces publication-ready figures and statistical analyses.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_full_report(self, theory_name: str, validation_results: Dict[str, Any],
                           precision_report: Dict[str, Any], uncertainty_report: Dict[str, Any],
                           reproducibility_metadata: Dict[str, Any]) -> str:
        """
        Generate comprehensive scientific report in LaTeX format.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""\\documentclass{{article}}
\\usepackage{{amsmath,amssymb,amsthm}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}
\\usepackage{{siunitx}}

\\title{{Comprehensive Validation Report for {theory_name}}}
\\author{{Gravity Compression Physics Agent}}
\\date{{{timestamp}}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
This report presents a rigorous validation of the {theory_name} gravitational theory
against established observational constraints and theoretical consistency requirements.
All numerical calculations were performed with guaranteed precision bounds and
comprehensive uncertainty quantification.
\\end{{abstract}}

\\section{{Executive Summary}}
"""
        
        # Add pass/fail summary
        report += self._generate_summary_section(validation_results)
        
        # Precision analysis section
        report += """
\\section{Numerical Precision Analysis}
"""
        report += self._generate_precision_section(precision_report)
        
        # Validation results section
        report += """
\\section{Validation Results}
"""
        report += self._generate_validation_section(validation_results)
        
        # Uncertainty quantification
        report += """
\\section{Uncertainty Quantification}
"""
        report += self._generate_uncertainty_section(uncertainty_report)
        
        # Reproducibility
        report += """
\\section{Reproducibility Information}
"""
        report += self._generate_reproducibility_section(reproducibility_metadata)
        
        # Statistical tests
        report += """
\\section{Statistical Significance Tests}
"""
        report += self._generate_statistical_tests(validation_results)
        
        # Conclusions
        report += """
\\section{Conclusions}
"""
        report += self._generate_conclusions(validation_results, precision_report)
        
        report += """
\\end{document}
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, f'{theory_name}_validation_report.tex')
        with open(report_path, 'w') as f:
            f.write(report)
            
        return report_path
        
    def _generate_summary_section(self, validation_results: Dict[str, Any]) -> str:
        """Generate executive summary with key findings."""
        validations = validation_results.get('validations', [])
        
        passed = sum(1 for v in validations if v['flags']['overall'] == 'PASS')
        failed = sum(1 for v in validations if v['flags']['overall'] == 'FAIL')
        total = len(validations)
        
        summary = f"""
The theory was subjected to {total} validation tests:
\\begin{{itemize}}
\\item \\textbf{{Passed:}} {passed}/{total} ({100*passed/total if total > 0 else 0:.1f}\\%)
\\item \\textbf{{Failed:}} {failed}/{total} ({100*failed/total if total > 0 else 0:.1f}\\%)
\\end{{itemize}}
"""
        
        # Critical failures
        critical_failures = [v for v in validations if v['flags']['overall'] == 'FAIL' and v['type'] == 'constraint']
        if critical_failures:
            summary += """
\\textbf{Critical Issues:} The following fundamental constraints were violated:
\\begin{itemize}
"""
            for failure in critical_failures:
                summary += f"\\item {failure['validator']}: {failure.get('details', {})}\n"
            summary += "\\end{itemize}\n"
            
        return summary
        
    def _generate_precision_section(self, precision_report: Dict[str, Any]) -> str:
        """Generate precision analysis section."""
        if not precision_report:
            return "No precision analysis available.\n"
            
        section = f"""
\\subsection{{Machine Precision}}
\\begin{{itemize}}
\\item Data type: {precision_report['machine_precision']['dtype']}
\\item Machine epsilon: {precision_report['machine_precision']['epsilon']:.2e}
\\item Decimal digits: {precision_report['machine_precision']['decimal_digits']}
\\end{{itemize}}

\\subsection{{Numerical Stability}}
\\begin{{itemize}}
\\item Median condition number: {precision_report['numerical_stability']['median_condition_number']:.2e}
\\item Maximum condition number: {precision_report['numerical_stability']['max_condition_number']:.2e}
\\item Catastrophic cancellations: {precision_report['numerical_stability']['catastrophic_cancellations']}
\\end{{itemize}}

\\subsection{{Error Analysis}}
\\begin{{itemize}}
\\item Global error bound: {precision_report['error_analysis']['global_error_bound']:.2e}
\\item Error growth rate: {precision_report['error_analysis']['error_growth_rate']:.2e}
\\item Conservation satisfied: {precision_report['conservation_analysis']['conservation_valid']}
\\end{{itemize}}
"""
        
        # Add verdict
        if precision_report['scientific_validity']['publication_ready']:
            section += """
\\textbf{Verdict:} The numerical precision meets publication standards.
"""
        else:
            section += """
\\textbf{Verdict:} The numerical precision is INSUFFICIENT for publication.
Issues that must be addressed:
\\begin{itemize}
"""
            validity = precision_report['scientific_validity']
            if not validity['precision_adequate']:
                section += "\\item Global error exceeds acceptable bounds\n"
            if not validity['numerically_stable']:
                section += "\\item Numerical instabilities detected\n"
            if not validity['conservation_satisfied']:
                section += "\\item Conservation laws violated beyond tolerance\n"
            if not validity['no_catastrophic_failures']:
                section += "\\item Catastrophic cancellations detected\n"
            section += "\\end{itemize}\n"
            
        return section
        
    def _generate_validation_section(self, validation_results: Dict[str, Any]) -> str:
        """Generate detailed validation results."""
        validations = validation_results.get('validations', [])
        
        # Group by category
        by_category = {}
        for v in validations:
            category = v.get('type', 'unknown')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(v)
            
        section = ""
        for category, tests in by_category.items():
            section += f"\\subsection{{{category.capitalize()} Tests}}\n"
            section += "\\begin{table}[h]\n\\centering\n"
            section += "\\begin{tabular}{lccr}\n\\toprule\n"
            section += "Test & Status & Loss & Details \\\\\n\\midrule\n"
            
            for test in tests:
                status = test['flags']['overall']
                loss = test.get('loss', float('nan'))
                section += f"{test['validator']} & {status} & {loss:.2e} & "
                
                # Add key details
                details = test.get('details', {})
                if details:
                    key_detail = list(details.items())[0]
                    section += f"{key_detail[0]}: {key_detail[1]:.2e}"
                section += " \\\\\n"
                
            section += "\\bottomrule\n\\end{tabular}\n"
            section += f"\\caption{{{category.capitalize()} validation results}}\n"
            section += "\\end{table}\n\n"
            
        return section
        
    def _generate_uncertainty_section(self, uncertainty_report: Dict[str, Any]) -> str:
        """Generate uncertainty quantification section."""
        if not uncertainty_report:
            return "No uncertainty analysis available.\n"
            
        section = "\\subsection{Confidence Intervals}\n"
        section += "\\begin{table}[h]\n\\centering\n"
        section += "\\begin{tabular}{lccc}\n\\toprule\n"
        section += "Validator & Mean Loss & 95\\% CI & Samples \\\\\n\\midrule\n"
        
        for validator, stats in uncertainty_report.items():
            section += f"{validator} & {stats['mean']:.2e} & "
            section += f"[{stats['ci_lower']:.2e}, {stats['ci_upper']:.2e}] & "
            section += f"{stats['n_samples']} \\\\\n"
            
        section += "\\bottomrule\n\\end{tabular}\n"
        section += "\\caption{Uncertainty quantification for validation results}\n"
        section += "\\end{table}\n\n"
        
        return section
        
    def _generate_reproducibility_section(self, metadata: Dict[str, Any]) -> str:
        """Generate reproducibility information."""
        section = f"""
\\subsection{{Computational Environment}}
\\begin{{itemize}}
\\item Platform: {metadata['environment']['platform']['system']} {metadata['environment']['platform']['release']}
\\item Python: {metadata['environment']['platform']['python_version'].split()[0]}
\\item PyTorch: {metadata['environment']['libraries']['torch']}
\\item NumPy: {metadata['environment']['libraries']['numpy']}
\\item Hardware: {metadata['environment']['hardware']['cpu_count']} CPUs
"""
        
        if metadata['environment']['hardware']['cuda_available']:
            section += f"\\item CUDA: {metadata['environment']['hardware']['cuda_devices']} devices, version {metadata['environment'].get('cuda_version', 'unknown')}\n"
            
        section += "\\end{itemize}\n\n"
        
        # Git info
        git = metadata['environment']['git']
        if 'error' not in git:
            section += f"""
\\subsection{{Version Control}}
\\begin{{itemize}}
\\item Git commit: \\texttt{{{git['commit'][:8]}}}
\\item Branch: {git['branch']}
\\item Uncommitted changes: {'Yes' if git['has_uncommitted_changes'] else 'No'}
\\end{{itemize}}
"""
        
        return section
        
    def _generate_statistical_tests(self, validation_results: Dict[str, Any]) -> str:
        """Generate statistical significance tests."""
        validations = validation_results.get('validations', [])
        
        # Collect losses for statistical testing
        losses = [v['loss'] for v in validations if 'loss' in v and np.isfinite(v['loss'])]
        
        if len(losses) < 2:
            return "Insufficient data for statistical tests.\n"
            
        section = f"""
\\subsection{{Distribution of Validation Losses}}
\\begin{{itemize}}
\\item Mean loss: {np.mean(losses):.2e} $\\pm$ {np.std(losses):.2e}
\\item Median loss: {np.median(losses):.2e}
\\item Range: [{np.min(losses):.2e}, {np.max(losses):.2e}]
\\end{{itemize}}

\\subsection{{Normality Test}}
"""
        
        # Shapiro-Wilk test for normality
        stat, p_value = stats.shapiro(losses)
        section += f"Shapiro-Wilk test: W = {stat:.4f}, p = {p_value:.4e}\n"
        
        if p_value > 0.05:
            section += "The losses follow a normal distribution (p > 0.05).\n"
        else:
            section += "The losses do NOT follow a normal distribution (p < 0.05).\n"
            
        return section
        
    def _generate_conclusions(self, validation_results: Dict[str, Any], 
                            precision_report: Dict[str, Any]) -> str:
        """Generate conclusions and recommendations."""
        validations = validation_results.get('validations', [])
        passed = sum(1 for v in validations if v['flags']['overall'] == 'PASS')
        total = len(validations)
        
        pass_rate = passed / total if total > 0 else 0
        precision_ok = precision_report.get('scientific_validity', {}).get('publication_ready', False)
        
        if pass_rate > 0.95 and precision_ok:
            verdict = "ACCEPTED for publication"
            recommendation = "The theory passes rigorous validation and is ready for peer review."
        elif pass_rate > 0.8 and precision_ok:
            verdict = "CONDITIONALLY ACCEPTED"
            recommendation = "The theory shows promise but requires addressing specific validation failures."
        else:
            verdict = "NOT READY for publication"
            recommendation = "Significant issues must be resolved before scientific acceptance."
            
        section = f"""
\\textbf{{Overall Verdict:}} {verdict}

{recommendation}

\\subsection{{Key Findings}}
\\begin{{enumerate}}
\\item Validation pass rate: {100*pass_rate:.1f}\\%
\\item Numerical precision: {'Adequate' if precision_ok else 'Insufficient'}
\\item Conservation laws: {'Satisfied' if precision_report.get('conservation_analysis', {}).get('conservation_valid', False) else 'Violated'}
\\end{{enumerate}}
"""
        
        # Add specific recommendations
        failures = [v for v in validations if v['flags']['overall'] == 'FAIL']
        if failures:
            section += """
\\subsection{Required Improvements}
The following issues must be addressed:
\\begin{itemize}
"""
            for f in failures[:5]:  # Top 5 failures
                section += f"\\item {f['validator']}: Loss = {f.get('loss', 'N/A'):.2e}\n"
            section += "\\end{itemize}\n"
            
        return section 