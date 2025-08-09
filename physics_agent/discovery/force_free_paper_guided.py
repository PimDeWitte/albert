"""
Paper-guided force-free discovery.

Directly constructs the expressions mentioned in Section 2.4 of the paper
and verifies they satisfy the foliation condition.
"""

from __future__ import annotations
from typing import List, Dict, Any
import time
from sympy import symbols, sqrt, exp, log, simplify, S
import json
import os
from datetime import datetime

from .validators import validate_force_free_foliation
from .known_formula_filter import filter_novel_candidates


def get_paper_expressions():
    """
    Get the exact expressions from Section 2.4 of the paper.
    
    Returns list of (expression, expected_name) tuples.
    """
    # Define symbols
    rho = symbols('rho', real=True, positive=True)
    z = symbols('z', real=True)
    b = symbols('b', real=True, positive=True)  # Parameter for hyperbolic
    k = symbols('k', real=True, positive=True)  # Parameter for bent
    
    # The 7 solutions from the paper (equations 2.20 and 2.21)
    expressions = [
        # Vacuum solutions (6)
        (rho**2, "Vertical field (external dipole)"),
        (rho**2 * z, "X-point (external quadrupole)"),
        (1 - z/sqrt(z**2 + rho**2), "Radial"),
        (rho**2 / (z**2 + rho**2)**(S(3)/2), "Dipolar"),
        (sqrt(z**2 + rho**2) - z, "Parabolic"),
        # Hyperbolic with b=1 as example
        ((sqrt(z**2 + (rho - 1)**2) - sqrt(z**2 + (rho + 1)**2))/2, "Hyperbolic"),
        
        # Non-vacuum solution (1)
        (rho**2 * exp(-2*z), "Bent (nonvacuum)"),  # with k=1
    ]
    
    # Also add some trivial solutions that should work
    expressions.extend([
        (rho, "Linear in rho"),
        (z, "Linear in z"),
        (1, "Constant"),
        (0, "Zero"),
    ])
    
    return expressions


def validate_paper_solutions():
    """Validate all solutions from the paper."""
    expressions = get_paper_expressions()
    results = []
    
    print(f"Validating {len(expressions)} expressions from the paper...")
    print("="*60)
    
    for i, (expr, expected_name) in enumerate(expressions):
        print(f"\n{i+1}. Testing: {expected_name}")
        print(f"   u = {expr}")
        
        start = time.time()
        try:
            validation = validate_force_free_foliation(expr)
            elapsed = time.time() - start
            
            if validation['valid']:
                print(f"   ✓ VALID ({elapsed:.2f}s)")
                print(f"   Classification: {validation.get('classification', 'Unknown')}")
                
                results.append({
                    'expression': str(expr),
                    'expected_name': expected_name,
                    'classification': validation.get('classification', 'Unknown'),
                    'valid': True,
                    'validation_time': elapsed
                })
            else:
                print(f"   ✗ INVALID ({elapsed:.2f}s)")
                print(f"   Reason: {validation['details']}")
                
                results.append({
                    'expression': str(expr),
                    'expected_name': expected_name,
                    'valid': False,
                    'reason': validation['details'],
                    'validation_time': elapsed
                })
                
        except Exception as e:
            print(f"   ✗ ERROR: {str(e)}")
            results.append({
                'expression': str(expr),
                'expected_name': expected_name,
                'valid': False,
                'reason': f"Exception: {str(e)}",
                'validation_time': 0
            })
    
    return results


def generate_variations(base_expressions: List[tuple], max_variations: int = 50) -> List[tuple]:
    """
    Generate variations of the base expressions.
    
    This simulates what a discovery algorithm might find.
    """
    rho = symbols('rho', real=True, positive=True)
    z = symbols('z', real=True)
    
    variations = []
    
    # Simple transformations
    for expr, name in base_expressions[:5]:  # Only vary the first few
        # Linear combinations
        variations.append((2*expr, f"2*({name})"))
        variations.append((expr + 1, f"{name} + 1"))
        
        # Powers (be careful with complexity)
        if len(str(expr)) < 20:  # Only for simple expressions
            variations.append((expr**2, f"({name})^2"))
        
        # Combinations with coordinates
        if not expr.has(rho):
            variations.append((expr + rho, f"{name} + rho"))
        if not expr.has(z):
            variations.append((expr + z, f"{name} + z"))
            
        if len(variations) >= max_variations:
            break
    
    return variations


def run_paper_guided_discovery(output_dir: str = None) -> Dict[str, Any]:
    """
    Run paper-guided force-free discovery.
    
    This approach:
    1. Tests the exact solutions from the paper
    2. Generates some variations
    3. Filters known formulas
    4. Produces a report
    """
    print("Paper-guided Force-Free Discovery")
    print("Based on: Force-Free Foliations, Compère et al., Section 2.4")
    print("="*60)
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('runs', f'force_free_paper_guided_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 1: Validate paper solutions
    print("\nPhase 1: Validating paper solutions...")
    paper_results = validate_paper_solutions()
    
    # Phase 2: Generate and test variations
    print("\n\nPhase 2: Testing variations...")
    base_expressions = get_paper_expressions()
    variations = generate_variations(base_expressions)
    
    variation_results = []
    for i, (expr, desc) in enumerate(variations[:20]):  # Limit variations
        if i % 5 == 0:
            print(f"  Testing variations {i+1}-{min(i+5, len(variations))}...")
        
        try:
            validation = validate_force_free_foliation(expr)
            if validation['valid']:
                variation_results.append({
                    'expression': str(expr),
                    'description': desc,
                    'classification': validation.get('classification', 'Unknown'),
                    'valid': True
                })
        except Exception:
            pass
    
    # Phase 3: Filter known formulas
    print("\n\nPhase 3: Filtering known formulas...")
    all_candidates = paper_results + variation_results
    novel_candidates = filter_novel_candidates(
        [c for c in all_candidates if c.get('valid', False)],
        category='force_free_foliations'
    )
    
    print(f"  Total valid: {sum(1 for c in all_candidates if c.get('valid', False))}")
    print(f"  Novel candidates: {len(novel_candidates)}")
    
    # Analyze results
    results = analyze_paper_results(paper_results, variation_results, novel_candidates)
    
    # Save results
    save_paper_results(results, output_dir)
    
    return results


def analyze_paper_results(paper_results, variation_results, novel_candidates):
    """Analyze results from paper-guided discovery."""
    # Check which paper solutions were valid
    paper_valid = [r for r in paper_results if r['valid']]
    paper_invalid = [r for r in paper_results if not r['valid']]
    
    # Expected types
    expected_types = [
        "Vertical field (external dipole)",
        "X-point (external quadrupole)",
        "Radial",
        "Dipolar",
        "Parabolic",
        "Hyperbolic",
        "Bent (nonvacuum)"
    ]
    
    # Check what we found
    found_types = set()
    for r in paper_valid:
        if any(exp_type in r['expected_name'] for exp_type in expected_types):
            found_types.add(r['expected_name'])
    
    return {
        'paper_solutions': {
            'total': len(paper_results),
            'valid': len(paper_valid),
            'invalid': len(paper_invalid),
            'details': paper_results
        },
        'variations': {
            'total': len(variation_results),
            'valid': len([r for r in variation_results if r['valid']]),
            'details': variation_results[:10]  # First 10 only
        },
        'novel_candidates': novel_candidates,
        'expected_vs_found': {
            exp_type: exp_type in found_types
            for exp_type in expected_types
        },
        'summary': {
            'total_valid': len(paper_valid) + len([r for r in variation_results if r['valid']]),
            'paper_solutions_verified': len(paper_valid),
            'expected_types_found': sum(1 for found in found_types if any(exp in found for exp in expected_types)),
            'novel_discoveries': len(novel_candidates)
        }
    }


def save_paper_results(results, output_dir):
    """Save results from paper-guided discovery."""
    # JSON
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # HTML
    html_path = os.path.join(output_dir, 'report.html')
    generate_paper_html_report(results, html_path)
    
    # Summary text
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Force-Free Foliation Discovery Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Paper solutions verified: {results['summary']['paper_solutions_verified']}/{len(results['paper_solutions']['details'])}\n")
        f.write(f"Expected types found: {results['summary']['expected_types_found']}/7\n")
        f.write(f"Total valid foliations: {results['summary']['total_valid']}\n")
        f.write(f"Novel discoveries: {results['summary']['novel_discoveries']}\n\n")
        
        f.write("Expected vs Found:\n")
        for exp_type, found in results['expected_vs_found'].items():
            status = "✓" if found else "✗"
            f.write(f"  {status} {exp_type}\n")
    
    print(f"\nResults saved to:")
    print(f"  {output_dir}/")
    print(f"    - results.json")
    print(f"    - report.html")
    print(f"    - summary.txt")


def generate_paper_html_report(results, output_path):
    """Generate HTML report for paper-guided results."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Paper-Guided Force-Free Discovery</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        .valid {{ background: #e8f5e9; }}
        .invalid {{ background: #ffebee; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Paper-Guided Force-Free Foliation Discovery</h1>
    <p>Based on: Force-Free Foliations, Compère et al., arXiv:1606.06727v2, Section 2.4</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <ul>
            <li>Paper solutions verified: {results['summary']['paper_solutions_verified']}/{len(results['paper_solutions']['details'])}</li>
            <li>Expected types found: {results['summary']['expected_types_found']}/7</li>
            <li>Total valid foliations: {results['summary']['total_valid']}</li>
            <li>Novel discoveries: {results['summary']['novel_discoveries']}</li>
        </ul>
    </div>
    
    <h2>Paper Solutions Verification</h2>
    <table>
        <tr><th>Expression</th><th>Expected</th><th>Valid</th><th>Classification</th></tr>
        {"".join(f'<tr class="{"valid" if r["valid"] else "invalid"}"><td>{r["expression"]}</td><td>{r["expected_name"]}</td><td>{"✓" if r["valid"] else "✗"}</td><td>{r.get("classification", r.get("reason", ""))}</td></tr>' for r in results['paper_solutions']['details'])}
    </table>
    
    <h2>Expected Types Coverage</h2>
    <ul>
        {"".join(f'<li>{"✓" if found else "✗"} {exp_type}</li>' for exp_type, found in results['expected_vs_found'].items())}
    </ul>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
