"""
Systematic force-free foliation discovery.

Instead of using MCTS/neural networks, systematically generate expressions
from primitives to reproduce the paper's results.
"""

from __future__ import annotations
from typing import List, Dict, Any, Set
import time
import itertools
from sympy import symbols, sqrt, exp, log, simplify, Basic, S
from sympy.core.numbers import Zero, One
import json
import os
from datetime import datetime

from .validators import validate_force_free_foliation


def generate_expressions_systematically(
    max_depth: int = 3,
    timeout_seconds: int = 300,
    progress_callback=None
) -> List[Basic]:
    """
    Generate expressions systematically from primitives.
    
    Based on Section 2.4 of the Force-Free Foliations paper.
    """
    # Define symbols
    rho = symbols('rho', real=True, positive=True)
    z = symbols('z', real=True)
    
    # Initial building blocks (depth 1) from the paper
    depth_1 = [
        rho,
        z, 
        rho**2 + z**2,
        rho/z,
    ]
    
    # Binary operations from the paper
    def geometric_sum(x, y):
        """Geometric sum: sqrt((x-1)^2 + y^2) + sqrt((x+1)^2 + y^2)"""
        return sqrt((x - 1)**2 + y**2) + sqrt((x + 1)**2 + y**2)
    
    binary_ops = [
        ('add', lambda x, y: x + y),
        ('sub', lambda x, y: x - y),
        ('mul', lambda x, y: x * y),
        ('div', lambda x, y: x / y),
        ('geom', geometric_sum),
        ('exp_mul', lambda x, y: x * exp(y)),
        ('log_mul', lambda x, y: x * log(y)),
    ]
    
    # Store expressions by depth
    expressions_by_depth = {
        1: depth_1.copy()
    }
    
    # Track unique expressions
    seen_expressions = set()
    all_expressions = []
    
    for expr in depth_1:
        str_expr = str(expr)
        if str_expr not in seen_expressions:
            seen_expressions.add(str_expr)
            all_expressions.append(expr)
    
    start_time = time.time()
    
    # Generate expressions at each depth
    for depth in range(2, max_depth + 1):
        if time.time() - start_time > timeout_seconds:
            print(f"Timeout reached at depth {depth}")
            break
            
        depth_expressions = []
        
        # Combine expressions from lower depths
        for p in range(1, depth):
            q = depth - p
            
            if p not in expressions_by_depth or q not in expressions_by_depth:
                continue
                
            for expr1 in expressions_by_depth[p]:
                for expr2 in expressions_by_depth[q]:
                    # Try each binary operation
                    for op_name, op_func in binary_ops:
                        try:
                            new_expr = op_func(expr1, expr2)
                            
                            # Skip if too complex
                            str_expr = str(new_expr)
                            if len(str_expr) > 200:
                                continue
                                
                            # Simplify
                            try:
                                new_expr = new_expr.simplify()
                            except Exception:
                                pass
                            
                            # Check if unique
                            str_expr = str(new_expr)
                            if str_expr not in seen_expressions:
                                seen_expressions.add(str_expr)
                                depth_expressions.append(new_expr)
                                all_expressions.append(new_expr)
                                
                        except Exception:
                            # Skip problematic operations
                            continue
        
        expressions_by_depth[depth] = depth_expressions
        
        if progress_callback:
            progress_callback(depth, len(all_expressions))
    
    # Add some special cases that might be missed
    special_cases = [
        1,  # constant
        0,  # constant
        rho**2,  # vertical field
        rho**2 * z,  # X-point  
        1 - z/sqrt(z**2 + rho**2),  # radial
        rho**2 / (z**2 + rho**2)**(S(3)/2),  # dipolar
        sqrt(z**2 + rho**2) - z,  # parabolic
        rho**2 * exp(-2*z),  # bent
    ]
    
    for expr in special_cases:
        str_expr = str(expr)
        if str_expr not in seen_expressions:
            seen_expressions.add(str_expr)
            all_expressions.append(expr)
    
    return all_expressions


def run_systematic_force_free_discovery(
    max_depth: int = 4,
    timeout_seconds: int = 300,
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    Run systematic force-free discovery.
    """
    print(f"Starting systematic force-free discovery (max_depth={max_depth})")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('runs', f'force_free_systematic_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate expressions
    print("\nPhase 1: Generating expressions systematically...")
    start_time = time.time()
    
    def progress_callback(depth, total):
        print(f"  Depth {depth}: {total} total expressions")
    
    expressions = generate_expressions_systematically(
        max_depth=max_depth,
        timeout_seconds=timeout_seconds // 2,  # Half time for generation
        progress_callback=progress_callback
    )
    
    gen_time = time.time() - start_time
    print(f"\nGenerated {len(expressions)} expressions in {gen_time:.1f}s")
    
    # Validate expressions
    print("\nPhase 2: Validating force-free condition...")
    valid_solutions = []
    
    for i, expr in enumerate(expressions):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time - gen_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(expressions) - i) / rate if rate > 0 else 0
            print(f"  Validated {i+1}/{len(expressions)} ({i/len(expressions)*100:.1f}%) - ETA: {eta:.1f}s")
        
        try:
            result = validate_force_free_foliation(expr)
            if result['valid']:
                valid_solutions.append({
                    'expression': str(expr),
                    'classification': result.get('classification', 'Unknown'),
                    'latex': _to_latex(expr)
                })
                print(f"    Found: {result['classification']} - u = {expr}")
                
        except Exception as e:
            # Skip problematic expressions
            continue
    
    # Analyze results
    results = analyze_results(valid_solutions)
    
    # Save results
    save_results(results, output_dir)
    
    return results


def analyze_results(solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze and categorize solutions."""
    # Expected solutions from paper
    expected = {
        'vacuum': [
            "Vertical field (external dipole)",
            "X-point (external quadrupole)",
            "Radial",
            "Dipolar",
            "Parabolic",
            "Hyperbolic"
        ],
        'nonvacuum': [
            "Bent (nonvacuum)"
        ]
    }
    
    # Categorize found solutions
    by_type = {}
    for sol in solutions:
        classification = sol['classification']
        if classification not in by_type:
            by_type[classification] = []
        by_type[classification].append(sol)
    
    # Check what we found
    found_expected = {}
    for category, expected_types in expected.items():
        found_expected[category] = []
        for exp_type in expected_types:
            found = any(exp_type in classification for classification in by_type.keys())
            found_expected[category].append({
                'type': exp_type,
                'found': found
            })
    
    # Count vacuum vs nonvacuum
    vacuum_count = sum(len(sols) for cls, sols in by_type.items() 
                      if 'nonvacuum' not in cls.lower())
    nonvacuum_count = sum(len(sols) for cls, sols in by_type.items() 
                         if 'nonvacuum' in cls.lower())
    
    return {
        'total_valid': len(solutions),
        'vacuum_count': vacuum_count,
        'nonvacuum_count': nonvacuum_count,
        'by_type': by_type,
        'expected_vs_found': found_expected,
        'solutions': solutions
    }


def save_results(results: Dict[str, Any], output_dir: str):
    """Save results to JSON and HTML."""
    # Save JSON
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate HTML report
    html_path = os.path.join(output_dir, 'report.html')
    generate_html_report(results, html_path)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")


def generate_html_report(results: Dict[str, Any], output_path: str):
    """Generate HTML report."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Force-Free Foliation Discovery Results</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .solution {{ border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .found {{ background: #e8f5e9; }}
        .missing {{ background: #ffebee; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Systematic Force-Free Foliation Discovery</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Reference: Force-Free Foliations, Compère et al., arXiv:1606.06727v2, Section 2.4</p>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total valid foliations</td><td>{results['total_valid']}</td></tr>
            <tr><td>Vacuum solutions</td><td>{results['vacuum_count']}</td></tr>
            <tr><td>Non-vacuum solutions</td><td>{results['nonvacuum_count']}</td></tr>
        </table>
    </div>
    
    <h2>Expected vs Found</h2>
    <h3>Vacuum Solutions (6 expected):</h3>
    <ul>
    {"".join(f'<li class="{"found" if item["found"] else "missing"}">{item["type"]} - {"✓ Found" if item["found"] else "✗ Missing"}</li>' for item in results['expected_vs_found']['vacuum'])}
    </ul>
    
    <h3>Non-vacuum Solutions (1 expected):</h3>
    <ul>
    {"".join(f'<li class="{"found" if item["found"] else "missing"}">{item["type"]} - {"✓ Found" if item["found"] else "✗ Missing"}</li>' for item in results['expected_vs_found']['nonvacuum'])}
    </ul>
    
    <h2>All Valid Solutions</h2>
    {"".join(f'<div class="solution"><h3>{cls}</h3>{"".join(f"<p>u = {sol['expression']}</p>" for sol in sols)}</div>' for cls, sols in results['by_type'].items())}
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)


def _to_latex(expr) -> str:
    """Convert to LaTeX."""
    try:
        from sympy import latex
        return latex(expr)
    except Exception:
        return str(expr)
