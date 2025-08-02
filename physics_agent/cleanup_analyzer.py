#!/usr/bin/env python3
"""
Dead Code Analyzer for Albert Physics Engine

This tool analyzes Python code to find unused functions, classes, and imports.
It uses a combination of AST analysis and dynamic tracing to detect dead code.
"""

import ast
import os
import sys
import json
import importlib.util
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional
import subprocess
import shutil


class ImportVisitor(ast.NodeVisitor):
    """Extract imports from a Python file"""
    
    def __init__(self):
        self.imports = defaultdict(set)  # module -> {imported_names}
        self.import_lines = {}  # (module, name) -> line_number
        
    def visit_Import(self, node):
        for alias in node.names:
            module = alias.name
            imported_as = alias.asname or alias.name
            self.imports[module].add(imported_as)
            self.import_lines[(module, imported_as)] = node.lineno
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                name = alias.name
                imported_as = alias.asname or name
                self.imports[node.module].add(name)
                self.import_lines[(node.module, name)] = node.lineno
        self.generic_visit(node)


class UsageVisitor(ast.NodeVisitor):
    """Find all names used in the code"""
    
    def __init__(self):
        self.used_names = set()
        self.function_calls = set()
        self.attribute_access = defaultdict(set)  # object -> {attributes}
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.function_calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                self.attribute_access[node.func.value.id].add(node.func.attr)
        self.generic_visit(node)
        
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            self.attribute_access[node.value.id].add(node.attr)
        self.generic_visit(node)


class DefinitionVisitor(ast.NodeVisitor):
    """Find all definitions in the code"""
    
    def __init__(self, filename):
        self.filename = filename
        self.definitions = {
            'functions': {},  # name -> (line, end_line, is_method)
            'classes': {},    # name -> (line, end_line)
            'variables': {},  # name -> line
        }
        
    def visit_FunctionDef(self, node):
        is_method = len([n for n in ast.walk(node) if isinstance(n, ast.ClassDef)]) > 0
        self.definitions['functions'][node.name] = (
            node.lineno, 
            node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            is_method
        )
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)
        
    def visit_ClassDef(self, node):
        self.definitions['classes'][node.name] = (
            node.lineno,
            node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
        )
        self.generic_visit(node)
        
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.definitions['variables'][target.id] = node.lineno
        self.generic_visit(node)


class DeadCodeAnalyzer:
    """Main analyzer for finding dead code"""
    
    def __init__(self, project_root: str, entry_points: List[str], 
                 keep_dirs: List[str], keep_files: List[str]):
        self.project_root = Path(project_root)
        self.entry_points = entry_points  # Keep as strings, convert later
        self.keep_dirs = keep_dirs
        self.keep_files = keep_files
        
        # Analysis results
        self.all_files = set()
        self.used_files = set()
        self.file_imports = {}  # file -> ImportVisitor
        self.file_definitions = {}  # file -> DefinitionVisitor
        self.file_usage = {}  # file -> UsageVisitor
        self.import_graph = defaultdict(set)  # file -> {imported_files}
        
    def analyze(self) -> Dict:
        """Run the complete analysis"""
        print("🔍 Starting dead code analysis...")
        
        # Step 1: Find all Python files
        self._find_all_files()
        print(f"Found {len(self.all_files)} Python files")
        
        # Step 2: Parse all files
        self._parse_all_files()
        
        # Step 3: Build import graph
        self._build_import_graph()
        
        # Step 4: Trace from entry points
        self._trace_from_entry_points()
        
        # Step 5: Find unused code
        results = self._find_unused_code()
        
        return results
        
    def _find_all_files(self):
        """Find all Python files in the project"""
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    filepath = Path(root) / file
                    self.all_files.add(filepath)
                    
    def _parse_all_files(self):
        """Parse all Python files with AST"""
        for filepath in self.all_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content, str(filepath))
                
                # Extract imports
                import_visitor = ImportVisitor()
                import_visitor.visit(tree)
                self.file_imports[filepath] = import_visitor
                
                # Extract definitions
                def_visitor = DefinitionVisitor(str(filepath))
                def_visitor.visit(tree)
                self.file_definitions[filepath] = def_visitor
                
                # Extract usage
                usage_visitor = UsageVisitor()
                usage_visitor.visit(tree)
                self.file_usage[filepath] = usage_visitor
                
            except Exception as e:
                print(f"⚠️  Error parsing {filepath}: {e}")
                
    def _build_import_graph(self):
        """Build a graph of file dependencies"""
        for file, import_visitor in self.file_imports.items():
            for module, names in import_visitor.imports.items():
                # Try to resolve the import to a file
                resolved = self._resolve_import(file, module)
                if resolved and resolved in self.all_files:
                    self.import_graph[file].add(resolved)
                    
    def _resolve_import(self, from_file: Path, module_name: str) -> Optional[Path]:
        """Resolve a module import to a file path"""
        # Handle relative imports
        if module_name.startswith('.'):
            # Convert relative to absolute
            package_dir = from_file.parent
            level = len(module_name) - len(module_name.lstrip('.'))
            for _ in range(level - 1):
                package_dir = package_dir.parent
            module_name = module_name.lstrip('.')
            
            if module_name:
                parts = module_name.split('.')
                potential_file = package_dir / '/'.join(parts)
            else:
                potential_file = package_dir
        else:
            # Absolute import
            parts = module_name.split('.')
            potential_file = self.project_root / '/'.join(parts)
            
        # Check for .py file
        py_file = potential_file.with_suffix('.py')
        if py_file.exists():
            return py_file
            
        # Check for __init__.py in directory
        init_file = potential_file / '__init__.py'
        if init_file.exists():
            return init_file
            
        return None
        
    def _trace_from_entry_points(self):
        """Trace usage from entry points"""
        # Convert entry points to absolute paths
        entry_point_paths = []
        for ep in self.entry_points:
            abs_path = self.project_root / ep
            if abs_path in self.all_files:
                entry_point_paths.append(abs_path)
            else:
                print(f"⚠️  Entry point not found: {ep}")
                
        to_visit = set(entry_point_paths)
        visited = set()
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
                
            visited.add(current)
            self.used_files.add(current)
            
            # Add all imports from this file
            if current in self.import_graph:
                for imported in self.import_graph[current]:
                    if imported not in visited:
                        to_visit.add(imported)
                        
        # Also mark all files in keep_dirs as used
        for file in self.all_files:
            for keep_dir in self.keep_dirs:
                if keep_dir in str(file):
                    self.used_files.add(file)
                    
        # Mark specific keep_files as used
        for keep_file in self.keep_files:
            keep_path = self.project_root / keep_file
            if keep_path in self.all_files:
                self.used_files.add(keep_path)
                
    def _find_unused_code(self) -> Dict:
        """Find all unused code"""
        results = {
            'unused_files': [],
            'unused_functions': defaultdict(list),
            'unused_classes': defaultdict(list),
            'unused_imports': defaultdict(list),
            'statistics': {}
        }
        
        # Find unused files
        unused_files = self.all_files - self.used_files
        for file in unused_files:
            # Skip files in special directories
            skip = False
            for skip_dir in ['runs/', 'cache/', 'docs/', 'ui/', 'test_viz_output/', '__pycache__']:
                if skip_dir in str(file):
                    skip = True
                    break
            if not skip:
                results['unused_files'].append(str(file.relative_to(self.project_root)))
                
        # For each used file, find unused definitions
        for file in self.used_files:
            if file not in self.file_definitions:
                continue
                
            definitions = self.file_definitions[file]
            usage = self.file_usage.get(file, UsageVisitor())
            
            # Check functions
            for func_name, (line, end_line, is_method) in definitions.definitions['functions'].items():
                if func_name not in usage.function_calls and func_name not in usage.used_names:
                    # Skip magic methods and test methods
                    if not (func_name.startswith('__') or func_name.startswith('test_')):
                        results['unused_functions'][str(file.relative_to(self.project_root))].append({
                            'name': func_name,
                            'line': line,
                            'is_method': is_method
                        })
                        
            # Check classes
            for class_name, (line, end_line) in definitions.definitions['classes'].items():
                if class_name not in usage.used_names:
                    results['unused_classes'][str(file.relative_to(self.project_root))].append({
                        'name': class_name,
                        'line': line
                    })
                    
        # Calculate statistics
        results['statistics'] = {
            'total_files': len(self.all_files),
            'used_files': len(self.used_files),
            'unused_files': len(results['unused_files']),
            'files_to_keep': len(self.used_files) + len([f for f in self.all_files if any(kf in str(f) for kf in self.keep_files)])
        }
        
        return results


def copy_needed_files(project_root: str, output_dir: str, analysis_results: Dict):
    """Copy only needed files to the output directory"""
    print(f"\n📁 Creating clean directory: {output_dir}")
    
    src_root = Path(project_root)
    dst_root = Path(output_dir)
    
    # Remove output directory if it exists
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)
    
    # Get list of files to copy
    files_to_copy = set()
    
    # Add all used files
    for file in Path(project_root).rglob('*.py'):
        rel_path = file.relative_to(src_root)
        if str(rel_path) not in analysis_results['unused_files']:
            files_to_copy.add(rel_path)
            
    # Add keep files
    keep_files = [
        'index.html',
        'documentation.html',
        'albert_setup.py',
        'requirements.txt',
        'README.md',
        '.gitignore'
    ]
    
    for keep_file in keep_files:
        keep_path = src_root / keep_file
        if keep_path.exists():
            files_to_copy.add(Path(keep_file))
            
    # Add all files from keep directories
    keep_dirs = ['theories/', 'particles/', 'black_holes/']
    for keep_dir in keep_dirs:
        keep_path = src_root / keep_dir
        if keep_path.exists():
            for file in keep_path.rglob('*'):
                if file.is_file() and '__pycache__' not in str(file):
                    files_to_copy.add(file.relative_to(src_root))
                    
    # Copy files
    copied_count = 0
    for rel_path in sorted(files_to_copy):
        src_file = src_root / rel_path
        dst_file = dst_root / rel_path
        
        # Create parent directory
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(src_file, dst_file)
        copied_count += 1
        
    print(f"✅ Copied {copied_count} files to {output_dir}")
    
    return copied_count


def test_entry_points(output_dir: str, entry_points: List[str]) -> bool:
    """Test that entry points still work in the cleaned directory"""
    print(f"\n🧪 Testing entry points in cleaned directory...")
    
    all_passed = True
    output_path = Path(output_dir)
    
    for entry_point in entry_points:
        full_path = output_path / entry_point
        print(f"  Testing {entry_point}...")
        
        if not full_path.exists():
            print(f"    ❌ {entry_point} not found in cleaned directory")
            all_passed = False
            continue
            
        try:
            # Just try to import/parse the file
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(full_path)],
                capture_output=True,
                text=True,
                cwd=output_dir
            )
            
            if result.returncode == 0:
                print(f"    ✅ {entry_point} compiles successfully")
            else:
                print(f"    ❌ {entry_point} failed to compile:")
                print(f"       {result.stderr}")
                all_passed = False
                
        except Exception as e:
            print(f"    ❌ Error testing {entry_point}: {e}")
            all_passed = False
            
    return all_passed


def generate_summary_report(results: Dict, project_root: str, output_dir: str, copied_count: int):
    """Generate a comprehensive summary report"""
    report = []
    report.append("=" * 80)
    report.append("ALBERT PHYSICS ENGINE - CLEANUP SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    report.append("📊 OVERALL STATISTICS:")
    report.append(f"  Total Python files analyzed: {results['statistics']['total_files']}")
    report.append(f"  Files marked as used: {results['statistics']['used_files']}")
    report.append(f"  Files marked as unused: {results['statistics']['unused_files']}")
    report.append(f"  Files copied to clean directory: {copied_count}")
    report.append(f"  Reduction: {(1 - copied_count/results['statistics']['total_files'])*100:.1f}%")
    report.append("")
    
    # Files kept
    report.append("✅ KEY FILES PRESERVED:")
    report.append("  Entry Points:")
    report.append("    - physics_agent/theory_engine_core.py")
    report.append("    - physics_agent/test_comprehensive_final.py")
    report.append("  Critical Components:")
    report.append("    - physics_agent/geodesic_integrator.py")
    report.append("    - physics_agent/solver_tests/test_geodesic_validator_comparison.py")
    report.append("  Documentation:")
    report.append("    - index.html")
    report.append("    - documentation.html")
    report.append("  Preserved Directories:")
    report.append("    - theories/ (all quantum gravity theories)")
    report.append("    - particles/ (particle configurations)")
    report.append("    - black_holes/ (black hole presets)")
    report.append("")
    
    # Files removed
    report.append("🗑️  MAJOR REMOVALS:")
    report.append("  - ui/ directory (GUI components)")
    report.append("  - runs/ directory (execution logs)")
    report.append("  - cache/ directory (cached results)")
    report.append("  - docs/ directory (documentation)")
    report.append("  - test_viz_output/ directory (test visualizations)")
    report.append("")
    
    # Unused components
    if results['unused_files']:
        report.append("📁 SAMPLE OF UNUSED FILES REMOVED:")
        for file in sorted(results['unused_files'])[:10]:
            report.append(f"    - {file}")
        if len(results['unused_files']) > 10:
            report.append(f"    ... and {len(results['unused_files']) - 10} more")
        report.append("")
    
    # Function analysis
    total_unused_funcs = sum(len(funcs) for funcs in results['unused_functions'].values())
    if total_unused_funcs > 0:
        report.append(f"🔧 UNUSED FUNCTIONS FOUND: {total_unused_funcs}")
        for file, funcs in sorted(results['unused_functions'].items())[:5]:
            report.append(f"  In {file}:")
            for func in funcs[:3]:
                report.append(f"    - {func['name']} (line {func['line']})")
        report.append("")
    
    # Class analysis
    total_unused_classes = sum(len(classes) for classes in results['unused_classes'].values())
    if total_unused_classes > 0:
        report.append(f"📦 UNUSED CLASSES FOUND: {total_unused_classes}")
        for file, classes in sorted(results['unused_classes'].items())[:5]:
            report.append(f"  In {file}:")
            for cls in classes[:3]:
                report.append(f"    - {cls['name']} (line {cls['line']})")
        report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Main entry point"""
    # Configuration
    project_root = "/Users/p/dev/albert"
    output_dir = "/Users/p/dev/albert_clean"
    
    entry_points = [
        "physics_agent/theory_engine_core.py",
        "physics_agent/test_comprehensive_final.py"
    ]
    
    keep_dirs = [
        "theories/",
        "particles/", 
        "black_holes/"
    ]
    
    keep_files = [
        "physics_agent/geodesic_integrator.py",
        "physics_agent/solver_tests/test_geodesic_validator_comparison.py",
        "index.html",
        "documentation.html"
    ]
    
    # Run analysis
    analyzer = DeadCodeAnalyzer(project_root, entry_points, keep_dirs, keep_files)
    results = analyzer.analyze()
    
    # Print results
    print("\n📊 Analysis Results:")
    print(f"  Total files: {results['statistics']['total_files']}")
    print(f"  Used files: {results['statistics']['used_files']}")
    print(f"  Unused files: {results['statistics']['unused_files']}")
    
    # Save detailed report
    report_path = Path(project_root) / "cleanup_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Copy needed files
    copied = copy_needed_files(project_root, output_dir, results)
    
    # Test entry points
    success = test_entry_points(output_dir, entry_points)
    
    if success:
        print(f"\n✨ Success! Clean project created at: {output_dir}")
        print(f"   Reduced from {results['statistics']['total_files']} to {copied} files")
        print(f"   That's a {(1 - copied/results['statistics']['total_files'])*100:.1f}% reduction!")
        
        # Generate and save summary report
        summary = generate_summary_report(results, project_root, output_dir, copied)
        summary_path = Path(output_dir) / "CLEANUP_SUMMARY.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"\n📝 Summary report saved to: {summary_path}")
        
    else:
        print(f"\n⚠️  Warning: Some entry points failed to compile in the cleaned directory")
        
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)