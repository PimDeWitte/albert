"""
Filter for known formulas to avoid redundant calculations.

This module connects to the SQLite database of known theorems and physics formulas,
allowing the discovery system to skip candidates that are already well-understood.
"""

from __future__ import annotations
from typing import List, Dict, Any, Set, Optional
from sympy import Basic, sympify, simplify
import sqlite3
import json
import os
import re


class KnownFormulaDatabase:
    """Database of known physics formulas using SQLite backend."""
    
    def __init__(self, database_path: str = None):
        """
        Initialize the database connection.
        
        Args:
            database_path: Path to SQLite database file. If None, uses theorems.db.
        """
        if database_path is None:
            # Look for theorems.db in project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            database_path = os.path.join(project_root, 'theorems.db')
            
            # If not found, check current directory
            if not os.path.exists(database_path):
                database_path = 'theorems.db'
        
        self.database_path = database_path
        self._expression_cache = {}
        
        # Physics-specific formulas (not in theorems.db)
        self.physics_formulas = {
            "lagrangians": [
                {
                    "name": "Newtonian kinetic energy",
                    "expression": "m*v**2/2",
                    "variables": ["m", "v"],
                    "category": "classical"
                },
                {
                    "name": "Harmonic oscillator",
                    "expression": "m*v**2/2 - k*x**2/2",
                    "variables": ["m", "v", "k", "x"],
                    "category": "classical"
                },
                {
                    "name": "Free particle relativistic",
                    "expression": "-m*c*sqrt(c**2 - v**2)",
                    "variables": ["m", "c", "v"],
                    "category": "relativistic"
                }
            ],
            "force_free_foliations": [
                {
                    "name": "Vertical field",
                    "expression": "rho**2",
                    "variables": ["rho"],
                    "category": "vacuum"
                },
                {
                    "name": "X-point",
                    "expression": "rho**2*z",
                    "variables": ["rho", "z"],
                    "category": "vacuum"
                },
                {
                    "name": "Radial",
                    "expression": "1 - z/sqrt(z**2 + rho**2)",
                    "variables": ["rho", "z"],
                    "category": "vacuum"
                },
                {
                    "name": "Dipolar",
                    "expression": "rho**2/(z**2 + rho**2)**(3/2)",
                    "variables": ["rho", "z"],
                    "category": "vacuum"
                },
                {
                    "name": "Parabolic",
                    "expression": "sqrt(z**2 + rho**2) - z",
                    "variables": ["rho", "z"],
                    "category": "vacuum"
                },
                {
                    "name": "Bent",
                    "expression": "rho**2*exp(-2*k*z)",
                    "variables": ["rho", "z", "k"],
                    "category": "nonvacuum"
                }
            ],
            "metrics": [
                {
                    "name": "Schwarzschild",
                    "expression": "1 - 2*M/r",
                    "variables": ["M", "r"],
                    "category": "black_hole"
                },
                {
                    "name": "Kerr (simplified)",
                    "expression": "1 - 2*M*r/(r**2 + a**2*cos(theta)**2)",
                    "variables": ["M", "r", "a", "theta"],
                    "category": "black_hole"
                }
            ]
        }
        
    def _get_connection(self) -> Optional[sqlite3.Connection]:
        """Get database connection if theorems.db exists."""
        if os.path.exists(self.database_path):
            return sqlite3.connect(self.database_path)
        return None
        
    def _search_theorems_db(self, expr: Basic, category: str = None) -> List[Dict[str, Any]]:
        """Search theorems.db for matching formulas."""
        matches = []
        conn = self._get_connection()
        
        if not conn:
            return matches
            
        try:
            cursor = conn.cursor()
            
            # Search by formula content if present
            query = """
                SELECT name, year, description, formula, category, source 
                FROM theorems 
                WHERE formula IS NOT NULL
            """
            
            if category:
                query += f" AND category = '{category}'"
                
            cursor.execute(query)
            
            for row in cursor.fetchall():
                name, year, desc, formula_text, cat, source = row
                if formula_text:
                    try:
                        # Check if the formula text contains our expression
                        if self._formula_contains_expr(formula_text, expr):
                            matches.append({
                                'name': name,
                                'category': cat,
                                'source': source,
                                'year': year,
                                'exact_match': False
                            })
                    except:
                        pass
                        
            # Also search descriptions for mathematical expressions
            query = """
                SELECT name, year, description, category, source 
                FROM theorems 
                WHERE description LIKE '%=%' OR description LIKE '%theorem%'
            """
            
            if category:
                query += f" AND category = '{category}'"
                
            cursor.execute(query)
            
            expr_str = str(expr)
            for row in cursor.fetchall():
                name, year, desc, cat, source = row
                # Look for mathematical expressions in description
                if self._description_matches_expr(desc, expr_str):
                    matches.append({
                        'name': name,
                        'category': cat,
                        'source': source,
                        'year': year,
                        'exact_match': False
                    })
                    
        except Exception as e:
            print(f"Error searching theorems.db: {e}")
        finally:
            conn.close()
            
        return matches
        
    def _formula_contains_expr(self, formula_text: str, expr: Basic) -> bool:
        """Check if formula text contains the expression."""
        try:
            # Extract mathematical expressions from text
            math_patterns = [
                r'[A-Za-z_]\w*\s*=\s*[^,;.]+',  # variable = expression
                r'\$[^$]+\$',  # LaTeX inline
                r'\\\[[^\]]+\\\]',  # LaTeX display
            ]
            
            for pattern in math_patterns:
                for match in re.finditer(pattern, formula_text):
                    try:
                        # Clean up the match
                        math_expr = match.group()
                        math_expr = re.sub(r'[\$\\]', '', math_expr)
                        math_expr = math_expr.split('=')[-1].strip()
                        
                        # Try to parse and compare
                        parsed = sympify(math_expr)
                        if self._expressions_equivalent(expr, parsed):
                            return True
                    except:
                        pass
                        
        except:
            pass
            
        return False
        
    def _description_matches_expr(self, desc: str, expr_str: str) -> bool:
        """Check if description contains expression-like content."""
        # Simple heuristic matching
        expr_parts = re.findall(r'\w+', expr_str)
        matches = 0
        
        for part in expr_parts:
            if len(part) > 2 and part in desc:
                matches += 1
                
        # If many parts match, likely related
        return matches >= len(expr_parts) * 0.5
    
    def is_known(self, expr: Basic, category: str = None) -> Dict[str, Any]:
        """
        Check if an expression matches a known formula.
        
        Args:
            expr: The expression to check
            category: Optional category to limit search
            
        Returns:
            Dict with 'is_known' bool and 'matches' list
        """
        expr_str = str(expr)
        
        # Check cache first
        cache_key = f"{category}:{expr_str}"
        if cache_key in self._expression_cache:
            return self._expression_cache[cache_key]
        
        matches = []
        categories = [category] if category else self.formulas.keys()
        
        for cat in categories:
            if cat not in self.formulas:
                continue
                
            for formula in self.formulas[cat]:
                try:
                    # Parse known formula
                    known_expr = sympify(formula['expression'])
                    
                    # Check for structural similarity
                    if self._expressions_equivalent(expr, known_expr):
                        matches.append({
                            'name': formula['name'],
                            'category': formula['category'],
                            'exact_match': True
                        })
                    elif self._expressions_similar(expr, known_expr):
                        matches.append({
                            'name': formula['name'],
                            'category': formula['category'],
                            'exact_match': False
                        })
                        
                except Exception:
                    continue
        
        result = {
            'is_known': len(matches) > 0,
            'matches': matches
        }
        
        # Cache result
        self._expression_cache[cache_key] = result
        
        return result
    
    def _expressions_equivalent(self, expr1: Basic, expr2: Basic) -> bool:
        """Check if two expressions are mathematically equivalent."""
        try:
            # Direct comparison
            if expr1 == expr2:
                return True
            
            # Simplified comparison
            if simplify(expr1 - expr2) == 0:
                return True
                
            # Normalized comparison (divide by a common factor)
            if expr1.free_symbols == expr2.free_symbols:
                ratio = simplify(expr1 / expr2)
                if ratio.is_constant():
                    return True
                    
        except Exception:
            pass
            
        return False
    
    def _expressions_similar(self, expr1: Basic, expr2: Basic) -> bool:
        """Check if expressions have similar structure."""
        try:
            # Check if they have the same operations
            ops1 = set(str(op) for op in expr1.atoms(type=type))
            ops2 = set(str(op) for op in expr2.atoms(type=type))
            
            if ops1 == ops2:
                # Check if variable count is similar
                if abs(len(expr1.free_symbols) - len(expr2.free_symbols)) <= 1:
                    return True
                    
        except Exception:
            pass
            
        return False
    
    def filter_candidates(self, candidates: List[Dict[str, Any]], 
                         category: str = None) -> List[Dict[str, Any]]:
        """
        Filter out known formulas from a list of candidates.
        
        Args:
            candidates: List of candidate dicts with 'expression' key
            category: Optional category to check against
            
        Returns:
            List of novel candidates
        """
        novel_candidates = []
        
        for candidate in candidates:
            try:
                expr = sympify(candidate['expression'])
                check_result = self.is_known(expr, category)
                
                if not check_result['is_known']:
                    novel_candidates.append(candidate)
                else:
                    # Add info about why it was filtered
                    candidate['filtered_reason'] = f"Known formula: {check_result['matches'][0]['name']}"
                    
            except Exception:
                # If we can't parse it, include it to be safe
                novel_candidates.append(candidate)
        
        return novel_candidates
    
    def add_formula(self, name: str, expression: str, variables: List[str],
                   category: str, formula_type: str = "general"):
        """Add a new formula to the database."""
        if formula_type not in self.formulas:
            self.formulas[formula_type] = []
            
        self.formulas[formula_type].append({
            "name": name,
            "expression": expression,
            "variables": variables,
            "category": category
        })
        
        # Clear cache
        self._expression_cache.clear()
        
        # Save to disk
        self.save_database()
    
    def save_database(self):
        """Save the database to disk."""
        with open(self.database_path, 'w') as f:
            json.dump(self.formulas, f, indent=2)


# Global instance
_default_db = None


def get_known_formula_db() -> KnownFormulaDatabase:
    """Get the default known formula database."""
    global _default_db
    if _default_db is None:
        _default_db = KnownFormulaDatabase()
    return _default_db


def filter_novel_candidates(candidates: List[Dict[str, Any]], 
                          category: str = None) -> List[Dict[str, Any]]:
    """
    Convenience function to filter candidates using the default database.
    
    Args:
        candidates: List of candidates with 'expression' key
        category: Optional category (e.g., 'force_free_foliations', 'lagrangians')
        
    Returns:
        List of novel candidates not in the known database
    """
    db = get_known_formula_db()
    return db.filter_candidates(candidates, category)
