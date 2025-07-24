#!/usr/bin/env python3
"""
Demo script to show UI functionality
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from physics_agent.ui.leaderboard import Leaderboard
from physics_agent.ui.llm_api import LLMApi

def demo_leaderboard():
    """Demo the leaderboard functionality"""
    print("=== Leaderboard Demo ===")
    leaderboard = Leaderboard()
    
    # Get latest run
    latest_run = leaderboard.get_latest_run()
    if latest_run:
        print(f"Latest run found: {latest_run}")
        
        # Generate leaderboard
        board = leaderboard.generate_leaderboard()
        print(f"\nRun timestamp: {board.get('run_timestamp', 'N/A')}")
        print(f"Total theories: {board.get('total_theories', 0)}")
        print(f"Valid theories: {board.get('valid_theories', 0)}")
        
        # Show top 5
        print("\nTop 5 theories by minimum loss:")
        for i, theory in enumerate(board.get('theories', [])[:5]):
            if theory.get('rank'):
                print(f"{theory['rank']}. {theory['name']} - Loss: {theory.get('min_loss', 'N/A')}")
    else:
        print("No runs found. Run theory_engine_core.py first to generate results.")

def demo_llm_api():
    """Demo the LLM API functionality"""
    print("\n=== LLM API Demo ===")
    llm = LLMApi()
    
    # Example base theory code
    base_code = '''import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """Simple test theory"""
    
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.name = f"Test Theory (α={alpha})"
        self.alpha = alpha
        self.is_symmetric = True
        
    def get_metric(self, r, M_param, C_param, G_param, **kwargs):
        rs = 2 * G_param * M_param / C_param**2
        f = 1 - rs/r
        
        g_tt = -f
        g_rr = 1/f
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp
'''
    
    # Show example modifications
    modifications = [
        "make alpha 0.75",
        "add quantum correction term",
        "increase the coupling by factor of 2"
    ]
    
    print(f"Example modifications:")
    for mod in modifications:
        print(f"  - '{mod}'")
    
    # Generate a variation (will use mock if no API key)
    print("\nGenerating variation with: 'make alpha 0.75'")
    result = llm.generate_theory_variation(base_code, "make alpha 0.75", ["Schwarzschild", "Kerr"])
    
    if result:
        print("Generated theory preview:")
        print(result[:200] + "..." if len(result) > 200 else result)
    else:
        print("Failed to generate theory")

if __name__ == "__main__":
    print("Gravity Theory Explorer UI Demo\n")
    
    demo_leaderboard()
    demo_llm_api()
    
    print("\n\nTo start the web UI, run:")
    print("  python -m physics_agent.ui.server")
    print("\nThen open: http://localhost:8000") 