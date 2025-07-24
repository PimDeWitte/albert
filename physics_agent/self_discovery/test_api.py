#!/usr/bin/env python3
"""Test script for LLM API functionality"""
import os
from physics_agent.ui.llm_api import LLMApi

def test_api():
    """Test the LLM API with a simple prompt"""
    print("Testing LLM API...")
    
    # Check for API key
    api_key = os.getenv("GROK_API_KEY", "")
    if api_key:
        print(f"✓ GROK_API_KEY is set (length: {len(api_key)})")
    else:
        print("✗ GROK_API_KEY is not set - will use mock response")
    
    # Initialize API
    api = LLMApi(provider="grok")
    
    # Test generation
    baseline_theories = ["Schwarzschild", "Kerr", "Reissner-Nordström"]
    initial_prompt = "Create a simple theory with torsion"
    
    print(f"\nGenerating theory with prompt: {initial_prompt}")
    print("Baseline theories:", baseline_theories)
    
    # Generate theory
    theory_code = api.generate_new_theory(initial_prompt, baseline_theories)
    
    if theory_code:
        print("\n✓ Successfully generated theory code!")
        print(f"Code length: {len(theory_code)} characters")
        print("\nFirst 500 characters:")
        print("-" * 50)
        print(theory_code[:500])
        print("-" * 50)
    else:
        print("\n✗ Failed to generate theory code")

if __name__ == "__main__":
    test_api() 