"""
LLM API interface for generating new gravitational theories
"""
import os
import requests
from typing import Optional

class LLMApi:
    """Handles LLM API calls for theory generation"""
    
    def __init__(self, provider: str = "grok"):
        self.provider = provider
        self.api_key = os.getenv("GROK_API_KEY", "")
        self.base_url = "https://api.x.ai/v1"
        
        # Load prompt template
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'self_discovery', 'prompt_template.txt'
        )
        
        try:
            with open(template_path, 'r') as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            # Fallback template
            self.prompt_template = """Generate a novel gravitational theory as a Python class inheriting from GravitationalTheory.

The new theory will be benchmarked against the following baseline theories:
{baseline_theories}

Your generated theory must:
- Be implemented as a Python class named 'CustomTheory' that inherits from 'GravitationalTheory'.
- Have a 'get_metric' method that returns the metric tensor components (g_tt, g_rr, g_pp, g_tp).
- Include a Lagrangian formulation of the theory in a docstring. This is a critical validation step.
- Aim to unify gravity and electromagnetism, or explore other novel geometric approaches to gravity.

Initial idea: {initial_prompt}
    
Return ONLY the Python code, no explanations."""
    
    def generate_theory_variation(self, base_theory_code: str, modification_prompt: str, baseline_theories: list[str]) -> Optional[str]:
        """Generate a variation of a theory based on user input"""
        
        # Construct the prompt
        baseline_list = '\n'.join(f"{i+1}. {name}" for i, name in enumerate(baseline_theories))
        
        full_prompt = f"""Given this existing gravitational theory:

```python
{base_theory_code}
```

Please modify it according to this instruction: {modification_prompt}

The modified theory must:
- Be implemented as a Python class named 'CustomTheory' that inherits from 'GravitationalTheory'.
- Have a 'get_metric' method that returns the metric tensor components (g_tt, g_rr, g_pp, g_tp).
- Include a Lagrangian formulation of the theory in a docstring. This is a critical validation step.
- Maintain the core structure while applying the requested modification.

The theory will be benchmarked against:
{baseline_list}

Return ONLY the Python code, no explanations."""
        
        return self._call_api(full_prompt)
    
    def generate_new_theory(self, initial_prompt: str, baseline_theories: list[str]) -> Optional[str]:
        """Generate a completely new theory based on initial prompt"""
        
        baseline_list = '\n'.join(f"{i+1}. {name}" for i, name in enumerate(baseline_theories))
        prompt = self.prompt_template.format(
            baseline_theories=baseline_list,
            initial_prompt=initial_prompt if initial_prompt else 'Explore modifications to the Reissner-Nordström or Dilaton metric.'
        )
        
        return self._call_api(prompt)
    
    def _call_api(self, prompt: str) -> Optional[str]:
        """Make the actual API call to Grok"""
        
        if not self.api_key:
            print("Warning: GROK_API_KEY not set. Using mock response.")
            return self._mock_response()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert theoretical physicist specializing in gravitational theories and general relativity."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "grok-beta",
            "stream": False,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"API call failed: {e}")
            return None
    
    def _mock_response(self) -> str:
        """Return a mock theory for testing when API key is not available"""
        return '''import torch
from physics_agent.base_theory import GravitationalTheory

class CustomTheory(GravitationalTheory):
    """
    Modified Reissner-Nordström metric with quantum corrections
    
    Lagrangian: L = R/(16πG) - F^{μν}F_{μν}/4 + α/r² * log(r/r_p)
    where α is a quantum correction parameter and r_p is the Planck length.
    """
    
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.name = f"Quantum Corrected RN (α={alpha})"
        self.alpha = alpha
        self.is_symmetric = True
        self.category = "quantum"
    
    def get_metric(self, r, M_param, C_param, G_param, Q_param=1e-5, **kwargs):
        """Returns the metric tensor components with quantum corrections"""
        
        rs = 2 * G_param * M_param / C_param**2
        rq = G_param * Q_param**2 / (4 * torch.pi * 8.854e-12 * C_param**4)
        
        # Quantum correction term
        r_planck = 1.616e-35  # Planck length in meters
        quantum_term = self.alpha / r**2 * torch.log(r / r_planck + 1)
        
        # Modified metric components
        f = 1 - rs/r + rq/r**2 + quantum_term
        
        g_tt = -f
        g_rr = 1/f
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp
''' 