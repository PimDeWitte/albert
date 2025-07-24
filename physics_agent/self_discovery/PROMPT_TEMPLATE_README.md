# Self-Discovery Prompt Template Guide

## Overview

The `prompt_template.txt` file contains the template used by the self-discovery system to generate new gravitational theories via LLM APIs. This prompt is customizable to explore different aspects of gravitational physics.

## Example Templates

We provide example templates for specific exploration directions:

- **`prompt_template_example_quantum.txt`**: Example focused on quantum gravity theories with detailed requirements for quantum corrections, unitarity, and effective field theory approaches.

You can copy and modify these examples to create your own specialized templates.

## SymPy Documentation Section

The prompt template now includes comprehensive SymPy syntax documentation. This is crucial because:

1. **Correct Syntax**: LLMs need to know the exact SymPy syntax for mathematical expressions
2. **Available Symbols**: The framework expects certain symbol names (R, F, etc.)
3. **Working Examples**: Real examples from the codebase ensure valid expressions

The SymPy section includes:
- Import statements
- Pre-defined symbols (R for Ricci scalar, F for field strength, etc.)
- Common operations (powers, logs, exponentials, fractions)
- Example Lagrangians from actual theories in the framework

This dramatically improves the success rate of generated theories by ensuring they use valid SymPy expressions that the validation system can process.

## Template Variables

The prompt template supports the following dynamic variables:

- `{baseline_theories}`: Automatically populated with the list of baseline theories for comparison
- `{initial_prompt}`: The initial idea or direction provided via the `--initial-prompt` CLI argument

## Customizing the Prompt

### 1. Edit `prompt_template.txt`

Open the file and modify the instructions to guide the LLM toward specific types of theories:

```
# Example: Focus on quantum gravity
Your generated theory must explore quantum corrections to general relativity...

# Example: Focus on emergent gravity  
Your theory should treat gravity as an emergent phenomenon from microscopic degrees of freedom...
```

### 2. Add Specific Requirements

You can add specific mathematical or physical requirements:

```
Your generated theory must:
- Include a scalar field φ(r) coupled to the metric
- Reduce to Schwarzschild in the φ → 0 limit
- Include a Yukawa-type potential term
- Have well-defined thermodynamic properties
```

### 3. Provide Examples

Include example code structures or patterns:

```
Example structure:
```python
class CustomTheory(GravitationalTheory):
    def __init__(self):
        super().__init__("Your Theory Name")
        self.parameter1 = 1.0  # Description
        self.lagrangian = sympy.sympify("R + alpha * phi**2")
    
    def get_metric(self, r, M, c, G):
        # Your metric implementation
        pass
```

### 4. Specify Validation Criteria

Add criteria the theory should aim to satisfy:

```
The theory should aim to:
- Pass conservation law validations
- Correctly predict Mercury's perihelion precession
- Reduce to Newtonian gravity in the weak-field limit
- Maintain numerical stability across all radial ranges
```

## Examples of Effective Prompts

### Example 1: Exploring Modified Gravity

```
Generate a modified gravity theory that introduces a running gravitational constant G(r).
The theory should interpolate between quantum corrections at small scales and 
classical GR at large scales. Include a characteristic length scale parameter.
```

### Example 2: Electromagnetic Unification

```
Create a theory that geometrically unifies gravity and electromagnetism using
a 5-dimensional Kaluza-Klein approach. The extra dimension should be compactified
with radius related to the electromagnetic coupling.
```

### Example 3: Emergent Gravity

```
Develop an emergent gravity theory where spacetime geometry arises from 
entanglement entropy. The theory should reproduce the Bekenstein-Hawking
entropy formula and include holographic corrections.
```

## Tips for Best Results

1. **Be Specific**: The more specific your requirements, the better the generated theory
2. **Include Physics**: Reference known physics principles the theory should respect
3. **Mathematical Clarity**: Specify the mathematical framework (differential geometry, field theory, etc.)
4. **Validation Focus**: Mention which validations are most important for your exploration
5. **Parameter Ranges**: Suggest reasonable parameter ranges for any new constants

## Command Line Usage

After customizing the prompt template, run:

```bash
python self_discovery.py --self-discover --initial-prompt "Your specific idea here"
```

The system will use your customized template combined with your initial prompt to generate theories.

## Advanced Customization

For more complex scenarios, you can:

1. Create multiple template files for different exploration directions
2. Modify `self_discovery.py` to select templates based on criteria
3. Add conditional logic to the template loading based on CLI arguments

## Note on Formatting

- Keep the template focused and concise
- Use clear formatting with bullet points for requirements
- Include the `{baseline_theories}` and `{initial_prompt}` placeholders where needed
- End with "Return ONLY the Python code, no explanations." to ensure clean output 