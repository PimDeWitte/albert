"""
Theory-related utility functions.
<reason>chain: Extract commonly used theory manipulation functions</reason>
"""

def get_preferred_values(theory):
    """
    Get preferred parameter values for a theory instance.
    
    Args:
        theory: Theory instance with potential preferred_params attribute
        
    Returns:
        Dictionary of preferred parameter values
    """
    # <reason>chain: Extract preferred parameter values from theory instance</reason>
    if hasattr(theory, 'preferred_params') and theory.preferred_params:
        return theory.preferred_params.copy()
    
    # <reason>chain: If no preferred params, return empty dict</reason>
    return {}


def extract_theory_name_from_dir(directory_name: str) -> str:
    """
    Extract theory name from directory name, handling parameter suffixes.
    
    Examples:
      "Participatory_QG_ω_0_00" -> "Participatory QG (ω=0.00)"
      "Stochastic_Loss_Conserved_γ_0_50_σ_1_00e-04_a_0_00_Q_0_00" -> "Stochastic Loss Conserved (γ=0.50, σ=1.00e-04, a=0.00, Q=0.00)"
    """
    # First, handle basic underscore-to-space conversion
    parts = directory_name.split('_')
    
    # Find where parameter section starts (look for single Greek letters or parameter names)
    param_chars = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'κ', 'λ', 'μ', 'ν', 'ξ', 'π', 'ρ', 'σ', 'τ', 'φ', 'χ', 'ψ', 'ω', 'a', 'Q', 'k']
    
    theory_parts = []
    param_parts = []
    in_params = False
    
    i = 0
    while i < len(parts):
        part = parts[i]
        
        if part in param_chars and i + 1 < len(parts):
            # This is a parameter
            in_params = True
            param_name = part
            
            # Collect the value parts (handle scientific notation)
            value_parts = []
            i += 1
            while i < len(parts) and parts[i] not in param_chars:
                value_parts.append(parts[i])
                i += 1
            
            # Reconstruct the value
            if value_parts:
                value = '_'.join(value_parts).replace('_', '.')
                # Handle 'e-' notation
                value = value.replace('.e-', 'e-').replace('.e+', 'e+')
                # Clean up any double dots
                value = value.replace('..', '.')
                param_parts.append(f"{param_name}={value}")
            i -= 1  # Back up one since we'll increment at the end of the loop
        else:
            if not in_params:
                theory_parts.append(part)
        
        i += 1
    
    # Construct the final name
    theory_name = ' '.join(theory_parts)
    
    if param_parts:
        param_str = ', '.join(param_parts)
        return f"{theory_name} ({param_str})"
    else:
        return theory_name 