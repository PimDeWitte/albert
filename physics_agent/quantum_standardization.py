"""
Quantum Theory Parameter Standardization

<reason>chain: Ensure fair comparison of quantum theories with different parameter scales</reason>
"""

# Standard quantum strength mappings for fair comparison
QUANTUM_PARAMETER_MAPPINGS = {
    # Theory class name -> (param_name, mapping_function)
    'QuantumCorrected': ('alpha', lambda q: 0.1 * q),  # Map to [0, 0.1] for moderate effects
    'LogCorrected': ('gamma', lambda q: 0.01 * q),  # Map to [0, 0.01]
    'EinsteinAsymmetric': ('alpha', lambda q: 1e-4 * q),  # Map to [0, 1e-4]
    'PostQuantumGravity': ('gamma_pqg', lambda q: 0.1 * q),  # Map to [0, 0.1]
    'StringTheory': ('alpha_prime', lambda q: 1e-66 * (10 ** q)),  # Log scale
    'ParticipatoryConcept': ('omega', lambda q: 0.01 * q),  # Map to [0, 0.01]
    'FractalSpacetime': ('D', lambda q: 3.0 + 0.1 * q),  # Map to [3.0, 3.1]
    'PhaseTransitionQG': ('T_c', lambda q: 1.0 + 0.1 * q),  # Map to [1.0, 1.1]
    'AsymptoticSafety': ('Lambda_as', lambda q: 1e17 * (10 ** q)),  # Log scale
    'EmergentGravity': ('eta', lambda q: 0.01 * q),  # Map to [0, 0.01]
    'GaugeGravity': ('kappa', lambda q: 0.01 * q),  # Map to [0, 0.01]
    'WeylEMQuantum': ('omega', lambda q: 0.01 * q),  # Map to [0, 0.01]
    'EinsteinRegularizedCore': ('epsilon', lambda q: 1e-3 * q),  # Map to [0, 1e-3]
    'StochasticNoise': ('sigma', lambda q: 1e-3 * q),  # Map to [0, 1e-3]
}

def get_standardized_params(theory_class_name: str, quantum_strength: float = 0.1) -> dict:
    """
    Get standardized parameters for a quantum theory.
    
    Args:
        theory_class_name: Name of the theory class
        quantum_strength: Normalized quantum strength (0 = classical, 1 = maximum quantum)
        
    Returns:
        Dictionary with appropriate parameter name and value
    """
    if theory_class_name not in QUANTUM_PARAMETER_MAPPINGS:
        return {}
        
    param_name, mapping_func = QUANTUM_PARAMETER_MAPPINGS[theory_class_name]
    param_value = mapping_func(quantum_strength)
    
    return {param_name: param_value}

def standardize_quantum_theories(theories: list, quantum_strength: float = 0.1) -> list:
    """
    Create standardized instances of quantum theories for fair comparison.
    
    Args:
        theories: List of theory classes or instances
        quantum_strength: Normalized quantum strength to use
        
    Returns:
        List of theory instances with standardized parameters
    """
    standardized = []
    
    for theory in theories:
        # Get class if instance
        theory_class = theory.__class__ if hasattr(theory, '__class__') else theory
        class_name = theory_class.__name__
        
        if class_name in QUANTUM_PARAMETER_MAPPINGS:
            # Get standardized parameters
            params = get_standardized_params(class_name, quantum_strength)
            
            # Create new instance with standardized parameters
            try:
                instance = theory_class(**params)
                print(f"Standardized {class_name} with {params}")
                standardized.append(instance)
            except Exception as e:
                print(f"Failed to standardize {class_name}: {e}")
                standardized.append(theory)
        else:
            # Non-quantum theory or unknown mapping
            standardized.append(theory)
            
    return standardized 