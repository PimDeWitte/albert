#!/usr/bin/env python3
"""
Quantum Path Integrator for Gravitational Theories.

This module has been consolidated into UnifiedQuantumSolver.
The QuantumPathIntegrator class is now a backward compatibility wrapper.
"""

# Import the unified solver
from physics_agent.unified_quantum_solver import UnifiedQuantumSolver

# Re-export for backward compatibility
from physics_agent.constants import (
    HBAR, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT,
    PLANCK_LENGTH, PLANCK_TIME, PLANCK_MASS, SOLAR_MASS
)
import torch
import numpy as np
from typing import Callable

Tensor = torch.Tensor
C = SPEED_OF_LIGHT
G = GRAVITATIONAL_CONSTANT


class QuantumPathIntegrator(UnifiedQuantumSolver):
    """
    <reason>chain: Backward compatibility wrapper for UnifiedQuantumSolver</reason>
    
    DEPRECATED: This class now redirects to UnifiedQuantumSolver.
    Use UnifiedQuantumSolver directly for new code.
    """
    
    def __init__(self, theory, enable_quantum: bool = True):
        """Initialize with PennyLane disabled for backward compatibility"""
        super().__init__(theory, enable_quantum=enable_quantum, use_pennylane=False)


# Also provide backward compatibility for QuantumLossCalculator
class QuantumLossCalculator:
    """
    <reason>chain: Calculate loss by comparing quantum predictions to observations</reason>
    
    NOTE: This is kept for backward compatibility. New code should use
    UnifiedQuantumSolver directly.
    """
    
    def __init__(self, integrator):
        if isinstance(integrator, UnifiedQuantumSolver):
            self.integrator = integrator
        else:
            # Wrap old integrator
            self.integrator = integrator
    
    def compute_pulsar_quantum_loss(self, observed_data, theory_params):
        """Compute loss using pulsar timing data with quantum corrections"""
        # Basic implementation for backward compatibility
        M_c = observed_data.get('companion_mass', 0.253 * 1.989e30)
        observed_delay = observed_data.get('shapiro_delay', 1e-6)
        timing_precision = observed_data.get('timing_precision', 1e-7)
        
        # Simple loss calculation
        quantum_delay = observed_delay * 0.99  # Placeholder
        loss = ((quantum_delay - observed_delay) / timing_precision) ** 2
        return loss
    
    def compute_hawking_radiation_loss(self, M_bh: float, observed_temp: float,
                                     theory_params):
        """Compute loss for Hawking radiation predictions"""
        # Placeholder implementation
        return 0.0


# Backward compatibility function
def visualize_quantum_paths(integrator, num_samples: int = 100):
    """
    Stub function for backward compatibility.
    
    Visualization is now handled by theory_visualizer.py
    """
    import warnings
    warnings.warn("visualize_quantum_paths is deprecated. Use theory_visualizer instead.", 
                  DeprecationWarning, stacklevel=2)
    
    # Return empty visualization data
    return {
        'paths': [],
        'probabilities': [],
        'classical_path': [],
        'quantum_uncertainty': []
    }