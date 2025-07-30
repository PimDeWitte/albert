# Quantum Visualizer Documentation

## Overview

The Quantum Visualizer is a comprehensive Python module for visualizing quantum mechanical phenomena. It provides various 2D and 3D visualization techniques based on research from quantum mechanics visualization literature.

## Features

### 1. Wavefunction Visualization

#### 1D Wavefunctions
- **Real and Imaginary Parts**: Traditional representation showing Re(ψ) and Im(ψ)
- **Probability Density**: |ψ|² visualization
- **Phase-Colored Probability**: Probability density colored by quantum phase
- **Phasor Representation**: Complex values shown as rotating arrows

#### 2D Wavefunctions
- **Probability Density**: Heat map of |ψ|²
- **Phase Visualization**: Phase information with magnitude as transparency
- **Real/Imaginary Parts**: Separate visualizations of Re(ψ) and Im(ψ)
- **Phasor Arrows**: Vector field representation of complex amplitude

#### 3D Wavefunction Plots
- **Surface Plots**: 3D surface with phase-based coloring
- **Wireframe**: 3D wireframe representation
- **Contour Plots**: 3D contour visualization

### 2. Quantum State Analysis

#### Density Matrix Tomography
- Visualize real, imaginary, and magnitude components of density matrices
- Useful for analyzing mixed states and entanglement

#### Bloch Sphere
- Plot pure and mixed quantum states on the Bloch sphere
- Support for state trajectories and multiple states
- Customizable labels and colors

### 3. Quantum Phenomena

#### Entanglement Visualization
- Bell inequality violation plots
- Correlation measurements vs theoretical predictions
- Classical bounds visualization

#### Path Integral Visualization
- Feynman path integral with multiple paths
- Phase-weighted path contributions
- Classical vs quantum path comparison

#### Uncertainty Principle
- Position and momentum distribution plots
- Uncertainty product visualization
- Heisenberg uncertainty relation verification

#### Quantum Tunneling
- Wavefunction behavior through potential barriers
- Classically forbidden regions highlighting
- Support for various barrier shapes

### 4. Time Evolution

#### Animated Wavefunction Evolution
- Time-dependent wavefunction animation
- Real-time phase evolution
- Dispersion effects visualization

## Installation

```bash
# Required dependencies
pip install numpy matplotlib torch
```

## Basic Usage

```python
from quantum_visualizer import QuantumVisualizer
import numpy as np

# Initialize visualizer
viz = QuantumVisualizer(style='dark')  # or 'light'

# Create a Gaussian wavepacket
x = np.linspace(-10, 10, 500)
k0 = 2.0  # momentum
sigma = 1.0  # width
psi = np.exp(-(x**2)/(4*sigma**2) + 1j*k0*x)
psi /= np.sqrt(np.sum(np.abs(psi)**2))

# Plot 1D wavefunction
fig = viz.plot_wavefunction_1d(x, psi, title="Gaussian Wavepacket")
fig.savefig('wavefunction.png')
```

## Advanced Examples

### 2D Wavefunction with Angular Momentum

```python
# Create 2D grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Wavefunction with angular momentum
l = 2  # angular momentum quantum number
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)
psi_2d = np.exp(-r**2/2) * np.exp(1j*l*theta)

# Visualize in different modes
fig = viz.plot_wavefunction_2d(X, Y, psi_2d, mode='phase')
```

### Quantum State on Bloch Sphere

```python
# Define quantum states
states = [
    (0, 0, 1),           # |0⟩ (north pole)
    (np.pi, 0, 1),       # |1⟩ (south pole)
    (np.pi/2, 0, 1),     # |+⟩ (x-axis)
    (np.pi/2, np.pi/2, 1) # |+i⟩ (y-axis)
]
labels = ['|0⟩', '|1⟩', '|+⟩', '|+i⟩']

fig = viz.plot_bloch_sphere(states, labels)
```

### Bell Inequality Visualization

```python
# Measurement angles
angles = np.linspace(0, np.pi, 100)

# Quantum correlation
quantum_theory = -np.cos(2 * angles)

# Add experimental noise
measured = quantum_theory + 0.1 * np.random.randn(len(angles))

fig = viz.plot_entanglement_correlation(
    angles, measured, quantum_theory,
    title="Bell Inequality Test"
)
```

## Visualization Modes

### Color Schemes

The module uses a circular HSV colormap for phase visualization:
- Red: φ = 0
- Yellow: φ = π/2
- Cyan: φ = π
- Blue: φ = 3π/2

### Style Options

- **Dark Mode**: Black background with bright colors (default)
- **Light Mode**: White background with dark colors

## Mathematical Background

### Phase Representation

For a complex wavefunction ψ = |ψ|e^(iφ), we visualize:
- Magnitude: |ψ|
- Phase: φ ∈ [-π, π]

### Uncertainty Principle

The module verifies: ΔxΔp ≥ ℏ/2

### Path Integral

Visualizes the sum over paths: ψ(x_f, t_f) = Σ A[path] exp(iS[path]/ℏ)

## Testing

Run the comprehensive test suite:

```bash
python test_quantum_visualizer.py
```

Run the demo to generate all visualization types:

```bash
python demo_quantum_visualizations.py
```

## Scientific References

Based on visualization techniques from:

1. **Wavefunction Visualization**
   - Styer, D. (2000). "Quantum Mechanics: See It Now", AAPT
   - Michielsen, K. & De Raedt, H. "Quantum Mechanics" multimedia presentation

2. **Phase Representation**
   - Feynman, R. "The Feynman Lectures on Physics"
   - Taylor, E. "Quantum Mechanics" visualization approach

3. **Entanglement Visualization**
   - Aspect, A. et al. Bell inequality experiments
   - Kwiat, P.G. & Hardy, L. (2000). "The mystery of the quantum cakes"

4. **Path Integral Visualization**
   - Feynman, R. & Hibbs, A. "Quantum Mechanics and Path Integrals"

## Performance Considerations

- For large 2D/3D arrays, consider downsampling for interactive performance
- Animation generation may be memory intensive for long sequences
- Use `plt.close()` after saving figures to free memory

## Limitations

- Entanglement visualization is limited to correlation plots (full visualization of entangled states remains challenging)
- 3D visualizations may have rendering artifacts depending on viewing angle
- Animation frame rate depends on system performance

## Future Enhancements

- WebGL-based interactive visualizations
- VR/AR support for 3D quantum states
- Real-time quantum simulation integration
- GPU acceleration for large-scale computations

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please submit pull requests with:
- New visualization techniques
- Performance improvements
- Additional quantum phenomena
- Bug fixes

## Citation

If you use this module in research, please cite:
```
@software{quantum_visualizer,
  title = {Quantum Visualizer: Comprehensive Quantum Mechanics Visualization},
  author = {Physics Agent Team},
  year = {2024},
  url = {https://github.com/physics_agent/quantum_visualizer}
}
``` 