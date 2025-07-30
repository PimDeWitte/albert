"""
Quantum Visualization Demo
=========================

Demonstrates all quantum visualization capabilities with example data.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_visualizer import QuantumVisualizer
import os


def demo_1d_wavefunction():
    """Demonstrate 1D wavefunction visualization."""
    print("Generating 1D wavefunction visualizations...")
    
    viz = QuantumVisualizer(style='dark')
    x = np.linspace(-10, 10, 500)
    
    # Example 1: Gaussian wavepacket
    k0 = 3.0
    sigma = 1.0
    psi_gaussian = np.exp(-(x**2)/(4*sigma**2) + 1j*k0*x)
    psi_gaussian /= np.sqrt(np.sum(np.abs(psi_gaussian)**2) * (x[1] - x[0]))
    
    fig1 = viz.plot_wavefunction_1d(
        x, psi_gaussian,
        title="Gaussian Wavepacket (k₀ = 3.0)",
        show_phase=True,
        show_probability=True
    )
    fig1.savefig('demo_1d_gaussian.png', dpi=150, facecolor='#0a0a0a')
    
    # Example 2: Superposition of eigenstates
    L = 20
    psi_super = np.zeros_like(x, dtype=complex)
    for n in [1, 2, 3]:
        psi_n = np.sqrt(2/L) * np.sin(n * np.pi * (x + L/2) / L)
        psi_n[x < -L/2] = 0
        psi_n[x > L/2] = 0
        phase = np.exp(1j * n * np.pi / 6)
        psi_super += psi_n * phase / np.sqrt(3)
    
    fig2 = viz.plot_wavefunction_1d(
        x, psi_super,
        title="Superposition of Energy Eigenstates (n=1,2,3)",
        show_phase=True
    )
    fig2.savefig('demo_1d_superposition.png', dpi=150, facecolor='#0a0a0a')


def demo_2d_wavefunction():
    """Demonstrate 2D wavefunction visualization modes."""
    print("Generating 2D wavefunction visualizations...")
    
    viz = QuantumVisualizer(style='dark')
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian with angular momentum
    sigma = 1.0
    l = 2  # Angular momentum quantum number
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    psi_2d = np.exp(-r**2/(2*sigma**2)) * np.exp(1j*l*theta)
    psi_2d /= np.sqrt(np.sum(np.abs(psi_2d)**2))
    
    # Generate all visualization modes
    modes = ['probability', 'phase', 'real', 'imag', 'phasor']
    for mode in modes:
        fig = viz.plot_wavefunction_2d(
            X, Y, psi_2d,
            title=f"2D Wavefunction with Angular Momentum (l={l}) - {mode.capitalize()}",
            mode=mode
        )
        fig.savefig(f'demo_2d_{mode}.png', dpi=150, facecolor='#0a0a0a')
        plt.close(fig)


def demo_3d_wavefunction():
    """Demonstrate 3D wavefunction visualization."""
    print("Generating 3D wavefunction visualizations...")
    
    viz = QuantumVisualizer(style='dark')
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    # Double slit pattern
    k = 5.0
    d = 1.5  # Slit separation
    a = 0.3  # Slit width
    
    # Two Gaussian slits
    psi_slit1 = np.exp(-((Y - d/2)**2)/(2*a**2)) * np.exp(1j*k*X)
    psi_slit2 = np.exp(-((Y + d/2)**2)/(2*a**2)) * np.exp(1j*k*X)
    psi_2d = (psi_slit1 + psi_slit2) / np.sqrt(2)
    
    # Add some decay
    psi_2d *= np.exp(-0.1*np.abs(X))
    
    modes = ['surface', 'wireframe', 'contour3d']
    for mode in modes:
        fig = viz.plot_wavefunction_3d(
            X, Y, psi_2d,
            title=f"Double Slit Interference Pattern - {mode.capitalize()}",
            mode=mode
        )
        fig.savefig(f'demo_3d_{mode}.png', dpi=150, facecolor='#0a0a0a')
        plt.close(fig)


def demo_quantum_state_tomography():
    """Demonstrate quantum state tomography visualization."""
    print("Generating quantum state tomography...")
    
    viz = QuantumVisualizer(style='dark')
    
    # Create Bell state density matrix
    # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    dim = 4
    bell_state = np.zeros((dim, dim), dtype=complex)
    bell_state[0, 0] = 0.5  # |00⟩⟨00|
    bell_state[0, 3] = 0.5  # |00⟩⟨11|
    bell_state[3, 0] = 0.5  # |11⟩⟨00|
    bell_state[3, 3] = 0.5  # |11⟩⟨11|
    
    fig = viz.plot_quantum_state_tomography(
        bell_state,
        title="Bell State |Φ⁺⟩ Density Matrix"
    )
    fig.savefig('demo_tomography.png', dpi=150, facecolor='#0a0a0a')


def demo_bloch_sphere():
    """Demonstrate Bloch sphere visualization."""
    print("Generating Bloch sphere...")
    
    viz = QuantumVisualizer(style='dark')
    
    # Create interesting quantum states
    states = []
    labels = []
    
    # Pure states on different axes
    states.extend([
        (0, 0, 1),              # |0⟩
        (np.pi, 0, 1),          # |1⟩
        (np.pi/2, 0, 1),        # |+⟩
        (np.pi/2, np.pi/2, 1),  # |+i⟩
    ])
    labels.extend(['|0⟩', '|1⟩', '|+⟩', '|+i⟩'])
    
    # Mixed states (inside the sphere)
    states.extend([
        (np.pi/4, 0, 0.7),      # Partially mixed
        (3*np.pi/4, np.pi/4, 0.5),  # Maximally mixed direction
    ])
    labels.extend(['Mixed 1', 'Mixed 2'])
    
    # Trajectory on sphere
    theta_traj = np.linspace(0, np.pi, 20)
    phi_traj = np.linspace(0, 4*np.pi, 20)
    for i in range(len(theta_traj)):
        states.append((theta_traj[i], phi_traj[i], 1))
        labels.append('')  # No label for trajectory points
    
    fig = viz.plot_bloch_sphere(
        states, labels,
        title="Quantum States on Bloch Sphere"
    )
    fig.savefig('demo_bloch_sphere.png', dpi=150, facecolor='#0a0a0a')


def demo_entanglement():
    """Demonstrate entanglement correlation (Bell inequality)."""
    print("Generating entanglement correlation plot...")
    
    viz = QuantumVisualizer(style='dark')
    
    # Measurement angles
    angles = np.linspace(0, np.pi, 100)
    
    # Theoretical quantum correlation for singlet state
    quantum_theory = -np.cos(2 * angles)
    
    # Simulated measurement with noise
    np.random.seed(42)
    measured = quantum_theory + 0.08 * np.random.randn(len(angles))
    
    fig = viz.plot_entanglement_correlation(
        angles, measured, quantum_theory,
        title="Bell Inequality Test - Quantum vs Classical Correlations"
    )
    fig.savefig('demo_entanglement.png', dpi=150, facecolor='#0a0a0a')


def demo_path_integral():
    """Demonstrate Feynman path integral visualization."""
    print("Generating path integral visualization...")
    
    viz = QuantumVisualizer(style='dark')
    
    # Generate quantum paths
    n_paths = 50
    n_points = 100
    
    start = np.array([-3, 0])
    end = np.array([3, 0])
    
    paths = []
    amplitudes = []
    
    # Classical path
    t = np.linspace(0, 1, n_points)
    classical_path = np.outer(1-t, start) + np.outer(t, end)
    
    # Quantum paths with different "actions"
    for i in range(n_paths):
        # Random fluctuations around classical path
        amplitude_factor = np.exp(-i/10)  # Paths with larger deviations have smaller amplitude
        
        # Create path
        deviation = amplitude_factor * np.random.randn(n_points, 2)
        deviation[0] = 0  # Fix endpoints
        deviation[-1] = 0
        
        path = classical_path + deviation
        paths.append(path)
        
        # Complex amplitude with phase related to action
        phase = 2 * np.pi * np.random.rand()
        amplitude = amplitude_factor * np.exp(1j * phase)
        amplitudes.append(amplitude)
    
    fig = viz.plot_path_integral_visualization(
        paths, np.array(amplitudes), classical_path,
        title="Feynman Path Integral - Multiple Quantum Paths"
    )
    fig.savefig('demo_path_integral.png', dpi=150, facecolor='#0a0a0a')


def demo_uncertainty_principle():
    """Demonstrate uncertainty principle visualization."""
    print("Generating uncertainty principle visualization...")
    
    viz = QuantumVisualizer(style='dark')
    
    # Create different uncertainty scenarios
    scenarios = [
        {'sigma_x': 0.5, 'sigma_p': 1.0, 'title': 'Near Minimum Uncertainty'},
        {'sigma_x': 0.3, 'sigma_p': 2.0, 'title': 'Position-Precise State'},
        {'sigma_x': 2.0, 'sigma_p': 0.3, 'title': 'Momentum-Precise State'},
    ]
    
    for i, scenario in enumerate(scenarios):
        x_vals = np.linspace(-5, 5, 300)
        p_vals = np.linspace(-5, 5, 300)
        
        # Gaussian distributions
        pos_dist = np.exp(-x_vals**2 / (2 * scenario['sigma_x']**2))
        pos_dist /= np.sum(pos_dist) * (x_vals[1] - x_vals[0])
        
        mom_dist = np.exp(-p_vals**2 / (2 * scenario['sigma_p']**2))
        mom_dist /= np.sum(mom_dist) * (p_vals[1] - p_vals[0])
        
        fig = viz.plot_uncertainty_principle(
            pos_dist, mom_dist, x_vals, p_vals,
            title=f"Uncertainty Principle - {scenario['title']}"
        )
        fig.savefig(f'demo_uncertainty_{i+1}.png', dpi=150, facecolor='#0a0a0a')
        plt.close(fig)


def demo_quantum_tunneling():
    """Demonstrate quantum tunneling visualization."""
    print("Generating quantum tunneling visualization...")
    
    viz = QuantumVisualizer(style='dark')
    
    # Create potential barriers
    x = np.linspace(-10, 10, 1000)
    
    scenarios = [
        {
            'name': 'rectangular',
            'V': lambda x: np.where((x > -2) & (x < 2), 5.0, 0.0),
            'E': 3.0,
            'title': 'Rectangular Barrier'
        },
        {
            'name': 'gaussian',
            'V': lambda x: 5.0 * np.exp(-x**2 / 2),
            'E': 2.5,
            'title': 'Gaussian Barrier'
        },
        {
            'name': 'double',
            'V': lambda x: 4.0 * (np.exp(-(x-2)**2) + np.exp(-(x+2)**2)),
            'E': 3.0,
            'title': 'Double Barrier'
        }
    ]
    
    for scenario in scenarios:
        V = scenario['V'](x)
        E = scenario['E']
        
        # Create approximate tunneling wavefunction
        # This is simplified - real solution would solve Schrödinger equation
        psi = np.zeros_like(x, dtype=complex)
        
        # Find classical turning points
        allowed = V < E
        
        # Simple model: oscillating in allowed regions, exponential decay in forbidden
        k_allowed = np.sqrt(2 * E)
        
        for i in range(len(x)):
            if allowed[i]:
                psi[i] = np.exp(1j * k_allowed * x[i])
            else:
                # Exponential decay
                k_forbidden = np.sqrt(2 * (V[i] - E))
                if i > 0 and allowed[i-1]:
                    # Just entered forbidden region
                    decay_start = i
                if i < len(x) - 1 and not allowed[i] and allowed[i+1]:
                    # Exiting forbidden region
                    decay_length = i - decay_start
                    transmission = np.exp(-k_forbidden * decay_length * (x[1] - x[0]))
                psi[i] = psi[max(0, i-1)] * np.exp(-k_forbidden * (x[1] - x[0]))
        
        # Normalize
        psi /= np.max(np.abs(psi))
        
        fig = viz.plot_quantum_tunneling(
            x, V, E, psi,
            title=f"Quantum Tunneling - {scenario['title']}"
        )
        fig.savefig(f"demo_tunneling_{scenario['name']}.png", dpi=150, facecolor='#0a0a0a')
        plt.close(fig)


def demo_wavefunction_evolution():
    """Demonstrate animated wavefunction evolution."""
    print("Generating wavefunction evolution animation...")
    
    viz = QuantumVisualizer(style='dark')
    
    # Create Gaussian wavepacket evolution
    x = np.linspace(-10, 10, 400)
    t_steps = np.linspace(0, 5, 50)
    
    # Initial conditions
    x0 = -3
    k0 = 4
    sigma = 0.5
    
    psi_evolution = []
    
    for t in t_steps:
        # Free particle evolution (with dispersion)
        sigma_t = sigma * np.sqrt(1 + (t/(2*sigma**2))**2)
        norm = (2*np.pi*sigma_t**2)**(-0.25)
        
        phase1 = -(x - x0 - k0*t)**2 / (4*sigma_t**2)
        phase2 = k0*(x - x0) - k0**2*t/2
        phase3 = -t/(4*sigma**2) * np.arctan(t/(2*sigma**2))
        
        psi = norm * np.exp(phase1 + 1j*(phase2 + phase3))
        psi_evolution.append(psi)
    
    anim = viz.animate_wavefunction_evolution(
        x, psi_evolution, t_steps,
        title="Free Particle Wavepacket Evolution",
        save_path='demo_evolution.gif'
    )
    print("  Animation saved as demo_evolution.gif")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Quantum Visualization Demonstrations")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('quantum_viz_demo', exist_ok=True)
    os.chdir('quantum_viz_demo')
    
    # Run all demos
    demo_1d_wavefunction()
    demo_2d_wavefunction()
    demo_3d_wavefunction()
    demo_quantum_state_tomography()
    demo_bloch_sphere()
    demo_entanglement()
    demo_path_integral()
    demo_uncertainty_principle()
    demo_quantum_tunneling()
    demo_wavefunction_evolution()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("Output files saved in 'quantum_viz_demo' directory")
    print("=" * 60)
    
    # Generate summary HTML
    generate_summary_html()


def generate_summary_html():
    """Generate HTML summary of all visualizations."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Visualization Gallery</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #0a0a0a;
            color: white;
            margin: 20px;
        }
        h1, h2 {
            color: #00ffff;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .image-container {
            background-color: #1a1a1a;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        .description {
            margin-top: 10px;
            font-size: 14px;
            color: #cccccc;
        }
    </style>
</head>
<body>
    <h1>Quantum Visualization Gallery</h1>
    
    <h2>1D Wavefunctions</h2>
    <div class="gallery">
        <div class="image-container">
            <img src="demo_1d_gaussian.png" alt="Gaussian Wavepacket">
            <div class="description">Gaussian wavepacket showing real/imaginary parts and phase-colored probability</div>
        </div>
        <div class="image-container">
            <img src="demo_1d_superposition.png" alt="Superposition">
            <div class="description">Superposition of energy eigenstates</div>
        </div>
    </div>
    
    <h2>2D Wavefunction Visualizations</h2>
    <div class="gallery">
        <div class="image-container">
            <img src="demo_2d_probability.png" alt="2D Probability">
            <div class="description">Probability density |ψ|²</div>
        </div>
        <div class="image-container">
            <img src="demo_2d_phase.png" alt="2D Phase">
            <div class="description">Phase with magnitude as transparency</div>
        </div>
        <div class="image-container">
            <img src="demo_2d_phasor.png" alt="2D Phasor">
            <div class="description">Phasor arrow representation</div>
        </div>
    </div>
    
    <h2>3D Visualizations</h2>
    <div class="gallery">
        <div class="image-container">
            <img src="demo_3d_surface.png" alt="3D Surface">
            <div class="description">Surface plot with phase coloring</div>
        </div>
        <div class="image-container">
            <img src="demo_3d_wireframe.png" alt="3D Wireframe">
            <div class="description">Wireframe representation</div>
        </div>
        <div class="image-container">
            <img src="demo_3d_contour3d.png" alt="3D Contour">
            <div class="description">3D contour plot</div>
        </div>
    </div>
    
    <h2>Quantum State Analysis</h2>
    <div class="gallery">
        <div class="image-container">
            <img src="demo_tomography.png" alt="State Tomography">
            <div class="description">Bell state density matrix tomography</div>
        </div>
        <div class="image-container">
            <img src="demo_bloch_sphere.png" alt="Bloch Sphere">
            <div class="description">Quantum states on the Bloch sphere</div>
        </div>
    </div>
    
    <h2>Quantum Phenomena</h2>
    <div class="gallery">
        <div class="image-container">
            <img src="demo_entanglement.png" alt="Entanglement">
            <div class="description">Bell inequality violation</div>
        </div>
        <div class="image-container">
            <img src="demo_path_integral.png" alt="Path Integral">
            <div class="description">Feynman path integral visualization</div>
        </div>
    </div>
    
    <h2>Uncertainty Principle</h2>
    <div class="gallery">
        <div class="image-container">
            <img src="demo_uncertainty_1.png" alt="Minimum Uncertainty">
            <div class="description">Near minimum uncertainty state</div>
        </div>
        <div class="image-container">
            <img src="demo_uncertainty_2.png" alt="Position Precise">
            <div class="description">Position-precise state</div>
        </div>
        <div class="image-container">
            <img src="demo_uncertainty_3.png" alt="Momentum Precise">
            <div class="description">Momentum-precise state</div>
        </div>
    </div>
    
    <h2>Quantum Tunneling</h2>
    <div class="gallery">
        <div class="image-container">
            <img src="demo_tunneling_rectangular.png" alt="Rectangular Barrier">
            <div class="description">Tunneling through rectangular barrier</div>
        </div>
        <div class="image-container">
            <img src="demo_tunneling_gaussian.png" alt="Gaussian Barrier">
            <div class="description">Tunneling through Gaussian barrier</div>
        </div>
        <div class="image-container">
            <img src="demo_tunneling_double.png" alt="Double Barrier">
            <div class="description">Tunneling through double barrier</div>
        </div>
    </div>
    
    <h2>Time Evolution</h2>
    <div class="gallery">
        <div class="image-container">
            <img src="demo_evolution.gif" alt="Wavepacket Evolution">
            <div class="description">Free particle wavepacket evolution showing dispersion</div>
        </div>
    </div>
    
    <p style="margin-top: 40px; text-align: center; color: #666;">
        Generated by Quantum Visualizer - Comprehensive quantum mechanics visualization toolkit
    </p>
</body>
</html>
"""
    
    with open('index.html', 'w') as f:
        f.write(html_content)
    
    print("\nSummary HTML generated: index.html")


if __name__ == '__main__':
    main() 