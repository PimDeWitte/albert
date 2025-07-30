"""
Quantum Visualization Module
==========================

Provides various 2D and 3D visualization techniques for quantum phenomena:
- Wavefunction probability density plots
- Phase visualization (color and phasor representations)
- Quantum state evolution animations
- Entanglement visualizations
- Path integral visualizations
- Uncertainty principle demonstrations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)


class QuantumVisualizer:
    """
    Comprehensive visualization tools for quantum mechanics phenomena.
    
    Includes:
    - Wavefunction probability density plots (2D/3D)
    - Phase visualization using color mapping
    - Phasor arrow representations
    - Quantum state evolution animations
    - Entanglement correlation plots
    - Path integral visualization
    - Uncertainty principle demonstrations
    """
    
    def __init__(self, style='dark'):
        """Initialize quantum visualizer with style settings."""
        self.style = style
        if style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#0a0a0a'
            self.text_color = 'white'
            self.grid_color = '#333333'
        else:
            plt.style.use('default')
            self.bg_color = 'white'
            self.text_color = 'black'
            self.grid_color = '#cccccc'
        
        # Phase-to-color mapping
        self.phase_cmap = self._create_phase_colormap()
        
    def _create_phase_colormap(self):
        """Create a circular colormap for phase visualization."""
        # Create HSV colormap that cycles through hues
        n = 256
        hue = np.linspace(0, 1, n)
        saturation = np.ones(n)
        value = np.ones(n)
        
        # Convert HSV to RGB
        hsv = np.stack([hue, saturation, value], axis=1)
        rgb = colors.hsv_to_rgb(hsv.reshape(1, n, 3)).reshape(n, 3)
        
        # Create colormap
        cmap = colors.LinearSegmentedColormap.from_list('phase', rgb)
        return cmap
    
    def plot_wavefunction_1d(self, x: np.ndarray, psi: np.ndarray, 
                           title: str = "1D Wavefunction", 
                           show_phase: bool = True,
                           show_probability: bool = True) -> plt.Figure:
        """
        Plot 1D wavefunction with magnitude and phase information.
        
        Args:
            x: Position array
            psi: Complex wavefunction values
            title: Plot title
            show_phase: Whether to show phase information
            show_probability: Whether to show probability density
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2 if show_phase else 1, 1, figsize=(10, 8 if show_phase else 5))
        if not show_phase:
            axes = [axes]
        
        # Probability density plot
        ax1 = axes[0]
        prob = np.abs(psi)**2
        
        if show_probability:
            ax1.fill_between(x, prob, alpha=0.3, color='cyan', label='|ψ|²')
            ax1.plot(x, prob, 'c-', linewidth=2)
        
        # Real and imaginary parts
        ax1.plot(x, np.real(psi), 'r-', linewidth=1.5, label='Re(ψ)', alpha=0.8)
        ax1.plot(x, np.imag(psi), 'b-', linewidth=1.5, label='Im(ψ)', alpha=0.8)
        
        ax1.set_xlabel('Position (x)', fontsize=12)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_title(title, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Phase plot
        if show_phase:
            ax2 = axes[1]
            phase = np.angle(psi)
            
            # Create color-coded line segments
            points = np.array([x, prob]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Normalize phase to [0, 1] for colormap
            phase_norm = (phase + np.pi) / (2 * np.pi)
            
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap=self.phase_cmap, linewidth=3)
            lc.set_array(phase_norm)
            line = ax2.add_collection(lc)
            
            ax2.set_xlim(x.min(), x.max())
            ax2.set_ylim(0, prob.max() * 1.1)
            ax2.set_xlabel('Position (x)', fontsize=12)
            ax2.set_ylabel('|ψ|²', fontsize=12)
            ax2.set_title('Phase-colored Probability Density', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(line, ax=ax2, orientation='horizontal', pad=0.1)
            cbar.set_label('Phase', fontsize=10)
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        plt.tight_layout()
        return fig
    
    def plot_wavefunction_2d(self, X: np.ndarray, Y: np.ndarray, psi: np.ndarray,
                           title: str = "2D Wavefunction", 
                           mode: str = 'probability') -> plt.Figure:
        """
        Plot 2D wavefunction using various visualization modes.
        
        Args:
            X, Y: Meshgrid arrays for positions
            psi: Complex wavefunction values (2D array)
            title: Plot title
            mode: Visualization mode ('probability', 'phase', 'real', 'imag', 'phasor')
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(10, 8))
        
        if mode == 'probability':
            ax = fig.add_subplot(111)
            prob = np.abs(psi)**2
            im = ax.imshow(prob, extent=[X.min(), X.max(), Y.min(), Y.max()],
                          cmap='hot', origin='lower', interpolation='bilinear')
            ax.set_title(f'{title} - Probability Density', fontsize=14)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('|ψ|²', fontsize=12)
            
        elif mode == 'phase':
            ax = fig.add_subplot(111)
            magnitude = np.abs(psi)
            phase = np.angle(psi)
            
            # Create RGBA image with magnitude as alpha
            phase_norm = (phase + np.pi) / (2 * np.pi)
            rgba = self.phase_cmap(phase_norm)
            rgba[..., 3] = magnitude / magnitude.max()  # Set alpha channel
            
            ax.imshow(rgba, extent=[X.min(), X.max(), Y.min(), Y.max()],
                     origin='lower', interpolation='bilinear')
            ax.set_title(f'{title} - Phase with Magnitude', fontsize=14)
            
            # Add phase colorbar
            sm = plt.cm.ScalarMappable(cmap=self.phase_cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Phase', fontsize=12)
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
            
        elif mode == 'real':
            ax = fig.add_subplot(111)
            im = ax.imshow(np.real(psi), extent=[X.min(), X.max(), Y.min(), Y.max()],
                          cmap='RdBu', origin='lower', interpolation='bilinear')
            ax.set_title(f'{title} - Real Part', fontsize=14)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Re(ψ)', fontsize=12)
            
        elif mode == 'imag':
            ax = fig.add_subplot(111)
            im = ax.imshow(np.imag(psi), extent=[X.min(), X.max(), Y.min(), Y.max()],
                          cmap='RdBu', origin='lower', interpolation='bilinear')
            ax.set_title(f'{title} - Imaginary Part', fontsize=14)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Im(ψ)', fontsize=12)
            
        elif mode == 'phasor':
            ax = fig.add_subplot(111)
            # Show probability as background
            prob = np.abs(psi)**2
            ax.imshow(prob, extent=[X.min(), X.max(), Y.min(), Y.max()],
                     cmap='gray', origin='lower', alpha=0.5, interpolation='bilinear')
            
            # Add phasor arrows at selected points
            skip = max(1, len(X) // 20)  # Show arrows at every nth point
            X_sub = X[::skip, ::skip]
            Y_sub = Y[::skip, ::skip]
            psi_sub = psi[::skip, ::skip]
            
            # Calculate arrow components
            magnitude = np.abs(psi_sub)
            phase = np.angle(psi_sub)
            dx = magnitude * np.cos(phase) * 0.5
            dy = magnitude * np.sin(phase) * 0.5
            
            # Normalize arrows
            max_mag = np.max(magnitude)
            if max_mag > 0:
                dx /= max_mag
                dy /= max_mag
            
            ax.quiver(X_sub, Y_sub, dx, dy, magnitude, cmap='viridis', 
                     scale=20, width=0.003, headwidth=3, headlength=4)
            ax.set_title(f'{title} - Phasor Representation', fontsize=14)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_wavefunction_3d(self, X: np.ndarray, Y: np.ndarray, psi: np.ndarray,
                           title: str = "3D Wavefunction", 
                           mode: str = 'surface') -> plt.Figure:
        """
        Plot 3D visualization of 2D wavefunction.
        
        Args:
            X, Y: Meshgrid arrays for positions
            psi: Complex wavefunction values (2D array)
            title: Plot title
            mode: Visualization mode ('surface', 'wireframe', 'contour3d')
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        prob = np.abs(psi)**2
        phase = np.angle(psi)
        phase_norm = (phase + np.pi) / (2 * np.pi)
        
        if mode == 'surface':
            # Create face colors based on phase
            facecolors = self.phase_cmap(phase_norm)
            
            surf = ax.plot_surface(X, Y, prob, facecolors=facecolors,
                                 linewidth=0, antialiased=True, alpha=0.9)
            
        elif mode == 'wireframe':
            # Wireframe with phase-based colors
            ax.plot_wireframe(X, Y, prob, colors='cyan', linewidth=0.5, alpha=0.8)
            
            # Add phase information as scatter points
            skip = max(1, len(X) // 30)
            X_sub = X[::skip, ::skip]
            Y_sub = Y[::skip, ::skip]
            prob_sub = prob[::skip, ::skip]
            phase_sub = phase_norm[::skip, ::skip]
            
            ax.scatter(X_sub, Y_sub, prob_sub, c=phase_sub, cmap=self.phase_cmap,
                      s=20, alpha=0.8)
            
        elif mode == 'contour3d':
            # 3D contour plot
            levels = np.linspace(prob.min(), prob.max(), 20)
            ax.contour3D(X, Y, prob, levels, cmap='viridis', linewidths=1, alpha=0.8)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('|ψ|²', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        return fig
    
    def animate_wavefunction_evolution(self, x: np.ndarray, 
                                     psi_evolution: List[np.ndarray],
                                     time_steps: np.ndarray,
                                     title: str = "Wavefunction Evolution",
                                     save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create animation of wavefunction evolution over time.
        
        Args:
            x: Position array
            psi_evolution: List of wavefunctions at each time step
            time_steps: Array of time values
            title: Animation title
            save_path: Path to save animation (optional)
            
        Returns:
            Animation object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Initialize plots
        prob_line, = ax1.plot([], [], 'c-', linewidth=2, label='|ψ|²')
        real_line, = ax1.plot([], [], 'r-', linewidth=1.5, label='Re(ψ)', alpha=0.8)
        imag_line, = ax1.plot([], [], 'b-', linewidth=1.5, label='Im(ψ)', alpha=0.8)
        
        # Phase plot setup
        from matplotlib.collections import LineCollection
        points = np.array([x, np.zeros_like(x)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=self.phase_cmap, linewidth=3)
        phase_line = ax2.add_collection(lc)
        
        # Set limits
        all_psi = np.concatenate(psi_evolution)
        y_max = np.max(np.abs(all_psi)) * 1.1
        prob_max = np.max([np.max(np.abs(psi)**2) for psi in psi_evolution]) * 1.1
        
        ax1.set_xlim(x.min(), x.max())
        ax1.set_ylim(-y_max, y_max)
        ax1.set_xlabel('Position (x)', fontsize=12)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_title(title, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_xlim(x.min(), x.max())
        ax2.set_ylim(0, prob_max)
        ax2.set_xlabel('Position (x)', fontsize=12)
        ax2.set_ylabel('|ψ|²', fontsize=12)
        ax2.set_title('Phase-colored Probability Density', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Time text
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                           bbox=dict(boxstyle='round', facecolor=self.bg_color, alpha=0.8))
        
        def animate(i):
            psi = psi_evolution[i]
            prob = np.abs(psi)**2
            
            # Update main plot
            prob_line.set_data(x, prob)
            real_line.set_data(x, np.real(psi))
            imag_line.set_data(x, np.imag(psi))
            
            # Update phase plot
            phase = np.angle(psi)
            phase_norm = (phase + np.pi) / (2 * np.pi)
            
            points = np.array([x, prob]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc.set_segments(segments)
            lc.set_array(phase_norm)
            
            # Update time
            time_text.set_text(f't = {time_steps[i]:.3f}')
            
            return prob_line, real_line, imag_line, lc, time_text
        
        anim = FuncAnimation(fig, animate, frames=len(psi_evolution),
                           interval=50, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
        
        plt.tight_layout()
        return anim
    
    def plot_quantum_state_tomography(self, density_matrix: np.ndarray,
                                    title: str = "Quantum State Tomography") -> plt.Figure:
        """
        Visualize quantum state density matrix.
        
        Args:
            density_matrix: Complex density matrix
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 6))
        
        # Real part
        ax1 = fig.add_subplot(131)
        im1 = ax1.imshow(np.real(density_matrix), cmap='RdBu', 
                        interpolation='nearest', vmin=-1, vmax=1)
        ax1.set_title('Real Part', fontsize=12)
        ax1.set_xlabel('Column', fontsize=10)
        ax1.set_ylabel('Row', fontsize=10)
        plt.colorbar(im1, ax=ax1)
        
        # Imaginary part
        ax2 = fig.add_subplot(132)
        im2 = ax2.imshow(np.imag(density_matrix), cmap='RdBu',
                        interpolation='nearest', vmin=-1, vmax=1)
        ax2.set_title('Imaginary Part', fontsize=12)
        ax2.set_xlabel('Column', fontsize=10)
        ax2.set_ylabel('Row', fontsize=10)
        plt.colorbar(im2, ax=ax2)
        
        # Magnitude
        ax3 = fig.add_subplot(133)
        im3 = ax3.imshow(np.abs(density_matrix), cmap='viridis',
                        interpolation='nearest', vmin=0, vmax=1)
        ax3.set_title('Magnitude', fontsize=12)
        ax3.set_xlabel('Column', fontsize=10)
        ax3.set_ylabel('Row', fontsize=10)
        plt.colorbar(im3, ax=ax3)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_bloch_sphere(self, states: List[Tuple[float, float, float]],
                         labels: Optional[List[str]] = None,
                         title: str = "Bloch Sphere") -> plt.Figure:
        """
        Plot quantum states on the Bloch sphere.
        
        Args:
            states: List of (theta, phi, r) coordinates for each state
            labels: Optional labels for each state
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, 
                         color='gray', linewidth=0.5)
        
        # Draw axes
        ax.plot([0, 1.2], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.5)
        ax.plot([0, 0], [0, 1.2], [0, 0], 'k-', linewidth=1, alpha=0.5)
        ax.plot([0, 0], [0, 0], [0, 1.2], 'k-', linewidth=1, alpha=0.5)
        
        # Add axis labels
        ax.text(1.3, 0, 0, 'X', fontsize=12)
        ax.text(0, 1.3, 0, 'Y', fontsize=12)
        ax.text(0, 0, 1.3, 'Z', fontsize=12)
        
        # Plot states
        colors = plt.cm.rainbow(np.linspace(0, 1, len(states)))
        
        for i, (theta, phi, r) in enumerate(states):
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            # Plot state point
            ax.scatter(x, y, z, color=colors[i], s=100, alpha=0.8)
            
            # Draw vector from origin
            ax.plot([0, x], [0, y], [0, z], color=colors[i], linewidth=2, alpha=0.6)
            
            # Add label if provided
            if labels and i < len(labels):
                ax.text(x*1.1, y*1.1, z*1.1, labels[i], fontsize=10)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        return fig
    
    def plot_entanglement_correlation(self, measurement_angles: np.ndarray,
                                    correlations: np.ndarray,
                                    theory_curve: Optional[np.ndarray] = None,
                                    title: str = "Entanglement Correlations") -> plt.Figure:
        """
        Plot quantum entanglement correlations (e.g., Bell inequality violations).
        
        Args:
            measurement_angles: Array of measurement angles
            correlations: Measured correlation values
            theory_curve: Optional theoretical prediction
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot measured correlations
        ax.scatter(measurement_angles, correlations, color='red', s=50, 
                  alpha=0.8, label='Measured', zorder=3)
        
        # Plot theory curve if provided
        if theory_curve is not None:
            ax.plot(measurement_angles, theory_curve, 'b-', linewidth=2,
                   label='Quantum Theory', alpha=0.8)
        
        # Plot classical bounds (Bell inequality)
        classical_bound = 2 / np.sqrt(2)
        ax.axhline(y=classical_bound, color='green', linestyle='--', 
                  linewidth=2, label=f'Classical Bound = {classical_bound:.3f}')
        ax.axhline(y=-classical_bound, color='green', linestyle='--', linewidth=2)
        
        # Shaded region for quantum violation
        ax.fill_between(measurement_angles, classical_bound, 
                       np.maximum(correlations, classical_bound),
                       color='yellow', alpha=0.3, label='Quantum Violation')
        ax.fill_between(measurement_angles, -classical_bound,
                       np.minimum(correlations, -classical_bound),
                       color='yellow', alpha=0.3)
        
        ax.set_xlabel('Measurement Angle (radians)', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-3, 3)
        
        plt.tight_layout()
        return fig
    
    def plot_path_integral_visualization(self, paths: List[np.ndarray],
                                       amplitudes: np.ndarray,
                                       classical_path: Optional[np.ndarray] = None,
                                       title: str = "Path Integral Visualization") -> plt.Figure:
        """
        Visualize Feynman path integral with multiple paths.
        
        Args:
            paths: List of paths (each path is 2D array of positions over time)
            amplitudes: Complex amplitudes for each path
            classical_path: Optional classical path for comparison
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Normalize amplitudes for visualization
        weights = np.abs(amplitudes)
        weights = weights / np.max(weights) if np.max(weights) > 0 else weights
        phases = np.angle(amplitudes)
        phase_norm = (phases + np.pi) / (2 * np.pi)
        
        # Plot quantum paths
        for i, path in enumerate(paths):
            color = self.phase_cmap(phase_norm[i])
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=weights[i]*0.5,
                   linewidth=1)
        
        # Highlight paths with highest amplitude
        top_indices = np.argsort(weights)[-5:]  # Top 5 paths
        for idx in top_indices:
            path = paths[idx]
            color = self.phase_cmap(phase_norm[idx])
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.8,
                   linewidth=2, label=f'Path {idx}: |A|={weights[idx]:.2f}')
        
        # Plot classical path if provided
        if classical_path is not None:
            ax.plot(classical_path[:, 0], classical_path[:, 1], 'w--',
                   linewidth=3, label='Classical Path')
        
        # Mark start and end points
        if len(paths) > 0:
            start = paths[0][0]
            end = paths[0][-1]
            ax.scatter(*start, color='green', s=200, marker='o', 
                      edgecolors='white', linewidth=2, label='Start', zorder=5)
            ax.scatter(*end, color='red', s=200, marker='*',
                      edgecolors='white', linewidth=2, label='End', zorder=5)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Add phase colorbar
        sm = plt.cm.ScalarMappable(cmap=self.phase_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Path Phase', fontsize=12)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        plt.tight_layout()
        return fig
    
    def plot_uncertainty_principle(self, position_dist: np.ndarray,
                                 momentum_dist: np.ndarray,
                                 x_vals: np.ndarray,
                                 p_vals: np.ndarray,
                                 title: str = "Uncertainty Principle") -> plt.Figure:
        """
        Visualize the uncertainty principle with position and momentum distributions.
        
        Args:
            position_dist: Position probability distribution
            momentum_dist: Momentum probability distribution
            x_vals: Position values
            p_vals: Momentum values
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 6))
        
        # Position distribution
        ax1 = fig.add_subplot(131)
        ax1.fill_between(x_vals, position_dist, alpha=0.5, color='blue')
        ax1.plot(x_vals, position_dist, 'b-', linewidth=2)
        
        # Calculate and display uncertainty
        x_mean = np.sum(x_vals * position_dist) / np.sum(position_dist)
        x_var = np.sum((x_vals - x_mean)**2 * position_dist) / np.sum(position_dist)
        delta_x = np.sqrt(x_var)
        
        ax1.axvline(x_mean, color='red', linestyle='--', alpha=0.7)
        ax1.axvspan(x_mean - delta_x, x_mean + delta_x, alpha=0.2, color='red')
        ax1.text(0.05, 0.95, f'Δx = {delta_x:.3f}', transform=ax1.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Position (x)', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_title('Position Distribution', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Momentum distribution
        ax2 = fig.add_subplot(132)
        ax2.fill_between(p_vals, momentum_dist, alpha=0.5, color='green')
        ax2.plot(p_vals, momentum_dist, 'g-', linewidth=2)
        
        # Calculate and display uncertainty
        p_mean = np.sum(p_vals * momentum_dist) / np.sum(momentum_dist)
        p_var = np.sum((p_vals - p_mean)**2 * momentum_dist) / np.sum(momentum_dist)
        delta_p = np.sqrt(p_var)
        
        ax2.axvline(p_mean, color='red', linestyle='--', alpha=0.7)
        ax2.axvspan(p_mean - delta_p, p_mean + delta_p, alpha=0.2, color='red')
        ax2.text(0.05, 0.95, f'Δp = {delta_p:.3f}', transform=ax2.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Momentum (p)', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Momentum Distribution', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Uncertainty product visualization
        ax3 = fig.add_subplot(133)
        
        # Show uncertainty relation
        uncertainty_product = delta_x * delta_p
        hbar = 1.0  # In natural units
        
        # Create visual representation
        theta = np.linspace(0, 2*np.pi, 100)
        r_min = hbar / 2  # Minimum uncertainty
        
        # Uncertainty ellipse
        a = delta_x
        b = delta_p
        x_ellipse = a * np.cos(theta)
        y_ellipse = b * np.sin(theta)
        
        ax3.fill(x_ellipse, y_ellipse, alpha=0.3, color='purple', 
                label=f'ΔxΔp = {uncertainty_product:.3f}')
        ax3.plot(x_ellipse, y_ellipse, 'purple', linewidth=2)
        
        # Minimum uncertainty circle
        x_circle = r_min * np.cos(theta)
        y_circle = r_min * np.sin(theta)
        ax3.plot(x_circle, y_circle, 'r--', linewidth=2,
                label=f'Minimum: ℏ/2 = {hbar/2:.3f}')
        
        # Check if uncertainty principle is satisfied
        if uncertainty_product >= hbar/2:
            status = "✓ Satisfied"
            color = 'green'
        else:
            status = "✗ Violated"
            color = 'red'
        
        ax3.text(0.5, 0.95, f'Uncertainty Principle: {status}',
                transform=ax3.transAxes, fontsize=14,
                horizontalalignment='center', verticalalignment='top',
                color=color, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('Δx', fontsize=12)
        ax3.set_ylabel('Δp', fontsize=12)
        ax3.set_title('Uncertainty Product', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.axis('equal')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_quantum_tunneling(self, x: np.ndarray, V: np.ndarray,
                             E: float, psi: np.ndarray,
                             title: str = "Quantum Tunneling") -> plt.Figure:
        """
        Visualize quantum tunneling through a potential barrier.
        
        Args:
            x: Position array
            V: Potential energy array
            E: Particle energy
            psi: Wavefunction
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Top plot: Potential and energy
        ax1.fill_between(x, 0, V, alpha=0.3, color='gray', label='Potential V(x)')
        ax1.plot(x, V, 'k-', linewidth=2)
        ax1.axhline(y=E, color='red', linestyle='--', linewidth=2, label=f'Energy E = {E:.2f}')
        
        # Mark classically forbidden regions
        forbidden = V > E
        if np.any(forbidden):
            ax1.fill_between(x, E, V, where=forbidden, alpha=0.2, color='red',
                           label='Classically Forbidden')
        
        ax1.set_ylabel('Energy', fontsize=12)
        ax1.set_title(title, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, max(V.max(), E) * 1.1)
        
        # Bottom plot: Wavefunction with phase
        prob = np.abs(psi)**2
        phase = np.angle(psi)
        phase_norm = (phase + np.pi) / (2 * np.pi)
        
        # Create phase-colored probability plot
        from matplotlib.collections import LineCollection
        points = np.array([x, prob]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, cmap=self.phase_cmap, linewidth=3)
        lc.set_array(phase_norm)
        line = ax2.add_collection(lc)
        
        # Show real and imaginary parts
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x, np.real(psi), 'r-', linewidth=1, alpha=0.5, label='Re(ψ)')
        ax2_twin.plot(x, np.imag(psi), 'b-', linewidth=1, alpha=0.5, label='Im(ψ)')
        
        # Highlight tunneling region
        if np.any(forbidden):
            for region in self._find_continuous_regions(forbidden):
                ax2.axvspan(x[region[0]], x[region[-1]], alpha=0.1, color='red')
        
        ax2.set_xlabel('Position (x)', fontsize=12)
        ax2.set_ylabel('|ψ|²', fontsize=12)
        ax2_twin.set_ylabel('ψ', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(x.min(), x.max())
        ax2.set_ylim(0, prob.max() * 1.1)
        
        # Add colorbar for phase
        cbar = plt.colorbar(line, ax=ax2, orientation='horizontal', pad=0.15)
        cbar.set_label('Phase', fontsize=10)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        # Add legends
        ax2_twin.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def _find_continuous_regions(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find continuous True regions in a boolean mask."""
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append(np.arange(start, i))
                in_region = False
        
        if in_region:
            regions.append(np.arange(start, len(mask)))
        
        return regions 