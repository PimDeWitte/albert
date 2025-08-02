# WebGL Trajectory Viewer

## Overview
Created an interactive WebGL-based trajectory visualization system that can be launched from the comprehensive test report. This allows step-by-step visualization of particle trajectories and accumulated loss.

## Features

### 1. Interactive WebGL Visualization
- **Real-time rendering** of particle trajectories using WebGL
- **2D projection** of 3D trajectories (r, φ plane)
- **Black hole visualization** with event horizon
- **Dual trajectory display**: Theory (blue) vs Kerr baseline (red)

### 2. Playback Controls
- **Play/Pause**: Animate through trajectory steps
- **Speed control**: Adjust playback speed (0.1x - 10x)
- **Step slider**: Scrub through any point in the trajectory
- **Reset button**: Return to the beginning
- **Trail toggle**: Show/hide trajectory history

### 3. Camera Controls
- **Mouse drag**: Pan the view
- **Mouse wheel**: Zoom in/out
- **Zoom slider**: Fine-tune zoom level

### 4. Real-time Information Display
- **Current step** and total steps
- **Time**: Physical time in seconds
- **Position**: Radial distance in units of M
- **Velocity**: Speed in units of c
- **Loss**: Per-step loss vs Kerr
- **Accumulated loss**: Total loss up to current step

### 5. Integration with Test Suite

#### Modified Files:
1. **`comprehensive_test_report_generator.py`**:
   - Added "View Trajectory" button to table
   - Generates viewer HTML for each theory
   - Extracts trajectory data from test results

2. **`test_comprehensive_final.py`**:
   - Stores actual trajectory data in test results
   - Includes both theory and Kerr baseline trajectories

3. **`trajectory_viewer.html`**:
   - WebGL-based viewer template
   - Responsive design with dark theme
   - Handles variable trajectory lengths and scales

4. **`trajectory_viewer_generator.py`**:
   - Generates viewer HTML with embedded data
   - Converts PyTorch tensors to JSON
   - Calculates per-step loss

## Usage

1. Run the comprehensive test:
```bash
python -m physics_agent.theory_engine_core
```

2. Open the generated report:
```bash
open theory_validation_comprehensive_[timestamp].html
```

3. Click "View Trajectory" button for any theory

4. In the viewer:
   - Click Play to animate the trajectory
   - Drag to pan, scroll to zoom
   - Use sliders for fine control
   - Watch loss accumulate in real-time

## Technical Details

### Trajectory Format
```python
# PyTorch tensor shape: [steps, features]
# features: [t, r, theta, phi, r_dot, theta_dot, phi_dot]
```

### Coordinate System
- Uses spherical coordinates (r, θ, φ)
- Projects to 2D using (r*cos(φ), r*sin(φ))
- Black hole at origin

### Loss Calculation
- Per-step MSE between theory and Kerr positions
- Accounts for spherical coordinate distances
- Visualized as accumulated loss over time

## Future Enhancements

1. **3D Visualization**: Full 3D WebGL rendering
2. **Cache Integration**: Direct loading from trajectory cache files
3. **Parameter Display**: Show theory parameters and black hole properties
4. **Multiple Particles**: Compare different particle types
5. **Export Options**: Save trajectory data or animations
6. **Performance Metrics**: Display solver timing information

## Example
The viewer shows:
- Black hole event horizon (dark circle)
- Theory trajectory (blue line/point)
- Kerr baseline (red line/point)
- Real-time position and loss information
- Smooth animation with adjustable speed

This provides researchers with an intuitive way to understand how different theories predict particle motion and where they deviate from General Relativity.