# Unified Multi-Particle Viewer Documentation

## Overview

The Unified Multi-Particle Viewer is an advanced 3D visualization tool that allows you to explore gravitational trajectories from all theories in a single, interactive interface. It provides scientifically accurate spacetime visualization using Flamm's paraboloid embedding of the Schwarzschild metric.

## Features

### 1. Unified Theory Selection
- **Single viewer for all theories** - No need to open multiple viewers
- **Dropdown selection** - Easily switch between theories
- **Automatic data loading** - Trajectories load instantly when switching theories
- **Memory management** - Previous trajectories are properly cleaned up

### 2. Scientifically Accurate Spacetime Grid
- **Flamm's Paraboloid Embedding** - Visualizes the intrinsic curvature of spacetime
  - Formula: `z = -2√(2M(r-2M))` scaled for visibility
- **Key Radius Markers**:
  - Red circle at 2M (Event Horizon)
  - Orange circle at 3M (Photon Sphere)
  - Yellow circle at 6M (ISCO - Innermost Stable Circular Orbit)
- **Coordinate Labels** - Shows radial distances (10M, 20M, 50M, 100M)
- **Static Grid** - Represents intrinsic spacetime curvature (not observer-dependent)

### 3. Advanced Camera Controls
- **Third-Person View** - Default camera position for overview
- **Top-Down View** - Bird's eye view of the orbital plane
- **Particle Mounting** - Attach camera to any particle for first-person perspective
- **Smart Focus** - Camera always centered on black hole
- **Adjustable Distance** - Zoom in/out with proper clipping plane management

### 4. Enhanced Visualization
- **Realistic Scales**:
  - Smaller particle spheres (0.5 unit radius)
  - Thinner trajectory tubes (0.3 unit radius)
  - Properly scaled black hole representation
- **Visual Effects**:
  - Particle glow for better visibility
  - Black hole event horizon glow
  - Semi-transparent spacetime grid
- **Trajectories on Curved Space** - All particles and paths follow the curved embedding

### 5. Live Information Panels
- **Spacetime Info**:
  - Camera distance from black hole
  - Time dilation factor
  - Spatial curvature strength
  - Scientific description with percentages
- **Particle States**:
  - Position and velocity in geometric units
  - Tensor properties (when mounted)
  - Degrees of freedom information

### 6. Animation Controls
- **Play/Pause** - Control trajectory animation
- **Speed Control** - Adjust animation speed (0.1x to 10x)
- **Time Slider** - Jump to any point in the trajectory
- **Auto-Reset** - Animation resets when switching theories

## Usage

### Basic Navigation
1. **Select Theory**: Use the dropdown at the top to choose a theory
2. **View Particles**: All four particles (electron, proton, neutrino, photon) are displayed
3. **Rotate View**: Click and drag to rotate around the black hole
4. **Zoom**: Scroll wheel to zoom in/out

### Camera Modes
- **Reset Camera**: Returns to default third-person view
- **Top View**: Switches to bird's eye perspective
- **Mount Camera**: Click "Mount Camera" on any particle for first-person view

### Understanding the Visualization
- **Green Grid**: Represents the curved spacetime around the black hole
- **Grid Depression**: Shows how mass warps spacetime (deeper near black hole)
- **Colored Circles**: Mark important orbital radii in general relativity
- **Particle Paths**: Show how different particles behave in curved spacetime

## Technical Details

### Coordinate System
- Uses **geometric units** where G = c = 1
- Distances measured in multiples of M (gravitational radius)
- 1M = GM/c² in SI units

### Spacetime Embedding
The visualization uses Flamm's paraboloid, which is a 2D slice of the Schwarzschild metric:
- Represents spatial curvature at a fixed time
- Height represents the "stretching" of space near the black hole
- Not a representation of spacetime in 4D (which cannot be directly visualized)

### File Generation
The viewer is automatically generated during comprehensive tests:
```bash
albert run  # Runs comprehensive test which generates the viewer
```

Or manually via Python:
```python
from physics_agent.ui.renderer import generate_unified_multi_particle_viewer

generate_unified_multi_particle_viewer(
    run_dir='physics_agent/runs/comprehensive_test_YYYYMMDD_HHMMSS',
    output_path='trajectory_viewers/unified_multi_particle_viewer_advanced.html'
)
```

## Browser Requirements
- Modern browser with WebGL support
- Recommended: Chrome, Firefox, or Safari (latest versions)
- GPU acceleration enabled for best performance

## Known Limitations
- Maximum of ~30 theories can be loaded before performance degrades
- Particle tensor information panel not fully implemented
- Observer-dependent effects (like gravitational time dilation from camera position) simplified for clarity