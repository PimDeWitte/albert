# Unified Multi-Particle Viewer - Advanced Features

## Overview
The new unified multi-particle viewer provides a single, comprehensive visualization for all theories in a run, with advanced spacetime visualization and interaction capabilities.

## Key Features Implemented

### 1. One Viewer Per Run with Theory Dropdown
- Single unified viewer for entire run instead of per-theory viewers
- Dropdown selector to switch between theories dynamically
- Theories grouped by category (baseline, classical, quantum)
- No page reload required when switching theories

### 2. Dynamic Spacetime Grid
- Real-time spacetime grid that updates based on camera position
- Grid distortion visualizes spacetime curvature effects
- Shows:
  - Distance from black hole
  - Proper time dilation factor
  - Spatial curvature coefficient
  - Contextual descriptions of spacetime regime

### 3. Camera Mounting to Particles
- "Make Primary" button - sets particle as reference for Schwarzschild radius
- "Mount Camera" button - attaches camera to follow particle
- When mounted, displays:
  - Live tensor components (position, velocity, 4-velocity)
  - Metric tensor values at particle location
  - Active degrees of freedom
  - Conservation quantities

### 4. Improved Geometry and Realism
- Smaller, more realistic particle sizes (0.05 M radius)
- Thinner accretion disk visualization
- Precise photon sphere indicator
- Less bulky trajectory lines with transparency
- Proper scaling for better measurement accuracy

### 5. Interactive Legends and Information
- **Hover Effects**: Shows tooltips with object information
- **Click Actions**: Displays detailed popup with:
  - Particle properties and classification
  - Black hole characteristics (mass, radii, ISCO)
  - Theory-specific information
- **Dynamic Legend**: Updates based on loaded particles
- **Object Info Panel**: Detailed information on click

## Technical Improvements

### Data Loading
- Efficiently loads all theories from a run directory
- Handles both new flat structure and legacy directory structure
- Preserves theory names with special characters and parameters

### Performance
- Single page load for all theories
- Efficient Three.js scene management
- Dynamic object creation/destruction when switching theories

### User Experience
- Smooth camera transitions
- Intuitive controls with clear labeling
- Responsive design with scrollable panels
- Professional dark theme with accent colors

## Usage

### For Users
1. Open the unified viewer from the comprehensive test report
2. Select a theory from the dropdown
3. Use mouse to orbit camera or mount to a particle
4. Toggle spacetime grid to visualize curvature
5. Hover over objects for quick info, click for details

### For Developers
```python
from physics_agent.ui.renderer import (
    generate_unified_multi_particle_viewer
)

# Generate unified viewer for a run
viewer_path = generate_unified_multi_particle_viewer(
    run_dir="path/to/run",
    output_path="output/unified_viewer.html",
    black_hole_mass=9.945e13  # kg
)
```

## Future Enhancements
- Add support for 3D trajectories (currently 2D projection)
- Include energy/momentum conservation plots
- Add time-synchronized multi-theory comparison mode
- Export trajectory data in various formats
- Add measurement tools for distances and angles