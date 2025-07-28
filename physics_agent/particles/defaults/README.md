# Particle Defaults Configuration

This directory contains JSON configuration files for standard particles used in gravitational theory simulations. Each file defines a particle's fundamental properties and orbital dynamics parameters.

## Particle Properties

Each particle JSON file contains:

### Basic Properties
- `name`: Particle name (e.g., "Electron", "Photon")
- `particle_type`: Either "massive" or "massless"
- `mass`: Rest mass in kg (0.0 for massless particles)
- `charge`: Electric charge in Coulombs
- `spin`: Spin quantum number

### Orbital Parameters (New)
The `orbital_parameters` section controls how particles behave in gravitational fields, particularly near black holes:

- `angular_velocity_factor`: Multiplier for angular velocity (u_φ)
  - < 1.0: Slower angular motion (more radial trajectory)
  - > 1.0: Faster angular motion (more tangential trajectory)
  
- `radial_velocity_factor`: Multiplier for radial velocity (u_r)
  - Negative values: Inward motion toward black hole
  - Positive values: Outward motion away from black hole
  - Magnitude controls speed (typically -1.0 to 0.0)
  
- `orbit_type`: Descriptive name for the orbit type
  - `"circular"`: Near-circular orbit (small radial velocity)
  - `"elliptical_precessing"`: Elliptical orbit with relativistic precession
  - `"plunging"`: Deep infall trajectory
  - `"hyperbolic_flyby"`: Fast close encounter
  - `"gravitational_lensing"`: Light deflection trajectory
  
- `photon_sphere_factor` (photons only): Additional factor for photon trajectories near the photon sphere

## Examples

### Electron (elliptical_precessing)
```json
"orbital_parameters": {
    "angular_velocity_factor": 0.8,    // Reduced for precession
    "radial_velocity_factor": -0.5,     // Moderate inward motion
    "orbit_type": "elliptical_precessing"
}
```

### Photon (gravitational_lensing)
```json
"orbital_parameters": {
    "angular_velocity_factor": 0.3,     // Low for maximum deflection
    "radial_velocity_factor": -0.9,     // Nearly radial infall
    "orbit_type": "gravitational_lensing",
    "photon_sphere_factor": 0.4         // Photon-specific parameter
}
```

### Proton (plunging)
```json
"orbital_parameters": {
    "angular_velocity_factor": 0.6,     // Slow rotation for plunge
    "radial_velocity_factor": -0.8,     // Strong inward velocity
    "orbit_type": "plunging"
}
```

### Neutrino (hyperbolic_flyby)
```json
"orbital_parameters": {
    "angular_velocity_factor": 1.5,     // Fast angular motion
    "radial_velocity_factor": -1.0,     // Maximum inward speed
    "orbit_type": "hyperbolic_flyby"
}
```

## Customization

To modify particle behavior:

1. Edit the corresponding JSON file
2. Adjust the `angular_velocity_factor` and `radial_velocity_factor`
3. The changes will be automatically loaded on next run

## Physical Interpretation

The orbital parameters create different types of geodesics in curved spacetime:

- **Bound Orbits**: Small negative radial velocity, moderate angular velocity
- **Plunging Orbits**: Large negative radial velocity, reduced angular velocity  
- **Scattering Orbits**: High angular velocity, varying radial velocity
- **Light Bending**: For photons, low angular momentum maximizes deflection

These parameters allow visualization of diverse particle behaviors near black holes without modifying core engine code. 