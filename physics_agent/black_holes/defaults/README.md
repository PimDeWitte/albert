# Default Black Hole Configurations

This directory contains JSON configuration files for various black hole presets used in gravitational theory simulations.

## Available Presets

### `stellar_mass.json`
- **Mass**: 10 solar masses
- **Schwarzschild radius**: 29.5 km
- **Description**: Typical stellar-mass black hole from core collapse
- **Use case**: Standard astrophysical simulations

### `primordial_mini.json` (Recommended Default)
- **Mass**: 10^15 kg (asteroid mass, ~5×10^-16 solar masses)
- **Schwarzschild radius**: 1.5 pm (smaller than a proton!)
- **Description**: Hypothetical primordial black hole
- **Use case**: Numerically stable testing, quantum gravity effects

### `laboratory_micro.json`
- **Mass**: 10^8 kg (~5×10^-23 solar masses)
- **Schwarzschild radius**: 1.5×10^-19 m
- **Description**: Theoretical micro black hole
- **Use case**: Extreme quantum gravity regime

### `intermediate_mass.json`
- **Mass**: 1000 solar masses
- **Schwarzschild radius**: 2953 km
- **Description**: Gap between stellar and supermassive
- **Use case**: Globular cluster dynamics

### `sagittarius_a_star.json`
- **Mass**: 4.15×10^6 solar masses
- **Schwarzschild radius**: 1.2×10^10 m
- **Description**: Milky Way's central black hole
- **Use case**: Galactic center physics

## JSON Structure

Each black hole configuration contains:

```json
{
    "name": "Human-readable name",
    "mass_kg": "Mass in kilograms",
    "mass_solar": "Mass in solar masses",
    "description": "Brief description",
    "schwarzschild_radius_m": "Schwarzschild radius in meters",
    "typical_orbits": {
        "ISCO": "Innermost Stable Circular Orbit radius (in M)",
        "photon_sphere": "Photon sphere radius (in M)",
        "stable_circular": "Array of typical stable orbit radii (in M)"
    },
    "integration_parameters": {
        "dtau_geometric": "Recommended timestep in geometric units",
        "max_steps": "Maximum integration steps",
        "r_min": "Minimum radius for integration (in M)",
        "r_max": "Maximum radius for integration (in M)"
    },
    "notes": "Additional notes or context"
}
```

## Units

- **Geometric units**: In numerical simulations, we use G=c=M=1
- **M**: Geometric unit of length = GM/c²
- **Orbital radii**: Expressed in units of M (e.g., ISCO at 6M)
- **Time**: Geometric unit = GM/c³

## Adding New Black Holes

To add a new black hole configuration:

1. Create a new JSON file in this directory
2. Follow the structure above
3. Ensure mass conversions are accurate
4. Set appropriate integration parameters based on the mass scale
5. The black hole will automatically be available in the loader

## Integration Guidelines

- **Stellar mass and larger**: Use `dtau_geometric = 0.01`
- **Mini black holes (10^15 kg)**: Use `dtau_geometric = 0.1`
- **Micro black holes (<10^10 kg)**: Use `dtau_geometric = 1.0`

Smaller black holes allow larger timesteps due to the scaling of geometric units.