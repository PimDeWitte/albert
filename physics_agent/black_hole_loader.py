"""
Black Hole Loader - Loads black hole configurations from JSON files.

Similar to ParticleLoader, but for black hole presets.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from physics_agent.constants import SOLAR_MASS, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT


@dataclass
class BlackHole:
    """Black hole configuration."""
    name: str
    mass_kg: float
    mass_solar: float
    schwarzschild_radius_m: float
    description: str
    typical_orbits: Dict[str, Any]
    integration_parameters: Dict[str, Any]
    notes: Optional[str] = None
    
    @property
    def length_scale(self) -> float:
        """GM/c^2 in meters."""
        return GRAVITATIONAL_CONSTANT * self.mass_kg / SPEED_OF_LIGHT**2
    
    @property
    def time_scale(self) -> float:
        """GM/c^3 in seconds."""
        return GRAVITATIONAL_CONSTANT * self.mass_kg / SPEED_OF_LIGHT**3
    
    @property
    def velocity_scale(self) -> float:
        """c in m/s."""
        return SPEED_OF_LIGHT


class BlackHoleLoader:
    """Loader for black hole configurations from JSON files."""
    
    def __init__(self, base_dir: str = 'physics_agent/black_holes'):
        """
        Initialize black hole loader.
        
        Args:
            base_dir: Base directory containing black hole JSON files
        """
        self.base_dir = base_dir
        self.black_holes: Dict[str, BlackHole] = {}
        self.discover_black_holes()
    
    def discover_black_holes(self):
        """Discover and load all black hole configurations."""
        defaults_dir = os.path.join(self.base_dir, 'defaults')
        if os.path.isdir(defaults_dir):
            for filename in os.listdir(defaults_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(defaults_dir, filename)
                    self._load_black_hole(filepath)
    
    def _load_black_hole(self, filepath: str):
        """Load a single black hole configuration from JSON."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Extract the key from filename (e.g., 'stellar_mass.json' -> 'stellar_mass')
                key = os.path.basename(filepath).replace('.json', '')
                
                # Create BlackHole object
                black_hole = BlackHole(
                    name=data['name'],
                    mass_kg=data['mass_kg'],
                    mass_solar=data['mass_solar'],
                    schwarzschild_radius_m=data['schwarzschild_radius_m'],
                    description=data['description'],
                    typical_orbits=data['typical_orbits'],
                    integration_parameters=data['integration_parameters'],
                    notes=data.get('notes')
                )
                
                self.black_holes[key] = black_hole
                
        except Exception as e:
            print(f"Warning: Failed to load black hole config from {filepath}: {e}")
    
    def get_black_hole(self, name: Optional[str] = None) -> BlackHole:
        """
        Get a black hole configuration by name.
        
        Args:
            name: Black hole preset name. If None, returns primordial_mini.
            
        Returns:
            BlackHole object
        """
        if name is None:
            name = 'primordial_mini'  # Default preset
        
        black_hole = self.black_holes.get(name)
        if black_hole is None:
            available = ', '.join(self.black_holes.keys())
            raise KeyError(f"Unknown black hole preset '{name}'. Available: {available}")
        
        return black_hole
    
    def list_black_holes(self) -> Dict[str, str]:
        """List all available black hole presets with descriptions."""
        return {
            key: bh.description 
            for key, bh in self.black_holes.items()
        }
    
    def get_available_black_holes(self) -> list:
        """Return a list of all available black hole names."""
        return sorted(list(self.black_holes.keys()))
    
    def create_custom(self, mass_kg: float, name: str = "Custom") -> BlackHole:
        """
        Create a custom black hole configuration.
        
        Args:
            mass_kg: Black hole mass in kg
            name: Name for the configuration
            
        Returns:
            BlackHole object
        """
        # Calculate derived quantities
        mass_solar = mass_kg / SOLAR_MASS
        rs_m = 2 * GRAVITATIONAL_CONSTANT * mass_kg / SPEED_OF_LIGHT**2
        
        # Determine appropriate integration parameters based on mass
        if mass_kg < 1e10:  # Micro black hole
            dtau = 1.0
            max_steps = 1000000
        elif mass_kg < 1e20:  # Mini black hole
            dtau = 0.1
            max_steps = 100000
        else:  # Stellar or larger
            dtau = 0.01
            max_steps = 10000
        
        return BlackHole(
            name=name,
            mass_kg=mass_kg,
            mass_solar=mass_solar,
            schwarzschild_radius_m=rs_m,
            description=f"Custom black hole with mass {mass_solar:.2e} solar masses",
            typical_orbits={
                "ISCO": 6.0,
                "photon_sphere": 3.0,
                "stable_circular": [10.0, 20.0, 50.0, 100.0]
            },
            integration_parameters={
                "dtau_geometric": dtau,
                "max_steps": max_steps,
                "r_min": 2.1,
                "r_max": 1000.0
            },
            notes="Custom configuration"
        )


# Example usage and testing
if __name__ == "__main__":
    # Load black holes
    loader = BlackHoleLoader()
    
    # List available black holes
    print("Available black hole presets:")
    for key, description in loader.list_black_holes().items():
        bh = loader.get_black_hole(key)
        print(f"  {key:20} - {bh.mass_solar:>10.2e} M☉ - {description}")
    
    # Get default preset
    default = loader.get_black_hole()
    print(f"\nDefault preset: {default.name}")
    print(f"  Mass: {default.mass_solar:.2e} solar masses")
    print(f"  Schwarzschild radius: {default.schwarzschild_radius_m:.2e} m")
    print(f"  Recommended dtau: {default.integration_parameters['dtau_geometric']}")
    
    # Create custom black hole
    print("\nCreating custom black hole (Moon mass):")
    moon_mass_bh = loader.create_custom(7.342e22, "Moon Mass Black Hole")
    print(f"  Mass: {moon_mass_bh.mass_solar:.2e} solar masses")
    print(f"  Schwarzschild radius: {moon_mass_bh.schwarzschild_radius_m:.2e} m")
    print(f"  Length scale: {moon_mass_bh.length_scale:.2e} m")
    print(f"  Time scale: {moon_mass_bh.time_scale:.2e} s")