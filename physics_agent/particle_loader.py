import os
import json
from typing import Dict

class Particle:
    def __init__(self, name: str, particle_type: str, mass: float, charge: float, spin: float, 
                 orbital_parameters: dict = None, color: str = None):
        self.name = name
        self.particle_type = particle_type
        self.mass = mass
        self.charge = charge
        self.spin = spin
        self.color = color or 'white' # Default to white if no color is provided
        # Store orbital parameters for trajectory visualization
        self.orbital_parameters = orbital_parameters or {
            'angular_velocity_factor': 1.0,
            'radial_velocity_factor': 0.0,
            'orbit_type': 'circular',
            'description': 'Default circular orbit'
        }

class ParticleLoader:
    def __init__(self, base_dir='physics_agent/particles'):
        self.base_dir = base_dir
        self.particles: Dict[str, Particle] = {}
        self.discover_particles()

    def discover_particles(self):
        defaults_dir = os.path.join(self.base_dir, 'defaults')
        if os.path.isdir(defaults_dir):
            for filename in os.listdir(defaults_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(defaults_dir, filename)
                    self._load_particle(filepath)
    
    def _load_particle(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
            particle = Particle(**data)
            self.particles[particle.name.lower()] = particle
            
    def get_particle(self, name: str = None) -> Particle:
        if name is None:
            return Particle('default', 'massive', 1.0, 0.0, 0.0)
        particle = self.particles.get(name.lower())
        if particle is None:
            print(f"Warning: Particle '{name}' not found. Using default massive neutral particle.")
            return Particle('default', 'massive', 1.0, 0.0, 0.0)
        return particle
    
    def get_available_particles(self) -> list:
        """Return a list of all available particle names"""
        return sorted(list(self.particles.keys())) 