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
    def __init__(self, base_dir=None):
        if base_dir is None:
            # Use absolute path relative to this file's location
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'particles')
        self.base_dir = base_dir
        self.particles: Dict[str, Particle] = {}
        self.discover_particles()

    def discover_particles(self):
        defaults_dir = os.path.join(self.base_dir, 'defaults')
        if os.path.isdir(defaults_dir):
            particle_files = [f for f in os.listdir(defaults_dir) if f.endswith('.json')]
            # print(f"Found {len(particle_files)} particle files in {defaults_dir}")
            for filename in particle_files:
                filepath = os.path.join(defaults_dir, filename)
                self._load_particle(filepath)
            # Debug: print loaded particles
            if len(self.particles) == 0:
                print(f"Warning: No particles loaded from {defaults_dir}")
                print(f"  Files found: {particle_files}")
        else:
            print(f"Warning: Particle defaults directory not found: {defaults_dir}")
            print(f"  Base dir: {self.base_dir}")
            print(f"  Current working directory: {os.getcwd()}")
    
    def _load_particle(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                particle = Particle(**data)
                self.particles[particle.name.lower()] = particle
        except Exception as e:
            print(f"Error loading particle from {filepath}: {e}")
            
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