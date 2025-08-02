#!/usr/bin/env python3
"""Debug the 3D viewer by creating a test with all particles."""

import os
import sys
import torch
import json

sys.path.insert(0, '.')

from physics_agent.ui.multi_particle_trajectory_viewer_generator import (
    generate_multi_particle_trajectory_viewer,
    extract_trajectory_data
)

def create_test_viewer():
    """Create a test viewer with all particle data."""
    
    cache_dir = 'physics_agent/cache/trajectories/1.0.0/Primordial_Mini_Black_Hole'
    particle_data = {}
    
    # Define test trajectories to load
    test_files = {
        'electron': 'Schwarzschild_53712288c4a7e7b9_steps_1000.pt',
        'neutrino': 'Schwarzschild_12c9bd24f5b7a4f2_steps_1000.pt', 
        'photon': 'Schwarzschild_0c891ee2c3b1e764_steps_1000.pt',
        'proton': 'Schwarzschild_8c08fdf8bc0dc306_steps_1000.pt'
    }
    
    # Try to find any Schwarzschild trajectory
    all_files = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
    schwarzschild_files = [f for f in all_files if f.startswith('Schwarzschild') and f.endswith('.pt')]
    
    print(f"Found {len(schwarzschild_files)} Schwarzschild cache files")
    
    if schwarzschild_files:
        # Use the first file for all particles as a test
        test_file = os.path.join(cache_dir, schwarzschild_files[0])
        print(f"Using test file: {test_file}")
        
        try:
            data = torch.load(test_file, weights_only=True)
            if isinstance(data, dict) and 'trajectory' in data:
                traj = data['trajectory']
            else:
                traj = data
            
            print(f"Loaded trajectory shape: {traj.shape}")
            
            # Extract trajectory data for each particle
            extracted = extract_trajectory_data(traj)
            print(f"Extracted data keys: {extracted.keys()}")
            print(f"First r values: {extracted['r'][:5] if len(extracted['r']) > 5 else extracted['r']}")
            
            # Assign to all particles
            for particle in ['electron', 'neutrino', 'photon', 'proton']:
                particle_data[particle] = {'theory': extracted}
                
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate debug output
    print("\nParticle data structure:")
    for particle, data in particle_data.items():
        if 'theory' in data:
            theory_data = data['theory']
            print(f"  {particle}:")
            print(f"    - r length: {len(theory_data.get('r', []))}")
            print(f"    - theta length: {len(theory_data.get('theta', []))}")
            print(f"    - phi length: {len(theory_data.get('phi', []))}")
    
    # Save particle data for inspection
    with open('test_particle_data.json', 'w') as f:
        # Convert tensors to lists for JSON serialization
        json_data = {}
        for particle, pdata in particle_data.items():
            if 'theory' in pdata:
                json_data[particle] = {
                    'theory': {
                        'r': pdata['theory']['r'][:10],  # First 10 points
                        'theta': pdata['theory']['theta'][:10],
                        'phi': pdata['theory']['phi'][:10]
                    }
                }
        json.dump(json_data, f, indent=2)
    print("\nSaved sample data to test_particle_data.json")
    
    # Generate viewer
    output_path = 'test_3d_viewer_debug.html'
    generate_multi_particle_trajectory_viewer(
        'Schwarzschild Debug Test',
        particle_data,
        9.945e13,  # Primordial mini black hole mass
        output_path
    )
    
    print(f"\nGenerated: {output_path}")
    
    # Also create a minimal test HTML to verify Three.js is working
    create_minimal_test()

def create_minimal_test():
    """Create a minimal Three.js test to ensure basics are working."""
    
    minimal_html = """<!DOCTYPE html>
<html>
<head>
    <title>Minimal Three.js Test</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body { margin: 0; overflow: hidden; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: Arial;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="info">
        Minimal Three.js Test<br>
        You should see a rotating cube
    </div>
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x202020);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;
        
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Add cube
        const geometry = new THREE.BoxGeometry();
        const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        const cube = new THREE.Mesh(geometry, material);
        scene.add(cube);
        
        // Add light
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(1, 1, 1);
        scene.add(light);
        
        // Animation
        function animate() {
            requestAnimationFrame(animate);
            cube.rotation.x += 0.01;
            cube.rotation.y += 0.01;
            renderer.render(scene, camera);
        }
        
        animate();
        
        // Log to console
        console.log('Three.js version:', THREE.REVISION);
        console.log('WebGL supported:', renderer.capabilities.isWebGL2);
    </script>
</body>
</html>"""
    
    with open('minimal_threejs_test.html', 'w') as f:
        f.write(minimal_html)
    
    print("Created minimal_threejs_test.html")

if __name__ == "__main__":
    create_test_viewer()