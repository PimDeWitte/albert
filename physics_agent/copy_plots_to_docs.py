#!/usr/bin/env python3
"""
Copy validator plots from physics_agent/latest_run to docs/latest_run
This ensures plots are always available for the landing page
"""
import os
import shutil
from datetime import datetime

def copy_plots_to_docs():
    """Copy all validator plots to docs directory"""
    source_dir = "physics_agent/latest_run"
    docs_dir = "docs/latest_run"
    
    # Create docs directories if they don't exist
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(f"{docs_dir}/validator_plots", exist_ok=True)
    os.makedirs(f"{docs_dir}/trajectory_viewers", exist_ok=True)
    
    copied_files = []
    
    # Copy conservation plots
    conservation_plots = [f for f in os.listdir(source_dir) if f.startswith('conservation_') and f.endswith('.png')]
    if conservation_plots:
        # Copy the most recent one as the main example
        latest_conservation = sorted(conservation_plots)[-1]
        shutil.copy2(f"{source_dir}/{latest_conservation}", f"{docs_dir}/energy_conservation_plot.png")
        copied_files.append("energy_conservation_plot.png")
        
        # Also copy to validator_plots
        shutil.copy2(f"{source_dir}/{latest_conservation}", f"{docs_dir}/validator_plots/conservation.png")
        copied_files.append("validator_plots/conservation.png")
    
    # Copy Mercury orbit plots
    mercury_plots = [f for f in os.listdir(source_dir) if f.startswith('mercury_orbit_') and f.endswith('.png')]
    if mercury_plots:
        latest_mercury = sorted(mercury_plots)[-1]
        shutil.copy2(f"{source_dir}/{latest_mercury}", f"{docs_dir}/trajectory_viewers/schwarzschild_orbit.png")
        shutil.copy2(f"{source_dir}/{latest_mercury}", f"{docs_dir}/validator_plots/mercury_orbit.png")
        copied_files.extend(["trajectory_viewers/schwarzschild_orbit.png", "validator_plots/mercury_orbit.png"])
    
    # Copy light deflection plots
    light_plots = [f for f in os.listdir(source_dir) if f.startswith('light_deflection_') and f.endswith('.png')]
    if light_plots:
        latest_light = sorted(light_plots)[-1]
        shutil.copy2(f"{source_dir}/{latest_light}", f"{docs_dir}/light_bending_visualization.png")
        shutil.copy2(f"{source_dir}/{latest_light}", f"{docs_dir}/validator_plots/light_deflection.png")
        copied_files.extend(["light_bending_visualization.png", "validator_plots/light_deflection.png"])
    
    # Copy any GW spectrum plots (placeholder for future)
    gw_plots = [f for f in os.listdir(source_dir) if f.startswith('gw_spectrum') and f.endswith('.png')]
    if gw_plots:
        latest_gw = sorted(gw_plots)[-1]
        shutil.copy2(f"{source_dir}/{latest_gw}", f"{docs_dir}/gw_spectrum.png")
        copied_files.append("gw_spectrum.png")
    
    # Create a timestamp file
    with open(f"{docs_dir}/last_plot_update.txt", 'w') as f:
        f.write(f"Plots last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Files copied:\n")
        for file in copied_files:
            f.write(f"  - {file}\n")
    
    return copied_files

if __name__ == "__main__":
    print("Copying validator plots to docs directory...")
    copied = copy_plots_to_docs()
    print(f"✅ Copied {len(copied)} plot files to docs/latest_run/")
    for file in copied:
        print(f"  - {file}")