#!/usr/bin/env python3
"""
Update homepage images with latest run results.
"""
import os
import glob
import re
from typing import List, Dict, Tuple


def get_latest_run_dir() -> str:
    """Find the most recent run directory."""
    run_dirs = glob.glob("runs/run_*")
    if not run_dirs:
        return None
    
    # Sort by timestamp in directory name
    run_dirs.sort(key=lambda x: os.path.basename(x).split('_')[1:3])
    return run_dirs[-1]


def find_trajectory_images(run_dir: str) -> Dict[str, List[str]]:
    """Find all trajectory comparison images in a run directory.
    
    Returns:
        Dict mapping theory names to list of trajectory image paths
    """
    trajectory_images = {}
    
    # Find all theory directories
    theory_dirs = glob.glob(os.path.join(run_dir, "*"))
    
    for theory_dir in theory_dirs:
        if not os.path.isdir(theory_dir):
            continue
            
        theory_name = os.path.basename(theory_dir)
        
        # Skip special directories
        if theory_name in ['fail', 'predictions']:
            continue
            
        # Find trajectory comparison images - check both possible directories
        for viz_dirname in ["visualizations", "viz"]:
            viz_dir = os.path.join(theory_dir, viz_dirname)
            if os.path.exists(viz_dir):
                # New format: trajectory_comparison_*.png
                images = glob.glob(os.path.join(viz_dir, "trajectory_comparison_*.png"))
                if images:
                    trajectory_images[theory_name] = images
                    break
                else:
                    # Fallback to old format
                    old_format = os.path.join(viz_dir, "trajectory_comparison.png")
                    if os.path.exists(old_format):
                        trajectory_images[theory_name] = [old_format]
                        break
    
    return trajectory_images


def update_homepage_images(run_dir: str, index_file: str = "docs/index.html"):
    """Update the homepage with images from the latest run.
    
    Args:
        run_dir: Path to the run directory
        index_file: Path to index.html file
    """
    # Read the current index.html
    with open(index_file, 'r') as f:
        content = f.read()
    
    # Get trajectory images
    trajectory_images = find_trajectory_images(run_dir)
    
    if not trajectory_images:
        print("No trajectory images found in run directory")
        return
    
    # Theory mapping for the homepage showcase
    # Maps display names to possible directory name patterns
    theory_showcase = [
        ("Einstein's Final Theory", ["Einstein_Asymmetric", "Einstein Asymmetric"]),
        ("Post-Quantum Gravity", ["Post-Quantum_Gravity", "Post-Quantum Gravity"]),
        ("Regularised Core QG", ["Regularised_Core_QG", "Regularised Core QG"]),
        ("Phase Transition QG", ["Phase_Transition_QG", "Phase Transition QG"]),
        ("Weyl-EM Quantum", ["Weyl-EM_Quantum", "Weyl-EM Quantum"]),
        ("Newtonian Limit", ["Newtonian_Limit", "Newtonian Limit"]),
        ("Quantum Corrected", ["Quantum_Corrected", "Quantum Corrected"]),  # Add quantum corrected
    ]
    
    # For each showcase theory, find and update its image
    updated_count = 0
    for display_name, patterns in theory_showcase:
        # Find matching theory directory
        matched_theory = None
        matched_images = None
        
        for theory_name, images in trajectory_images.items():
            for pattern in patterns:
                if pattern in theory_name:
                    matched_theory = theory_name
                    matched_images = images
                    break
            if matched_theory:
                break
        
        if matched_theory and matched_images:
            # Use the first image (typically electron)
            new_image_path = matched_images[0]
            
            # Convert to relative path from docs directory
            rel_path = os.path.relpath(new_image_path, "docs")
            
            # Try multiple patterns to find the image tag
            patterns_to_try = [
                # Pattern 1: img tag with alt text containing theory name
                rf'(<img\s+src=")[^"]+("\s+[^>]*alt="[^"]*{re.escape(pattern)}[^"]*"[^>]*>)',
                # Pattern 2: Look for specific theory display name in nearby text
                rf'(<h5[^>]*>[^<]*{re.escape(display_name)}[^<]*</h5>[\s\S]*?<img\s+src=")[^"]+(")',
            ]
            
            updated = False
            for regex_pattern in patterns_to_try:
                # Find all matches to see what we're working with
                matches = list(re.finditer(regex_pattern, content, flags=re.IGNORECASE | re.MULTILINE))
                
                if matches:
                    # Replace with local path
                    if '<img' in regex_pattern:
                        replacement = rf'\1{rel_path}\2'
                    else:
                        # For pattern 2, we need to preserve the h5 tag
                        replacement = lambda m: m.group(1) + rel_path + m.group(2)
                    
                    # Perform replacement
                    new_content = re.sub(regex_pattern, replacement, content, count=1, flags=re.IGNORECASE | re.MULTILINE)
                    
                    if new_content != content:
                        content = new_content
                        updated_count += 1
                        print(f"Updated image for {display_name}: {rel_path}")
                        updated = True
                        break
            
            if not updated:
                print(f"Could not update image for {display_name} - pattern not found")
    
    # Write updated content back
    if updated_count > 0:
        with open(index_file, 'w') as f:
            f.write(content)
        print(f"\nUpdated {updated_count} images on homepage")
    else:
        print("No images were updated - patterns may not match")


def update_homepage_after_run():
    """Convenience function to update homepage with latest run results."""
    latest_run = get_latest_run_dir()
    if latest_run:
        print(f"Updating homepage with images from: {latest_run}")
        update_homepage_images(latest_run)
    else:
        print("No run directories found")


if __name__ == "__main__":
    # Test update
    update_homepage_after_run() 