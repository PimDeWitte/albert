#!/usr/bin/env python3
"""
Manually update homepage images with latest run results.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from physics_agent.update_homepage_images import update_homepage_after_run, get_latest_run_dir, update_homepage_images

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use specified run directory
        run_dir = sys.argv[1]
        if os.path.exists(run_dir):
            print(f"Updating homepage with images from: {run_dir}")
            update_homepage_images(run_dir)
        else:
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)
    else:
        # Use latest run
        update_homepage_after_run() 