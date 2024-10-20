#!/usr/bin/env python3
import os
import shutil
import simulate_cameras
import generate_patterns
import generate_plots

if __name__ == "__main__":
    # Clean run directory
    path = "./run"
    # if os.path.isdir(path):
    #     shutil.rmtree(path)
    # Create run directory
    os.makedirs(path, exist_ok=True)
    generate_patterns.run()
    simulate_cameras.run()
    generate_plots.run()
