Code used for my stereo camera simulator built for my undergraduate thesis.

# Usage
Can be run/tested by calling ```./run.py``` in the base directory. Other scripts aren't shebanged nor have their executable flag set so require Python to run directly, e.g. ```python ./simulate_cameras.py```

Simulations are controlled via the sims.csv and positions.csv in the assets directory. 

# Dependencies
Nix support, shell with all dependencies can be had using ```nix develop```

Otherwise requires Python 3.11 with the following Python packages:
- glcontext
- jinja2
- matplotlib
- moderngl
- numpy
- opencv4
- pillow
- pycairo
- pyrr
- sympy
