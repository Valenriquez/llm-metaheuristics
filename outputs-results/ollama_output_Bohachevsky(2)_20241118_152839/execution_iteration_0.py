# Name: Dingo-Bees Metaheuristic
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Zakharov(3)  # Changed dimension to 3 for larger space coverage.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial'
        },
        'all'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))


# Short explanation and justification:
# The Dingo-Bees Metaheuristic combines two search operators: swarm_dynamic and spiral_dynamic.
# This combination aims to explore the complex search space more effectively, as demonstrated by increasing dimensionality from 2D to 3D in the problem formulation.

# Swarm Dynamic is selected for its inertial behavior that can help escape local optima,
# while Spiral Dynamic helps in systematically exploring potential solutions around the current best point.
# The selection of 'all' for selector_name allows these operators to dynamically adjust their search strategies based on progress and landscape features.


# In case there was an error, please fix it. This is the error: .