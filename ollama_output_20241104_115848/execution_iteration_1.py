 # Name: Custom Metaheuristic 1
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'local_random_walk',
    { 
        'probability': 0.75,
        'scale': 1.0,
        'distribution': 'uniform'
    },
    'metropolis'
    ),
    (  
    'spiral_dynamic',
    {
        "radius": 0.9,
        "angle": 22.5,
        "sigma": 0.1
    },
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This custom metaheuristic uses two operators: 'local_random_walk' and 'spiral_dynamic'. 
# The 'local_random_walk' operator is configured with a probability of 0.75, scale of 1.0, and distribution set to 'uniform'. 
# This operator will use the metropolis selector for probabilistic selection. 
# The 'spiral_dynamic' operator has specific parameters: radius of 0.9, angle of 22.5 degrees, and sigma of 0.1. 
# It uses the 'probabilistic' selector to guide its search. 
# These operators are chosen based on their ability to explore different regions of the solution space through diverse mechanisms, which should lead to more effective global optimization for the given benchmark function Rastrigin(2).