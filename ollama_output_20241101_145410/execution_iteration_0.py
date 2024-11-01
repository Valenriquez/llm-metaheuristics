 # Name: Ant Colony Optimization (ACO)
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
        'radius': 0.9,
        'angle': 22.5,
        'sigma': 0.1
    },
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is Ant Colony Optimization (ACO), which is inspired by the behavior of ants in finding food sources. 
# In this implementation, we use two operators: 'local_random_walk' for exploring local areas with a probability of 0.75 and scale of 1.0, 
# distributed uniformly; and 'spiral_dynamic' to simulate spirals with radius 0.9, angle 22.5 degrees, and sigma 0.1. 
# The 'local_random_walk' uses the Metropolis selector for probabilistic exploration, while 'spiral_dynamic' employs a probabilistic selector 
# to guide its dynamic spiral search behavior. This combination aims to balance between local exploitation and global exploration, 
# leveraging both strategies within their respective strengths in different regions of the solution space.