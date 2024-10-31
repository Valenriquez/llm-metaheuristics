 # Name: Adaptive Random Search (ARS)
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
# The Adaptive Random Search (ARS) metaheuristic is designed to balance exploration and exploitation by integrating two main operators: local random walk and spiral dynamic search. 
# Local random walk, with a probability of 0.75 and uniform distribution for its movement, helps in fine-tuning the solution within the current vicinity, ensuring that the algorithm does not get stuck in local minima. 
# Spiral dynamic search, on the other hand, introduces a spiral pattern to explore new areas by adjusting both radius and angle parameters dynamically. This operator is set with probabilistic selection to balance between exploration and exploitation effectively.
# Both operators are carefully chosen based on their ability to handle different aspects of optimization: local random walk for neighborhood exploration and spiral dynamic search for broader area exploration, making the metaheuristic adaptable to various problem landscapes.