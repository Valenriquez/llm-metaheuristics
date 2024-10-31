 # Name: Adaptive Random Walk with Metropolis Selector
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
    'random_sample',
    {
        'selector': 'probabilistic'
    },
    'probabilistic'
)
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic uses an adaptive random walk for exploration with a Metropolis selector, 
# which is suitable for local search improvements by accepting or rejecting proposed moves based on a probability.
# Additionally, it employs a probabilistic selection strategy from the random sample operator to balance between exploration and exploitation.
# The Rastrigin function is chosen as the benchmark problem due to its multimodal nature, allowing multiple operators to demonstrate their effectiveness in handling complex landscapes.