 # Name: Adaptive Random Search Algorithm (ARSA)
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
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (  
        'random_sample',
        {},
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# The Adaptive Random Search Algorithm (ARSA) is a metaheuristic that combines two main components: local random walk and probabilistic selection. 
# Local random walk, with its adjustable probability and scale parameters, allows the algorithm to explore different regions of the search space by adjusting its step sizes according to a Gaussian distribution. This helps in escaping local minima and exploring new areas where better solutions might be found.
# Probabilistic selection is used in this case as part of the 'probabilistic' selector, which means that the algorithm will follow a probabilistic approach to decide whether to accept or reject a candidate solution based on some probability threshold. This helps balance between exploration (trying out new solutions) and exploitation (improving upon current best solutions).
# The choice of using local random walk with Gaussian distribution is justified because it allows for both fine-grained exploration close to the current position, as well as broader jumps across the search space. This adaptability in step sizes helps in efficiently navigating the landscape of the Rastrigin function, which has many local minima and a single global minimum.