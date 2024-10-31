 # Name: Local Random Walk with Metropolis Selection
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
            'selector': 'greedy'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic is a combination of the local random walk operator with Metropolis selection and a greedy selection from random sample operators. 
# The local random walk operator is configured to use a probability of 0.75 for accepting new solutions, with a scale factor of 1.0 and a uniform distribution for exploration. 
# The Metropolis selector allows the algorithm to decide whether to accept or reject based on predefined criteria, which in this case is set to 'metropolis'.
# The random sample operator uses a greedy selector, aiming to select the best candidates from the population according to their fitness values.
# This combination aims to balance between exploration and exploitation, ensuring that the algorithm can explore new areas while also refining solutions based on the current best found. 
# By using both operators with different selection methods, we aim to leverage the strengths of each approach in optimizing complex objective functions like those provided by benchmark_func.