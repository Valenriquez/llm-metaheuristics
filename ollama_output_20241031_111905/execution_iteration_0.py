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
        'distribution': 'levy'
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

# Explanation and Justification:
# This metaheuristic combines two operators for exploration and exploitation. 
# The first operator, 'local_random_walk', is designed to perform a local random walk with a specified probability of movement (0.75) and uses a levy distribution for the step sizes. This helps in exploring different regions of the search space more effectively than purely random or Gaussian distributed steps.
# The second operator, 'random_sample', selects candidates probabilistically ('probabilistic' selector). This allows for a balance between exploration (trying new solutions) and exploitation (improving upon existing ones), which is crucial for optimization problems like the Rastrigin function that may have multiple local minima.
# Both operators are configured to work in harmony, with the 'local_random_walk' enhancing the search by allowing more directed movement within the vicinity of current solutions, while 'random_sample' introduces diversity by selecting candidates probabilistically, thus avoiding premature convergence to a local minimum. This combination is expected to yield better overall performance compared to using either operator alone, especially in complex multi-modal functions like the Rastrigin function where multiple minima exist.