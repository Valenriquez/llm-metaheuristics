 # Name: PSOGravitationalSearchOptimizer
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'metropolis'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This script defines a metaheuristic optimization problem using the Gravitational Search Optimizer (GSO) with parameters for gravity and alpha, and a Random Flight operator with scale, distribution type, and beta. The GSO is chosen based on its gravitational force parameter which influences the exploration of search space, while the Random Flight uses a probabilistic approach to escape local minima by exploring different directions in the search space. These operators are selected using 'metropolis' selector for probabilistic selection.
# PSOGravitationalSearchOptimizer combines aspects of both Gravitational Search and Random Flight strategies to leverage the strengths of each method in optimizing complex, multi-modal functions such as those found in benchmark problems like the Sphere function used here. The combination allows for a balance between exploration (GSO) and exploitation (Random Flight), which is expected to yield better convergence properties compared to using either technique alone.