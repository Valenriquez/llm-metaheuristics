 # Name: GravitationalSearchOptimization
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'all'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'probabilistic'
    ),
    ( # Search operator 3
    'local_random_walk',
    {
        'probability': 0.75,
        'scale': 1.0,
        'distribution': 'uniform'
    },
    'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code implements a Gravitational Search Optimization (GSO) metaheuristic with three main operators: gravitational_search, random_flight, and local_random_walk. The Gravitational Search operator is configured with parameters gravity set to 1.0 and alpha to 0.02. It operates on all possible solutions in the population.
# The Random Flight operator uses a scale of 1.0 and follows a levy distribution with beta set to 1.5, employing a probabilistic selector among greedy, all, metropolis, or probabilistic operators based on the problem's characteristics.
# Local Random Walk is configured with a probability of 0.75, scale of 1.0, and uses uniform distribution for exploration within the local vicinity. This operator follows the metropolis selector to decide whether to accept new solutions probabilistically.
# These operators are chosen based on their ability to balance exploration and exploitation in optimization problems, where GSO is known to perform well without getting stuck in local minima due to its diverse set of search patterns. The configuration of these parameters aims to enhance the global convergence and accuracy of the solution while maintaining diversity in the population for better exploration of the search space.