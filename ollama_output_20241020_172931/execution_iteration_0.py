 # Name: GravitationalSearchAlgorithm
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
    'greedy'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code implements a gravitational search algorithm where the first operator is set to use gravitational search with parameters gravity=1.0 and alpha=0.02, 
# using a greedy selector. The second operator uses random flight with scale=1.0, distribution='levy', and beta=1.5, configured to select operators based on the Metropolis criterion.
# Both operators are applied in an iterative process over 100 iterations to solve the Rastrigin function optimization problem in a 2-dimensional space.
