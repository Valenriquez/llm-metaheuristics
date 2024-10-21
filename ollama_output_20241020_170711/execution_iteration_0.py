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
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic chosen is Gravitational Search Algorithm (GSA). 
# GSA mimics the gravitational force between masses to search for the optimal solution.
# The first operator used is 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02. 
# This operator will be applied to all possible solutions ('all').
# The second operator is 'random_flight' which uses a scale of 1.0, distribution set to levy, and beta of 1.5. 
# This operator will follow a probabilistic approach ('probabilistic') during the search.
# These parameters are selected based on typical settings for GSA as described in the provided 'parameters_to_take.txt' file.