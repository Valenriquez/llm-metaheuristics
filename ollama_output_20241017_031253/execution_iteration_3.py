 # Name: GravitationalSearchMetaheuristic
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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the Gravitational Search algorithm, which simulates the gravitational force between particles to explore the search space. The algorithm involves two main operators: 
# 1. Gravitational Search (gravitational_search) with parameters gravity set to 1.0 and alpha to 0.02, applied to all possible solutions.
# 2. Random Flight (random_flight) with scale of 1.0, distribution set to levy, and beta of 1.5, also configured for probabilistic selection among possible solutions.
# These operators are designed to balance between exploration and exploitation in the optimization process, leveraging both deterministic and stochastic elements as per the parameters specified.