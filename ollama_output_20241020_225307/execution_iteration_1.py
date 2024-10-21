 # Name: GravitationalSearchOptimizer
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
# This code defines a metaheuristic named GravitationalSearchOptimizer using the gravitational search algorithm, which is characterized by its 'gravity' parameter that influences the strength of the gravitational force, and an 'alpha' parameter that affects how quickly the population converges. The algorithm also includes random flights represented by the 'random_flight' operator with parameters for scale and distribution type ('levy' or 'beta') to explore the solution space probabilistically. The selector is set to 'probabilistic' to ensure that these operators operate under a probabilistic framework, promoting exploration in the search space while maintaining some level of randomness during optimization.
