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
# This code implements a metaheuristic based on the Gravitational Search Algorithm (GSA). The algorithm uses two main operators: 
# 1. gravitational_search with parameters gravity set to 1.0 and alpha to 0.02, applied to all possible solutions ('all').
# 2. random_flight with scale set to 1.0, distribution set to 'levy', and beta set to 1.5, also applied probabilistically ('probabilistic').
# The purpose of using these operators is to explore the solution space by combining both gravitational attraction-based search (to converge towards promising regions) and random flight for diversity (to avoid local minima). This approach helps in efficiently finding a good balance between exploration and exploitation.
# Both parameters gravity and alpha are critical as they determine the strength of the gravitational force, while the beta parameter influences the distribution of the random flights. The use of 'probabilistic' selector ensures that these operators have an element of randomness which is beneficial for escaping local optima.