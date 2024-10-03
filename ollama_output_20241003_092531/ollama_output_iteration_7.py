 # Name: GravitationalSearchOptimizer
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Optimizer is implemented with a single gravitational search operator. 
# The parameters for the gravitational search are set to default values as per the provided template, which includes gravity (1.0) and alpha (0.02). 
# The selector used is 'all', indicating that all particles will undergo this search method during each iteration.
# This setup aims to explore the solution space uniformly by applying gravitational forces across all particles in a population-based manner.