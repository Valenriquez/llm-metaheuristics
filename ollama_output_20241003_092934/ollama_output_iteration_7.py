 # Name: GravitationalSearchOptimizer
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
# The code defines a metaheuristic named GravitationalSearchOptimizer using the gravitational search and random flight operators. 
# The gravitational search operator is configured with gravity set to 1.0 and alpha to 0.02, while the random flight operator has scale set to 1.0 and distribution set to levy with beta of 1.5.
# Both operators are used with a probabilistic selector. This setup aims to leverage both local search capabilities of gravitational search and global exploration properties of random flight for optimizing the benchmark function Sphere with dimension 2.
