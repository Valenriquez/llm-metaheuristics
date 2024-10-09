 # Name: GravitationalSearchOptimization
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
# The GravitationalSearchOptimization (GSO) is a metaheuristic inspired by the laws of gravity and motion in physics. 
# It uses two main operators: gravitational_search and random_flight. 
# The gravitational_search operator simulates the effect of gravity on particles, adjusting their positions based on predefined parameters (gravity and alpha). 
# The random_flight operator introduces a probabilistic element by considering both exploration and exploitation through its distribution type (levy) and beta parameter. 
# Both operators are configured to work together using the 'all' selector for each, ensuring comprehensive search across the solution space while maintaining diversity in the population.
# This setup is appropriate for problems where balance between global exploration and local exploitation is crucial, as demonstrated by the use of both gravitational and random-based mechanisms.