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
        'distribution': 'gaussian'
    },
    'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The code defines a metaheuristic named GravitationalSearchOptimization which uses three search operators: 
# gravitational_search, random_flight, and local_random_walk. These operators are configured with specific parameters based on the provided template. 
# The gravitational_search operator is set to use all elements in the population, while random_flight and local_random_walk are configured to use a probabilistic selector. 
# This setup aims to leverage diverse search mechanisms to optimize the Rastrigin function effectively. GravitationalSearchOperator uses gravity and alpha parameters for its behavior, RandomFlight utilizes scale, distribution, and beta, and LocalRandomWalk is defined by probability, scale, and distribution. These operators are chosen based on their ability to handle different types of problems as per the template's guidance.
