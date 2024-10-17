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
# The metaheuristic chosen is Gravitational Search Algorithm (GSA), which simulates the gravitational force between masses to search for optimal solutions. 
# Two operators are used in this implementation: gravitational_search and random_flight.
# Gravitational search utilizes parameters gravity (1.0) and alpha (0.02) to simulate the gravitational forces, with a selector set to 'metropolis' which suggests that the algorithm will use a probabilistic approach to accept worse solutions for exploration. 
# Random flight uses scale (1.0), distribution ('levy'), and beta (1.5). The distribution is set to 'levy', indicating that the random walk follows a levy distribution, aiding in better global search capabilities. The selector is set to 'probabilistic' to allow for probabilistic acceptance of worse solutions during the search process.
# These operators are chosen based on their ability to balance between exploration and exploitation, with gravitational_search focusing on local search and random_flight promoting global search through its levy distribution approach.