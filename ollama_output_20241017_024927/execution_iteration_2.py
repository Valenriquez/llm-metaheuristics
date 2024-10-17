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
# The chosen metaheuristic is Gravitational Search Algorithm (GSA), which simulates the gravitational force among particles to search for optimal solutions. 
# In this case, two operators are used: 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02; and 'random_flight' with scale set to 1.0, distribution as levy, and beta as 1.5. 
# The selector for both operators is 'probabilistic', which means these operators will be applied probabilistically during the search process. 
# This combination aims to leverage both gravitational attraction forces and random exploration to efficiently find the optimal solution of the Rastrigin function in a 2D space.