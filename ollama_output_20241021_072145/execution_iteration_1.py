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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic, which uses the gravitational search algorithm to solve an optimization problem. 
# The Rastrigin function is chosen as the benchmark function with dimension 2.
# Two operators are included in the heuristic list: gravitational_search and random_flight.
# The gravitational_search operator has parameters gravity set to 1.0 and alpha set to 0.02. It uses a selector named 'all'.
# The random_flight operator has parameters scale set to 1.0, distribution set to 'levy', and beta set to 1.5. It uses a selector named 'probabilistic'.
# The metaheuristic is run for 100 iterations with verbose output enabled, and the best solution found is printed.