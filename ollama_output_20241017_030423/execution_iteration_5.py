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
    'greedy'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm for optimization. 
# The Rastrigin function is chosen as the benchmark problem with dimension 2.
# Two heuristic operators are employed:
# 1. Gravitational Search (gravitational_search) with parameters gravity set to 1.0 and alpha to 0.02, using a greedy selector.
# 2. Random Flight (random_flight) with scale set to 1.0, distribution as levy, and beta of 1.5, operating on all possible selections.
# The metaheuristic runs for 100 iterations, and the best solution is printed after execution. This setup aims to leverage both deterministic and probabilistic search strategies for effective optimization.