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
# The chosen metaheuristic is Gravitational Search Optimizer (GSO). 
# I selected this algorithm because it mimics the gravitational interaction between masses in space, 
# which can be effective for solving continuous optimization problems like the Rastrigin function.

# For the first search operator, 'gravitational_search', 
# I set the gravity parameter to 1.0 and alpha to 0.02 as per parameters_to_take.txt. 
# This setup allows for exploration of the solution space with a balance between random and directed movement.

# The second operator, 'random_flight', is designed to mimic random search mechanisms. 
# Here, I set scale to 1.0, distribution to 'levy' which typically exhibits long-range correlations, 
# and beta to 1.5 based on the default settings suggested for this type of operator in parameters_to_take.txt.

# Both operators are configured with a selector setting 'probabilistic', 
# which means they will operate probabilistically across the solution space during optimization.

# This setup aims to leverage both gravitational attraction and random flight characteristics, 
# providing a comprehensive exploration of the problem's landscape for potential solutions.