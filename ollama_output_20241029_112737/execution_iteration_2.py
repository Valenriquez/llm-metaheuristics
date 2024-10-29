 # Name: GravitationalSearchOptimizer
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
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
# The Gravitational Search Optimizer (GSO) is used to solve the Rastrigin function optimization problem with two dimensions. 
# The GSO algorithm mimics the gravitational force among particles, where each particle represents a potential solution in the search space. 
# The 'gravitational_search' operator is configured with parameters: 'gravity' set to 1.0 and 'alpha' set to 0.02. 
# These parameters control the strength of the gravitational field and the scaling factor, respectively. 
# The selector 'all' ensures that all particles participate in the search process, allowing for exploration of the entire solution space. 
# This setup aims to balance between local exploitation and global exploration, which is suitable for a multi-dimensional optimization problem like Rastrigin function.