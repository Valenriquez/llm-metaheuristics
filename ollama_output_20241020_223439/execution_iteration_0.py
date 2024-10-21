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
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic chosen is Gravitational Search Optimization (GSO), which emulates the gravitational force between masses in a multidimensional space to search for optimal solutions. 
# In this implementation, we use two operators: 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02, and 'random_flight' with scale set to 1.0 and distribution as levy, beta of 1.5. 
# The selector for both operators is set to 'probabilistic', which means that the selection process will be influenced by a probability function during the search, enhancing exploration in the solution space. 
# This combination aims to balance between exploitation (using gravitational force) and exploration (random flight) effectively, which should lead to better convergence towards an optimal solution for the Rastrigin function.