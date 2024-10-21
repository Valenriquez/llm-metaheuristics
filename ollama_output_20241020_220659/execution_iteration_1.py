 # Name: GravitationalSearchMetaheuristic
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
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is designed to mimic the behavior of gravitational search, 
# where 'gravity' represents the strength of attraction between particles, 
# influencing their movement towards better solutions. The parameter 'alpha' 
# controls how quickly this attraction can be adjusted during the optimization process.
# The selector 'all' ensures that all possible states are considered at each iteration, 
# allowing for a comprehensive exploration and exploitation of the search space.
# This approach aims to balance between global and local explorations by dynamically adjusting 
# the gravitational forces acting on particles, promoting convergence towards an optimal solution.