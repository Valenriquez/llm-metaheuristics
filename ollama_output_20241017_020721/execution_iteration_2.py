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
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The code defines a metaheuristic named GravitationalSearchMetaheuristic using the Gravitational Search algorithm. 
# It is initialized with parameters gravity set to 1.0 and alpha to 0.02, both taken from parameters_to_take.txt.
# The selector used in this implementation is 'all', which indicates that the gravitational search will be applied to all possible solutions during each iteration.
# This approach aims to balance exploration and exploitation by adjusting the strength of gravity, allowing for effective global optimization without getting stuck in local minima. 
# The Gravitational Search algorithm uses a combination of gravitational forces between particles to explore and converge towards the optimal solution.