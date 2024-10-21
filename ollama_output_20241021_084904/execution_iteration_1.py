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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm. 
# The gravitational search operator is configured with gravity set to 1.0 and alpha to 0.02, both parameters taken from parameters_to_take.txt.
# The selector used for all operators in this metaheuristic is 'all', which means these operators will be applied to the entire population during each iteration.
# This setup aims to explore a diverse range of solutions by applying gravitational forces across the search space, with a focus on finding optimal parameters that lead to better performance in solving the Rastrigin function for 2 dimensions.
