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
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The algorithm is named GravitationalSearchOptimizer as it uses the concept of gravitational search to optimize the problem. 
# We have selected the 'gravitational_search' operator from the parameters provided in parameters_to_take.txt.
# The gravity parameter is set to 1.0, which is a standard value for this parameter according to the literature.
# Alpha is set to 0.02, which is another default setting that has shown good performance in similar optimization tasks.
# The selector 'all' is used because we want the gravitational search operator to be applied to all particles or solutions during each iteration.
# This setup aims to explore and exploit the solution space effectively by simulating the gravitational forces between mass particles, which encourages convergence towards better solutions while avoiding premature convergence.