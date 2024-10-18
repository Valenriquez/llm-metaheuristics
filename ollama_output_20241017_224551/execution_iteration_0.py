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
# This code implements a gravitational search metaheuristic to solve the Rastrigin function optimization problem. 
# The GravitationalSearch operator is configured with parameters 'gravity' set to 1.0 and 'alpha' set to 0.02, which are typical for this type of algorithm.
# The selector used is 'all', indicating that these operators should be applied to all elements in the population during each iteration.
# This approach leverages gravitational forces to search for better solutions by iteratively adjusting positions based on mass and distance, mimicking a physical system's dynamics.
# The metaheuristic runs for 100 iterations, and its performance is evaluated through verbose output showing the best solution found after all iterations are completed.