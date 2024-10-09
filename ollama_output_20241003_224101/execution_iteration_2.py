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
# The GravitationalSearchMetaheuristic is designed to mimic the behavior of gravitational search within a metaheuristic framework. 
# This approach utilizes parameters such as 'gravity' (with a value of 1.0) and 'alpha' (set at 0.02). 
# The selector 'all' ensures that this operator will be applied to all elements in the population, allowing for comprehensive exploration of the search space.
# This specific implementation is tailored for continuous optimization problems like Rastrigin function evaluation, where gravitational forces are simulated to guide the search towards better solutions.