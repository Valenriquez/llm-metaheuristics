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
# This code implements a gravitational search metaheuristic for optimizing the Rastrigin function with two dimensions. The Gravitational Search algorithm is designed to mimic the behavior of celestial bodies under the influence of gravity, where each "body" represents a potential solution to the optimization problem. In this case, we use an all-selector that applies the gravitational search operator to all candidate solutions at every iteration. Parameters include 'gravity' (representing the strength of the gravitational force) and 'alpha' which controls the scaling factor for the velocity updates. These parameters are taken from the provided template, ensuring adherence to the specifications in parameters_to_take.txt.
