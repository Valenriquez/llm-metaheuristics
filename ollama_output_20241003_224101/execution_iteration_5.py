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
# The GravitationalSearchMetaheuristic is designed to solve optimization problems using a gravitational search algorithm. 
# It begins with the gravitational_search operator, which has parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. 
# This operator uses all possible selectors as specified ('all').
# Following this, the random_flight operator is introduced, designed for probabilistic selection based on a levy distribution with scaling factor of 1.0 and beta value of 1.5. 
# It operates under the 'probabilistic' selector to ensure randomness in its operation.
# These operators are chosen as they represent fundamental components of gravitational search algorithms and probabilistic exploration, which is typical for metaheuristics aiming at global optimization.