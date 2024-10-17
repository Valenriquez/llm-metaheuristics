 # Name: GravitationalSearchMetaheuristic
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
    'metropolis'
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
# The GravitationalSearchMetaheuristic uses a gravitational search operator with gravity set to 1.0 and alpha to 0.02, combined with a random flight operator using a scale of 1.0 and levy distribution for beta of 1.5. Both operators are configured to use the probabilistic selector as they rely on stochastic processes to guide their exploration. This combination aims to balance local search capabilities with global exploration, suitable for continuous optimization problems like Rastrigin function. The GravitationalSearchMetaheuristic leverages both deterministic and probabilistic strategies to efficiently navigate the search space in a more informed manner compared to purely random approaches.
