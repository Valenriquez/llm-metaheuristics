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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm with specified parameters for exploration. The first operator is 'gravitational_search' with gravity set to 1.0 and alpha to 0.02, utilizing the metropolis selector for probabilistic selection of solutions. The second operator is 'random_flight' with scale set to 1.0 and distribution as levy, using a probabilistic approach indicated by the 'probabilistic' selector. These operators are designed to navigate through the search space efficiently, combining both deterministic and stochastic elements based on the defined parameters and selectors for optimal performance in solving optimization problems like the Rastrigin function.