 # Name: GravitationalSearchAlgorithm
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
        'gravity': 0.5,
        'alpha': 0.01
    },
    'all'
    ),
    (  
    'random_flight',
    {
        'scale': 0.5,
        'distribution': 'levy',
        'beta': 2.0
    },
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=150)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Algorithm (GSA) is a population-based metaheuristic inspired by the principles of gravitational force and mass interactions. In this implementation, we set the gravity parameter to 0.5 and alpha to 0.01 to emphasize local search capabilities while still allowing for global exploration.
# The Random Flight operator uses a scale factor of 0.5 with a levy distribution and beta value of 2.0. This configuration promotes both directed and random movements, enhancing the probabilistic nature of the search, which is suitable for exploring diverse areas in the solution space.
# Both operators are configured to run continuously during the iterations using the 'all' selector, ensuring comprehensive exploration and exploitation throughout the optimization process.