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
# The Gravitational Search Algorithm (GSA) is a metaheuristic inspired by the principles of gravitational force and mass interaction. 
# In this implementation, we use GSA with parameters gravity set to 1.0 and alpha to 0.02. 
# This operator will be used in conjunction with a Metropolis selector for probabilistic exploration.
# The Random Flight operator simulates random movements using a levy distribution scaled by 1.0 and beta of 1.5, configured for probabilistic selection.
# Both operators are designed to explore the solution space through different mechanisms, combining traditional search methods with stochastic elements.