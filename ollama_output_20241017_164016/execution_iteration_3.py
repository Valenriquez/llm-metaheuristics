 # Name: GravitationalSearchAndMetropolisSelector
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
        'alpha': 0.05
    },
    'metropolis'
    ),
    (  
    'local_random_walk',
    {
        'probability': 0.8,
        'scale': 1.2,
        'distribution': 'gaussian'
    },
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The algorithm is designed to use a gravitational search with modified parameters for exploration, 
# accompanied by a local random walk that employs genetic mutation for better convergence properties. 
# Gravitational search allows for global exploration controlled by gravity and alpha parameters, 
# while the local random walk introduces diversity through Gaussian distribution mutations. 
# The metropolis selector is chosen to ensure probabilistic acceptance of new solutions based on energy differences, 
# which is beneficial for escaping local minima in multi-modal functions like Rastrigin's function.