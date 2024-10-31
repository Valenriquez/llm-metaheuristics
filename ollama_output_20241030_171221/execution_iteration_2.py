 # Name: GravitationalSearchOptimization
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
# The chosen metaheuristic is Gravitational Search Optimization (GSO), which is inspired by the gravitational force in physics. 
# In this implementation, we have defined two operators: 'gravitational_search' and 'random_flight'. 
# The 'gravitational_search' operator uses parameters gravity and alpha to simulate the gravitational pull between particles. 
# These parameters are set to 1.0 and 0.02 respectively. 
# The selector for this operator is set to 'all', meaning all particles will be influenced by this force during each iteration.
# The 'random_flight' operator simulates random movements of particles with a scale factor of 1.0, using a levy distribution for the random step size. 
# This operator uses a probabilistic selector as it operates based on a probability threshold affecting the likelihood of performing a random walk.
# Together, these operators and parameters aim to explore the solution space effectively while maintaining diversity in search.