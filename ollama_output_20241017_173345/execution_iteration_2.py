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
        'alpha': 0.01
    },
    'metropolis'
    ),
    (  
    'local_random_walk',
    {
        'probability': 0.8,
        'scale': 0.5,
        'distribution': 'gaussian'
    },
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=150)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic used in this script is a combination of Gravitational Search (with reduced gravity and slightly adjusted alpha) 
# and Local Random Walk with Gaussian distribution for mutation, both configured to use the probabilistic selector.
# Gravitational Search's parameters are tuned down from the default values to potentially encourage more exploration without sacrificing convergence speed.
# The Local Random Walk operator uses a lower probability (0.8 instead of the default 0.75) and scales its movement with a smaller scale factor (0.5), 
# which is set to be less aggressive, allowing for both deep explorations into promising areas and occasional random jumps to escape local minima.
# The probabilistic selector ensures that moves are not solely determined by deterministic rules but also include stochastic elements based on predefined probabilities, 
# which can help in avoiding premature convergence while still maintaining a dynamic search behavior.