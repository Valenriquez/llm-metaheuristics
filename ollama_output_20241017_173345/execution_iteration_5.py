 # Name: GravitationalSearchAndLocalRandomWalkMetaheuristic
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
    ( # Local Random Walk Operator
        'local_random_walk',
        { 
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines the gravitational search algorithm with a local random walk strategy. 
# The Gravitational Search Operator is configured with gravity set to 1.0 and alpha to 0.02, aiming to simulate the force of gravity in optimization. 
# It uses all possible solutions for exploration ('all' selector).
# The Local Random Walk Operator is set to use a Gaussian distribution for its random steps, with a probability of 0.75 and scale of 1.0. 
# This operator operates under the Metropolis selection criteria, which are suitable for exploring local neighborhoods while avoiding being trapped in suboptimal solutions.
