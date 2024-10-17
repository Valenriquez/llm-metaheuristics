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
# The algorithm is designed to mimic the behavior of gravitational search by using parameters for gravity and alpha, which influences the movement within the solution space. 
# The first operator 'gravitational_search' uses a predefined gravity value of 1.0 and an alpha factor of 0.02. It operates on all possible solutions in the problem space.
# The second operator 'random_flight' has parameters for scale, distribution type (levy), and beta. These are set to explore the solution space probabilistically by using a levy distribution.
# Both operators utilize different strategies for exploring the search space: gravitational search focuses on simulating physical forces, while random flight employs probabilistic methods based on the selected distribution. 
# The combination of these two operators aims to leverage both deterministic and stochastic approaches in the optimization process, potentially leading to better convergence towards optimal solutions within the Rastrigin function.