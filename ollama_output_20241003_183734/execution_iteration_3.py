 # Name: GravitationalSearchAlgorithmMetaheuristic
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
    'greedy'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'probabilistic'
    ),
    ( # Search operator 3
    'local_random_walk',
    {
        'probability': 0.75,
        'scale': 1.0,
        'distribution': 'uniform'
    },
    'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchAlgorithmMetaheuristic using the gravitational search, random flight, and local random walk operators. The gravitational search operator has gravity set to 1.0 and alpha to 0.02. The random flight operator uses scale of 1.0 with a levy distribution and beta of 1.5. The local random walk operator employs probability of 0.75, scale of 1.0, and uniform distribution. Selectors for these operators are set as greedy, probabilistic, and metropolis respectively. These settings aim to explore the search space effectively by combining deterministic and probabilistic elements in a metaheuristic framework that runs for 100 iterations on the Rastrigin function with two dimensions.
