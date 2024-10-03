 # Name: PSOGravitationalSearch
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
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
# The PSOGravitationalSearch metaheuristic is designed to optimize the Sphere function with a combination of gravitational search and random flight operators. Gravitational search, controlled by gravity (1.0) and alpha (0.02), attracts particles towards promising regions based on their mass, which represents fitness values. Random flight, characterized by scale (1.0), distribution ('levy' or 'uniform'), and beta (1.5), introduces exploration by allowing random jumps in the search space. The gravitational search uses all solutions, while the probabilistic selector for the random flight operator allows it to be applied with a probability, promoting both exploitation and exploration during optimization.
