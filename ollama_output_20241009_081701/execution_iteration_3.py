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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic, using the gravitational search algorithm with specific parameters for exploration. The gravitational search operator is configured with a gravity of 1.0 and an alpha value of 0.02. Additionally, a random flight operator is included with a scale of 1.0 and a distribution set to levy, utilizing probabilistic selection for making decisions. Both operators are applied using the 'all' selector, meaning they will be used in each iteration of the metaheuristic. The benchmark function Rastrigin(2) is selected as the optimization problem, and the algorithm runs for 100 iterations with verbose output enabled to track progress.
