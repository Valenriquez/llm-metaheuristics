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
# The code defines a metaheuristic named GravitationalSearchAlgorithm using the provided parameters for each operator from the parameters_to_take.txt file. It initializes the Rastrigin benchmark function with dimension 2, sets up the gravitational search and random flight operators with specified parameters, and runs the metaheuristic for 100 iterations. The gravitational search utilizes gravity and alpha as key parameters to influence the search behavior, while the random flight operator uses scale, distribution (levy), and beta to determine its movement characteristics. Both operators are applied using a probabilistic selector that considers only 'probabilistic' selections during their operation. This setup aims to leverage both deterministic and stochastic exploration capabilities of these operators for optimizing the Rastrigin function.