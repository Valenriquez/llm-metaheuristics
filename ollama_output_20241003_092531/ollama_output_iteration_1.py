 # Name: GravitationalSearchAlgorithm
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
# The Gravitational Search Algorithm (GSA) is used with a gravity parameter of 1.0 and an alpha value of 0.02. This search operator aims to simulate the gravitational force acting on masses in a multidimensional space, promoting particles towards regions with lower function values by adjusting their positions based on gravity.
# The Random Flight operator uses a scale factor of 1.0 and follows a levy distribution (a type of probability distribution often used to model scales or distances). With beta set to 1.5, it introduces randomness in the search process, allowing for exploration of various regions in the problem space. This selector 'probabilistic' means that this operator will be applied with a probabilistic approach, where the selection is not deterministic but influenced by probabilities based on fitness values and other factors.
# Both operators are combined within the metaheuristic framework to leverage the strengths of gravitational search for exploitation and random flight for exploration, aiming to converge towards an optimal solution efficiently.