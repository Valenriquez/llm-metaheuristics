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
    'metropolis'
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
# This code implements a gravitational search optimization metaheuristic for solving the Rastrigin function with 2 dimensions. The first operator is a gravitational search, which uses parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. The selector used here is 'metropolis'. The second operator is a random flight, configured with 'scale' at 1.0 and distribution type as 'levy', along with the selector set to 'probabilistic'. Both operators are chosen based on their ability to explore and exploit in different ways, which should help in finding good solutions for the Rastrigin function despite its multi-modal nature. The gravitational search focuses on global optimization by simulating the effect of gravity on particles, while random flight introduces a stochastic element that can jump out of local minima.