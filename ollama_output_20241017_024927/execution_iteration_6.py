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
# This code implements a Gravitational Search Optimization (GSO) metaheuristic with two operators: gravitational search and random flight. 
# The gravitational search operator uses parameters gravity set to 1.0 and alpha set to 0.02, while the selector is set to 'metropolis'. 
# The random flight operator utilizes scale of 1.0, distribution set to 'levy', and beta value of 1.5 with a selector set to 'probabilistic'. 
# These operators are chosen based on their ability to explore and exploit search spaces effectively according to the parameters specified in the parameters_to_take.txt file. The GSO is used here as it combines elements of both gravitational force-based optimization (which aids in exploration) and random flight for exploitation, making it suitable for optimizing complex functions like the Rastrigin function represented by prob.