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
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchMetaheuristic, which incorporates two main search operators: gravitational_search and random_flight. The gravitational_search operator uses parameters 'gravity' set to 1.0 and 'alpha' set to 0.02, while the selector for this operator is set to 'greedy'. The random_flight operator utilizes parameters 'scale' set to 1.0, 'distribution' set to 'levy', and 'beta' set to 1.5, with its selector configured as 'probabilistic'. These operators are chosen based on their ability to balance exploration and exploitation in the search space according to the problem requirements specified in parameters_to_take.txt.
