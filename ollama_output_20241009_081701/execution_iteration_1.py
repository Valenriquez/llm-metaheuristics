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
# The GravitationalSearchMetaheuristic is designed to explore the search space using a gravitational search algorithm. 
# The first operator, gravitational_search, employs a gravity parameter of 1.0 and an alpha value of 0.02. 
# This search method utilizes all possible solutions ('all') for exploration.
# The second operator, random_flight, has a scale set to 1.0 with a levy distribution and a beta value of 1.5. 
# It uses a probabilistic selector to guide its exploratory moves within the search space.
# These operators together aim to balance between global and local exploration, leveraging different dynamics for effective optimization.