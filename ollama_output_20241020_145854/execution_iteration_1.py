 # Name: GravitationalSearchAndRandomFlightMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1: Gravitational Search
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
    'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic GravitationalSearchAndRandomFlightMetaheuristic combines two search operators, gravitational search and random flight, to explore the solution space of the Rastrigin function. 
# Gravitational search is designed with parameters 'gravity' set to 1.0 and 'alpha' to 0.02, focusing on a balance between exploration and exploitation. This operator uses all possible selections ('all').
# Random flight employs 'scale' at 1.0, with a distribution type selected as 'levy', and 'beta' set to 1.5. It operates under the selector 'metropolis', which implies a probabilistic selection strategy that allows for more exploration in less favorable conditions.
# These operators are chosen based on their ability to handle diverse search spaces and optimize functions, with gravitational search emphasizing gravity-like behavior for convergence properties and random flight using levy distribution for broader exploration.