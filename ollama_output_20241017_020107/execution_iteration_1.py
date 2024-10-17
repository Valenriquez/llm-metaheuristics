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
# The GravitationalSearchMetaheuristic is designed to solve optimization problems using a gravitational search algorithm. 
# The first operator, 'gravitational_search', employs a gravity parameter of 1.0 and an alpha value of 0.02. This operator uses the Metropolis selector for probabilistic selection of candidate solutions.
# The second operator, 'random_flight', scales its operations by 1.0 with a levy distribution and a beta factor of 1.5. It is configured to use a probabilistic selection method as well, specifically the 'probabilistic' selector.
# These operators are chosen based on their ability to explore diverse search spaces effectively, leveraging different distributions and parameters tailored for optimization tasks in continuous domains. The inclusion of both gravitational search and random flight dynamics aims to balance exploration and exploitation, enhancing the algorithm's performance in finding global optima or near-optimal solutions.