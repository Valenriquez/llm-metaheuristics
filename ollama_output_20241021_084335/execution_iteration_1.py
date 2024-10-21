 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
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
    ),
    ( # Local Random Walk Operator
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic used in this code is Gravitational Search (GS), which is inspired by the gravitational force acting between masses. 
# GS consists of three main operators: gravitational_search, random_flight, and local_random_walk. 
# The gravitational_search operator uses parameters gravity and alpha to control the search behavior, with a greedy selector.
# The random_flight operator explores the solution space using a levy distribution, also with a probabilistic selector.
# Local_random_walk operator performs a local walk based on gaussian distribution, employing metropolis selection for optimization. 
# These operators are combined in a metaheuristic framework to solve the Rastrigin function optimization problem. The GS algorithm's parameters and selectors are selected based on their effectiveness in handling complex non-linear and multi-modal functions, as suggested by the literature.