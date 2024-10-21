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
    ),
    ( # Search operator 3
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
# The metaheuristic used in this script is Gravitational Search Optimization (GSO). 
# GSO mimics the gravitational force and motion within a system, where each 'particle' represents a solution in the search space.
# The first operator is set to 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02. This operator uses all possible selectors.
# The second operator is 'random_flight' with scale set to 1.0, distribution as levy, and beta of 1.5. It utilizes a probabilistic selector.
# The third operator is 'local_random_walk' with probability set to 0.75, scale to 1.0, and distribution as gaussian. This operator uses the metropolis selector.
# These operators are chosen based on their ability to explore diverse areas of the search space through gravitational attraction, random flights, and local walks, respectively. Each operator is configured according to its specific parameters detailed in the parameters_to_take.txt file.