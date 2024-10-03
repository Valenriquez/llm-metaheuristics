 # Name: GravitationalSearchOptimizer
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1: GravitationalSearchOptimizer with gravitational search
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'all'
    ),
    (  
    'random_flight', # Search operator 2: Random Flight with random flight parameters
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
# The GravitationalSearchOptimizer is used with a gravitational search operator, 
# where gravity and alpha are set to their respective default values from parameters_to_take.txt. 
# This operator aims to simulate the behavior of particles influenced by gravitational forces in space.
# The random flight operator is introduced next, configured with scale set to 1.0, distribution as levy, and beta at 1.5.
# This probabilistic approach allows for a broader exploration by following a levy distribution, enhancing diversity in search.
# Both operators are applied globally across the population using the 'all' selector, ensuring they influence all particles during each iteration.