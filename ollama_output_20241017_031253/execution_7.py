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
# This metaheuristic is named GravitationalSearchMetaheuristic as it combines the gravitational search algorithm with two operators - 'gravitational_search' and 'random_flight'. The parameters for these operators are taken from the provided configuration, ensuring they adhere to the guidelines specified in the task. 
# The gravitational search operator uses a gravity constant of 1.0 and an alpha value of 0.02. This setup is typical for exploring regions with higher probability density according to the probabilistic selector 'greedy'.
# The random flight operator utilizes a scale factor of 1.0, and its distribution is set to levy which is beneficial for diversifying the search space while maintaining exploration and exploitation balance as per the 'probabilistic' selector.
# Both operators are configured with parameters that follow the template provided in the task, ensuring consistency and correctness in their implementation.