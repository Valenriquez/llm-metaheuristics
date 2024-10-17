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
# The GravitationalSearchMetaheuristic is designed to optimize the Rastrigin function with two main operators. 
# The first operator, gravitational_search, employs a gravity parameter of 1.0 and an alpha value of 0.02. 
# This operator uses a 'all' selector, indicating that it will be applied to all possible scenarios within the optimization process.
# The second operator is random_flight, which has parameters scale set to 1.0, distribution set to 'levy', and beta set to 1.5.
# This operator utilizes a 'probabilistic' selector, meaning that its behavior will be governed by probabilistic criteria during the optimization process.
# These operators are chosen based on their ability to explore diverse regions of the search space, which is crucial for optimizing complex functions like the Rastrigin function.