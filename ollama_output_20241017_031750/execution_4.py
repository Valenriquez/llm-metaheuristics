 # Name: GravitationalSearchAndRandomFlightMetaheuristic
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
        'all'
    ),
    ( # Random Flight Operator
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
# The metaheuristic combines the Gravitational Search algorithm with a Random Flight operator. 
# Gravitational search uses parameters gravity (1.0) and alpha (0.02) to control the strength of the gravitational force.
# Random flight involves scaling factor (1.0), distribution type ('levy'), and beta (1.5).
# The selector for both operators is set to 'probabilistic' as it fits well with the probabilistic nature of random flight. 
# Both algorithms are applied iteratively, allowing for exploration based on probability in the search space, which can be beneficial for optimization tasks requiring diverse search behaviors.