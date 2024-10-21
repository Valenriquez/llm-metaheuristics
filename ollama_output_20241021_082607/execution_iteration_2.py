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
# This metaheuristic combines the Gravitational Search (to exploit local minima) with Random Flight (for exploration of new areas). The Gravitational Search uses a gravity parameter to attract solutions towards better regions, while the Random Flight introduces randomness through different distributions (levy in this case), allowing for both exploitation and exploration. Both operators are applied throughout the iterations using the 'all' selector to ensure they contribute at every step, and the 'probabilistic' selector is used for the random flight operator to introduce stochastic behavior based on probability. This combination aims to balance between exploring diverse areas of the search space and refining solutions around current bests.
