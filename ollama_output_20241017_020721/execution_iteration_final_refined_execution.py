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
# This metaheuristic combines the Gravitational Search Algorithm (GSA) with a Random Flight mechanism to explore the search space. 
# The GSA is designed to mimic the gravitational interactions among particles, where each particle represents a potential solution to the optimization problem. 
# The gravity parameter controls the strength of the interaction, while alpha determines the scaling factor for the movement based on the current position and the global best position.
# The random flight operator introduces stochastic elements by simulating movements influenced by Levy distributions or uniform randomness. This helps in escaping local minima and exploring new areas of the search space. 
# By integrating these two operators with a probabilistic selector, we aim to leverage both the exploration capabilities of random flight and the convergence properties of gravitational search, which should lead to more effective optimization across various benchmark functions.