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
# The metaheuristic named GravitationalSearchAndRandomFlightMetaheuristic is designed to solve the Rastrigin function optimization problem with two variables. 
# It utilizes a gravitational search algorithm for global exploration combined with a random flight strategy that employs both Levy and uniform distributions for local intensification.
# The gravitational search operator is configured with gravity set to 1.0 and alpha to 0.02, encouraging diverse exploration across the solution space.
# The random flight operator is set to scale by 1.0, using a Levy distribution for its randomness, which aids in escaping local minima and exploring new areas of the search space.
# The selector 'probabilistic' ensures that the random flight operator acts stochastically, increasing the probability of finding better solutions during each iteration.
# This combination aims to balance between global exploration and local intensification, leveraging both deterministic (gravitational search) and probabilistic (random flight) mechanisms for optimal performance in solving the Rastrigin function.