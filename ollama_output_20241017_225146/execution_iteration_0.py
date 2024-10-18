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
# The metaheuristic combines Gravitational Search with Random Flight to explore the search space. 
# Gravitational Search utilizes a gravitational constant (gravity) and an alpha parameter to influence the movement of solutions, 
# while Random Flight uses scale, distribution type, and beta parameters to introduce random movements in different distributions.
# The selector 'all' is used for Gravitational Search to ensure all possible solutions are considered during search, 
# and 'probabilistic' for Random Flight to allow for stochastic decisions based on probability distributions.
# This combination aims to leverage the strengths of both methods by exploring systematically with gravitational forces and diversifying randomly with flight dynamics.