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
# This metaheuristic combines the Gravitational Search (GSO) algorithm with the Random Flight (RF) operator to explore the search space. 
# The GSO is designed with parameters gravity and alpha, which control the strength of the gravitational force and the exploration/exploitation balance. 
# The RF introduces a random component scaled by 'scale', where the distribution can be either levy or uniform, affecting how the steps are taken during the flight phase. 
# By using both operators together with probabilistic selection, we aim to leverage the strengths of gravitational forces for convergence and random exploration for diversity in the search space. This approach is justified as it combines the benefits of different metaheuristic techniques to potentially improve solution quality and convergence speed.