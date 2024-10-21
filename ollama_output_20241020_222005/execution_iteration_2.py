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
# The metaheuristic chosen for this task is a combination of Gravitational Search and Random Flight algorithms. 
# Gravitational Search (GS) mimics the gravitational force between masses to search for optimal solutions by iteratively adjusting positions based on predefined parameters. 
# Random Flight (RF), inspired by random walks, allows exploration through space with different distributions: uniform, Gaussian, or Levy. This helps in escaping local minima and exploring broader regions of the search space.
# The Gravitational Search algorithm is used with all possible solutions ('all' selector) to ensure comprehensive exploration, while the Random Flight operator employs a probabilistic selection strategy ('probabilistic') to balance between exploitation and exploration effectively.