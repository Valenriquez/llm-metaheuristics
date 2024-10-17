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
        'greedy'
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
# The metaheuristic combines two distinct search operators: Gravitational Search and Random Flight. 
# Gravitational Search, with its gravity parameter set to 1.0 and alpha to 0.02, aims to simulate the gravitational force among particles in a space, promoting convergence towards better regions of the solution space.
# Random Flight, characterized by its scale set to 1.0, distribution set to 'levy', and beta coefficient at 1.5, uses a probabilistic approach inspired by the foraging patterns of animals during search processes, allowing exploration of diverse areas in the solution space.
# Both operators are assigned specific selectors: Gravitational Search uses 'greedy' selection pressure, while Random Flight employs 'probabilistic' selection to balance between exploitation and exploration effectively. This combination aims to leverage both deterministic convergence properties of gravitational search and probabilistic exploration capabilities of random flight for optimizing the Rastrigin function in a multi-dimensional space.