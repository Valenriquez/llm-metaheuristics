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
# The metaheuristic chosen for this experiment is a combination of Gravitational Search and Random Flight. 
# The Gravitational Search, with its gravity parameter set to 1.0 and alpha to 0.02, aims to simulate the gravitational force acting on particles in space which helps them converge towards better solutions by adjusting their positions according to mass distributions. 
# The Random Flight operator is configured with a scale of 1.0, using a levy distribution for exploration, and a beta value that defines the characteristics of this flight, set to 1.5. This probabilistic approach allows it to escape local minima effectively, contributing significantly to the overall performance by combining both exploitation (Gravitational Search) and exploration (Random Flight).
# The selector 'probabilistic' is chosen for the Random Flight operator because its behavior aligns well with the probabilistic nature of this search strategy which relies on random walks that can either move towards better solutions or explore new areas based on predefined probabilities.