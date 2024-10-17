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
# The metaheuristic combines Gravitational Search and Random Flight operators to explore the solution space. 
# Gravitational Search is designed with a gravity parameter of 1.0 and an alpha value of 0.02, while Random Flight uses a scale of 1.0 and follows a levy distribution. 
# The selector for the Gravitational Search is set to 'all' to ensure it operates on all particles, and for the Random Flight, it is set to 'probabilistic' which means that moves are accepted based on probabilistic criteria. 
# This combination leverages both gravitational attraction (as in Gravitational Search) and random exploration (as in Random Flight), aiming to balance between exploitation and exploration efficiently.