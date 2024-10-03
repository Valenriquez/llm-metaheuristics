 # Name: GravitationalSearchAndRandomFlightMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
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
# This metaheuristic combines the Gravitational Search (GS) algorithm with a Random Flight operator. 
# The GS is designed to mimic the gravitational interaction between masses, promoting convergence towards better solutions by adjusting the positions of particles based on their mass and distance.
# The random flight operator introduces diversity into the search process, allowing exploration of new areas in the solution space through both Levy flights (which favor larger steps) and uniform distribution for smaller jumps. 
# By using 'all' as the selector for GS and 'probabilistic' for the Random Flight operator, we ensure that these operators are applied to all particles during each iteration, balancing exploration and exploitation effectively.