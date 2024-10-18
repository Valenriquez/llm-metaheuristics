 # Name: GravitationalSearchMetaheuristic
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
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code implements a gravitational search metaheuristic for optimizing the Rastrigin function with two dimensions. The Gravitational Search algorithm is used to explore the solution space, with specific parameters set as per the provided template. The selector 'all' ensures that all operators are applied in each iteration, allowing the exploration of diverse solutions across the problem space.
# The Gravitational Search operator is defined with parameters: gravity set to 1.0 and alpha to 0.02. These parameters influence the strength of gravitational forces acting on particles within the search space. The selector 'all' ensures that these operators are applied in every iteration, promoting a comprehensive exploration strategy.
# This approach aims to balance between global and local explorations by adjusting the gravity parameter dynamically or setting it to a fixed value. The alpha parameter controls the scaling of gravitational forces relative to the particle velocities, affecting the convergence speed and ability to escape local minima.
