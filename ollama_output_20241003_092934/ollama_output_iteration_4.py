 # Name: GravitationalSearchOptimization
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
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Optimization (GSO) is a metaheuristic inspired by the gravitational force in physics, where particles are attracted to more significant masses based on their mass and distance. In this implementation, we use an all-selector which means that every particle will be influenced by the gravitational search operator with gravity set to 1.0 and alpha to 0.02. This setup aims to mimic a population-based optimization where particles adjust their positions according to the combined effects of mass (which is not explicitly defined in this model, focusing on distance and strength of gravitational force) and the beta coefficient which is typically used to tune the distribution of masses but here set as 1 for simplicity. The GSO algorithm operates by iteratively updating the position of each particle based on its current velocity and acceleration influenced by the global best solution found so far. This approach allows exploration of the search space, potentially leading to a globally optimal or near-optimal solution for continuous optimization problems like the Sphere function used here.
