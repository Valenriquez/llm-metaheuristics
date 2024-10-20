 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(22)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        { 
            'gravity': 1.0,
            'alpha': 0.02
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm. The algorithm is configured with gravity set to 1.0 and alpha to 0.02, which are parameters for controlling the strength of the gravitational force. The selector used is 'metropolis', indicating that this method will decide whether to accept new solutions based on a probabilistic criterion.
# The problem being solved is the Sphere function with dimension 22. This choice is justified by the need to find an optimal solution in a high-dimensional space, which is typical for testing metaheuristic algorithms due to its continuous nature and simplicity of evaluation.
# The gravitational search algorithm involves simulating physical laws such as gravity to guide the search process, which can be effective for optimization problems where exploration and exploitation are balanced through adjustable parameters that influence the strength of the "gravitational" force applied to candidate solutions.