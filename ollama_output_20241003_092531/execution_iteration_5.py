 # Name: GravitationalSearchMetaheuristic
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
# This code implements a gravitational search metaheuristic to solve the Sphere function optimization problem. The Gravitational Search algorithm is based on the physical principle of gravity, where "gravity" represents the strength of attraction between particles (in this case, candidate solutions), and "alpha" controls the scaling factor for the movement of particles. The selector 'all' indicates that all operators should be applied to each solution in the population during each iteration. This approach is justified by its ability to balance exploration and exploitation, making it suitable for continuous optimization problems like the Sphere function.
