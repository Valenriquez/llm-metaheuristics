 # Name: GravitationalSearchOptimization
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
# This code implements a Gravitational Search Optimization (GSO) metaheuristic. The GSO algorithm is inspired by the gravitational force between masses, where each solution in the population represents a mass influenced by other masses through attractive forces proportional to their masses and inversely proportional to the square of the distance between them. The parameters for this implementation are set as follows:
# - 'gravity': This parameter sets the strength of the gravitational constant, influencing the range over which solutions can be affected (set to 1.0).
# - 'alpha': This is a scaling factor that controls the sensitivity of the gravitational force (set to 0.02).
# The selector used in this implementation is set to 'all', meaning that the gravitational search operator will operate on all candidate solutions during each iteration.
# This setup aims to balance exploration and exploitation by allowing for both global and local searches through the influence of masses, which can be adjusted using the gravity and alpha parameters.