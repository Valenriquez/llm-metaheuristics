 # Name: GravitationalSearchOptimizer
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
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Optimizer is implemented with a gravity parameter set to 1.0, which represents the strength of gravitational force affecting particle movements. The alpha parameter is set to 0.02, influencing how quickly the system converges towards an optimal solution. This implementation uses 'all' as the selector, meaning that all particles in the population will undergo the gravitational search operation at each iteration. This approach aims to balance exploration and exploitation by allowing every candidate solution to potentially adjust its position according to the simulated gravitational forces within the optimization process.
