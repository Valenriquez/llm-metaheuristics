 # Name: GravitationalSearchAlgorithm
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
# This code implements a Gravitational Search Algorithm (GSA). GSA is inspired by the laws of gravity and mass interactions. In this implementation, we use the 'gravitational_search' operator with parameters 'gravity' set to 1.0 and 'alpha' set to 0.02. The selector used is 'all', which means that all operators will be applied in each iteration. This setup aims to explore the search space by simulating gravitational forces between mass-like particles, adjusting their positions based on their masses and distances.
# The Rastrigin function is chosen as the benchmark problem due to its multimodal nature, allowing for multiple local minima, which is suitable for a metaheuristic approach like GSA that can potentially discover various solutions.