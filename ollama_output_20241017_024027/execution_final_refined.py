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
# This code implements a Gravitational Search Algorithm (GSA) for optimization. 
# The GSA is a metaheuristic inspired by the laws of gravity and consists of simulating the gravitational attraction between masses to search for optimal solutions.
# In this specific implementation, we use the gravitational_search operator with predefined parameters: 'gravity' set to 1.0 and 'alpha' set to 0.02.
# The selector is set to 'all', indicating that all agents should be updated according to the gravitational force model during each iteration.
# The algorithm runs for a total of 100 iterations, with verbose output enabled to track the progress.
# The results include the best solution found and its corresponding fitness value after the optimization process completes.