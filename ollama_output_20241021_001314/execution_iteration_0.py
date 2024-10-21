 # Name: GravitationalSearchOptimization
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

gravitational_search_params = {
    "gravity": 1.0,
    "alpha": 0.02
}

heur = [
    ( # Gravitational Search Operator
        'gravitational_search',
        gravitational_search_params,
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code implements a Gravitational Search Optimization (GSO) metaheuristic for the Rastrigin function with 2 dimensions. The GSO is characterized by two main parameters: gravity and alpha, which control the strength of gravitational attraction and the exploration vs exploitation balance, respectively. In this implementation, we use 'all' as the selector to apply the gravitational search operator to all particles in the population during each iteration. This approach allows for a comprehensive exploration of the solution space, potentially leading to better convergence properties compared to other selection strategies.
