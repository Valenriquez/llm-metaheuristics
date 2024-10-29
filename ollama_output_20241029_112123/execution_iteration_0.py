 # Name: GravitationalSearchAndRandomFlight
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Search operator 1
    'gravitational_search',
    { 
        'gravity': 1.0,
        'alpha': 0.02
    },
    'all'
    ),
    (  
    'random_flight',
    {
        'scale': 1.0,
        'distribution': 'levy',
        'beta': 1.5
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The chosen metaheuristic combines the gravitational search and random flight operators to solve the Rastrigin function optimization problem. Gravitational search uses a gravity parameter (1.0) and an alpha value (0.02), while random flight involves scaling (1.0), distribution type ('levy'), and beta factor (1.5). The selectors used are 'all' for gravitational search and 'probabilistic' for the random flight, which is appropriate because it includes a probability-based selection mechanism. This combination aims to leverage both global exploration through gravitational forces and local exploitation with adaptive mutations in the random flight phase, potentially leading to better convergence towards the optimal solution of the Rastrigin function.