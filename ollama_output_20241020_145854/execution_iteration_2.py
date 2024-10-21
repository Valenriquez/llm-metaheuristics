 # Name: GravitationalSearchOptimizer
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin
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
        'distribution': "levy",
        'beta': 1.5
    },
    'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This code defines a metaheuristic named GravitationalSearchOptimizer using the Gravitational Search algorithm, which is inspired by gravitational mechanics to explore the search space. The first operator used is 'gravitational_search' with parameters gravity set to 1.0 and alpha set to 0.02. This operator uses all possible solutions in the population for exploration. The second operator is 'random_flight', configured with scale set to 1.0, distribution as 'levy', and beta set to 1.5. It operates in a probabilistic mode, applying random flight patterns to enhance the search capabilities by allowing broader exploration of the solution space. Together, these operators leverage both gravitational attraction for convergence and random flight for diversification within the optimization process.
