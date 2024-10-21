 # Name: GravitationalSearchMetaheuristic
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
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic is named GravitationalSearchMetaheuristic as it combines the gravitational search algorithm with probabilistic selection based on the parameters provided. 
# We use the Rastrigin function for benchmarking, which is a standard test problem in optimization.
# The heuristic includes two operators: gravitational_search and random_flight. The gravitational_search uses gravity and alpha parameters to influence the search direction, while the random_flight operator explores the solution space with scale, distribution (levy), and beta values. 
# Both operators use 'all' as the selector, meaning they will be applied in every iteration of the metaheuristic algorithm. This setup allows for a balance between exploration and exploitation during optimization. The GravitationalSearchMetaheuristic aims to leverage both deterministic and probabilistic elements from the provided operators to find an optimal solution efficiently.