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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm for optimization. 
# The problem is set to Rastrigin with dimension 2, which aims to minimize the function's values in a multi-dimensional space. 
# Two operators are employed: gravitational_search and random_flight. Gravitational search uses parameters gravity (1.0) and alpha (0.02), while random flight uses scale (1.0), distribution ('levy') and beta (1.5).
# The selector for both operators is set to 'all', indicating that these operators will be applied in every iteration of the metaheuristic. 
# The GravitationalSearchMetaheuristic aims to find the optimal solution by combining explorative gravitational search with random flight movements, leveraging probabilistic selection based on beta distribution parameters. This approach allows for a balance between exploration and exploitation across multiple iterations until convergence or termination criteria are met.