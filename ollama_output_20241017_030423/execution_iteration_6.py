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
# The GravitationalSearchMetaheuristic is designed to optimize the Rastrigin function with a combination of gravitational search and random flight operators. 
# Gravitational search mimics the behavior of celestial bodies under the influence of gravity, exploring the solution space by simulating attractive forces between masses.
# Random flight operator introduces diversity in the search by allowing for stochastic movements based on specified distributions (levy in this case), which aids in escaping local minima and exploring new areas of the search space.
# Both operators are configured with parameters that reflect their specific behaviors: gravity strength, alpha factor for gravitational search, scale and distribution type for random flight, and beta coefficient.
# The 'all' selector is used to apply both greedy and probabilistic strategies during optimization, allowing for a balance between exploitation and exploration as required by the Rastrigin function's characteristics.