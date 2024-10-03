 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Sphere(2)
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
# The GravitationalSearchMetaheuristic is designed to optimize the Sphere function with two variables using a combination of gravitational search and random flight operators. 
# The gravitational search operator utilizes gravity as its primary parameter, setting it at 1.0, while alpha is set to 0.02. This search method will operate on all possible solutions ('all' selector).
# The random flight operator introduces a scale factor of 1.0 and uses a levy distribution for exploration. It also applies a beta value of 1.5, which influences the probabilistic nature of this operator. This setting is appropriate to balance between exploration and exploitation during optimization.
# These parameters are chosen based on typical values recommended for gravitational search (gravity) and random flight (scale and distribution). The 'probabilistic' selector allows the random flight operator to operate with a chance-based approach, which can help in escaping local minima by exploring more diverse areas of the search space.