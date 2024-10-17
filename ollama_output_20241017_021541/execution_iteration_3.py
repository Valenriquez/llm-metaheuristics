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
    'metropolis'
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
# The algorithm is named GravitationalSearchMetaheuristic based on the parameters provided in the configuration file, which includes gravitational search with gravity set to 1.0 and alpha to 0.02. Additionally, a random flight operator is included with scale set to 1.0 and distribution as levy, using probabilistic selector for exploration. These settings are chosen to balance between exploration and exploitation based on typical behaviors of these operators in optimization tasks. The GravitationalSearchMetaheuristic combines both local and global search capabilities by integrating the gravitational search and random flight mechanisms, which is beneficial for exploring diverse areas of the solution space while fine-tuning towards an optimal solution.
