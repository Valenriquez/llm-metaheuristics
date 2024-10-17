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
# The GravitationalSearchMetaheuristic is designed to optimize the Rastrigin function with a combination of two main operators: gravitational search and random flight. 
# Gravitational search operates under the influence of gravity, aiming to simulate the physical laws of gravitation for exploration in the solution space. Parameters include gravity (set at 1.0) and alpha (set at 0.02).
# Random flight is another operator that uses a probabilistic approach, allowing the algorithm to perform random movements scaled by beta (set at 1.5), with distribution set as 'levy' for exploring diverse regions of the problem space.
# Both operators are configured to operate across all possible solutions ('all') and utilize a probabilistic selector to determine their application based on predefined probabilities derived from each operator's properties, which is crucial for balancing exploration and exploitation in the optimization process. This setup aims to enhance convergence speed and solution quality by dynamically adjusting search strategies during iterations.