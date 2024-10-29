 # Name: GravitationalSearchAndRandomFlightMetaheuristic
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
    ),
    ( # Random Flight Operator
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
# This metaheuristic combines two main operators: Gravitational Search and Random Flight. 
# The gravitational search is designed to simulate the behavior of celestial bodies under gravity, using parameters such as gravity and alpha to influence the exploration and exploitation of the solution space. 
# In contrast, the random flight operator introduces a probabilistic element by allowing movements based on a levy distribution scaled by beta, suitable for exploring new areas in the search space.
# These operators are selected to run across all possible solutions ('all') in order to balance between global exploration and local exploitation effectively.
# The combination of gravitational search's focused exploitation with random flight's broad exploration aims to improve the overall performance of the metaheuristic by ensuring a thorough examination of potential solution areas without getting stuck in局部最优解.