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
# The metaheuristic named GravitationalSearchAndRandomFlightMetaheuristic combines two main search operators, gravitational search and random flight, to solve the optimization problem using a set of parameters defined in the parameters_to_take.txt file. 
# Gravitational search is configured with gravity set to 1.0 and alpha at 0.02, aiming to simulate the gravitational force acting on particles within a multi-dimensional space. This operator uses all possible solutions ('all') as selectors to guide its exploration across the solution space.
# Random flight, on the other hand, is configured with a scale of 1.0 and distribution set to 'levy', which aids in the exploration by introducing random jumps. The probabilistic selector allows for a controlled randomness in the search process, ensuring diverse exploration while still aiming for better solutions through probability-based decisions.
# Both operators are designed to leverage different aspects of the optimization problem: gravitational search focuses on convergence and exploitation due to its deterministic nature, while random flight promotes diversification and exploration by introducing unpredictable elements into the solution space. This combination helps in navigating the landscape of possible solutions more effectively, balancing between exploring new areas and refining current best solutions found through gravitational attraction.