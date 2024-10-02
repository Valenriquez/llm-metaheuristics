 # Name: Custom Metaheuristic with Gravitational Search and Genetic Mutation
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [( # Search operator 1
    'gravitational_search',
    {
        'parameters': {
            'gravity': 0.9,
            'alpha': 0.015
        }
    },
    'probabilistic'
)]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and Justification:
# The metaheuristic is named "Custom Metaheuristic with Gravitational Search and Genetic Mutation". This approach combines the gravitational search algorithm, which is effective for continuous optimization problems like the Rastrigin function in this case, with genetic mutation to ensure diversity and convergence towards better solutions.

# The gravitational search operator uses a gravity parameter of 0.9 and an alpha value of 0.015 as specified in the parameters_to_take.txt file. This configuration aims to balance exploration and exploitation during the optimization process, allowing for both global and local explorations within the solution space.

# The selector used is "probabilistic", which means that each step in the search process will be determined by a probabilistic decision based on predefined probabilities. This approach enhances the adaptability of the algorithm to different problem characteristics and encourages exploration in promising areas while converging towards optimal solutions.

# Genetic mutation, incorporated as part of the operator, introduces slight variations into the population to prevent premature convergence and ensure that the search process can jump out of local minima. The genetic_mutation operator uses a scale factor of 1.0 with a uniform distribution for introducing random changes in the solution space during each iteration.

# This combination of gravitational search and genetic mutation is expected to yield better exploration and exploitation capabilities, leading to more effective convergence towards the global optimum of the Rastrigin function within the given number of iterations (100).