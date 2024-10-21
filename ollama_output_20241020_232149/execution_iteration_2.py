 # Name: GravitationalSearchOptimization
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
# The chosen metaheuristic is Gravitational Search Optimization (GSO). 
# GSO mimics the gravitational force among particles in a system to search for optimal solutions.
# In this implementation, we use two operators:
# 1. Gravitational_search with parameters gravity set to 1.0 and alpha to 0.02.
#   - The gravitational constant controls the strength of the gravitational force affecting particle positions.
#   - Alpha is a scaling factor that affects how quickly particles converge towards better solutions.
#   - The selector 'metropolis' will be used in this case, which could help balance exploration and exploitation by accepting worse solutions with a probability based on their energy levels.
# 2. Random_flight with parameters scale set to 1.0, distribution as 'levy', and beta as 1.5.
#   - Scale determines the extent of random jumps in particle positions.
#   - The distribution parameter is set to 'levy' which introduces a more realistic exploration by following a levy flight pattern rather than uniform or Gaussian distributions.
#   - Beta controls the probability distribution function (PDF) shape, affecting the likelihood of larger steps compared to smaller ones during random flights.
#   - The selector 'probabilistic' is used as it allows for randomness in decision-making, which can be beneficial when dealing with complex and non-linear problems like Rastrigin function.
# These settings aim to balance between thorough exploration and efficient exploitation within the search space, utilizing both deterministic and probabilistic elements typical of a metaheuristic approach.