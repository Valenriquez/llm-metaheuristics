 # Name: GravitationalSearchOptimizer
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
# This code defines a metaheuristic named GravitationalSearchOptimizer using the gravitational search algorithm, which is characterized by its use of gravity as a parameter to guide the search towards better solutions. The first operator in this implementation is a gravitational search with parameters for gravity set to 1.0 and alpha to 0.02. This configuration aims to balance exploration and exploitation through probabilistic selection based on the Metropolis criterion, which is reflected by the selector 'metropolis'.
# The second operator is a random flight using Levy distribution, with scale set to 1.0 and beta at 1.5. This operator uses a probabilistic selector ('probabilistic') that allows for both deterministic and stochastic behavior in navigating through the search space, promoting exploration while considering the probability distribution settings.
# These operators are chosen based on their applicability to continuous optimization problems like those tackled by the gravitational search algorithm, ensuring diversity in the search approach with both explorative (random flight) and exploitative (gravitational search) components, as indicated by the selector parameters.