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
# This code defines a metaheuristic named GravitationalSearchOptimizer, which is inspired by the gravitational search algorithm (GSA). The algorithm includes two main operators: 'gravitational_search' and 'random_flight'. 
# The 'gravitational_search' operator uses parameters 'gravity' set to 1.0 and 'alpha' set to 0.02, while the 'random_flight' operator uses 'scale' set to 1.0 with a distribution type of 'levy' and 'beta' value of 1.5.
# The selectors for both operators are set to 'metropolis', which indicates that the algorithm will use probabilistic selection based on Metropolis criteria during its search process. This combination aims to balance between exploration and exploitation, allowing the algorithm to effectively navigate through the search space while optimizing the Rastrigin function. 
# The choice of these parameters and selectors is based on typical settings for gravitational search algorithms and the characteristics of the problem at hand, which in this case is the Rastrigin function known for its multimodal nature requiring exploration across multiple modes to achieve an optimal solution.