 # Name: GravitationalSearchAlgorithm
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
# The Gravitational Search Algorithm (GSA) is a metaheuristic inspired by the principles of gravitational force in physics. 
# In this implementation, the first search operator uses 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02. 
# The selector for this operator is 'metropolis', which suggests a probabilistic approach where the selection of candidate solutions is governed by probability rules.
# The second search operator is 'random_flight' with parameters scale set to 1.0, distribution type as 'levy', and beta as 1.5. 
# The selector for this operator is 'probabilistic', indicating that it will operate under a probabilistic framework similar to the first operator but potentially using different probability rules depending on the implementation details not specified here.
# Both operators are designed to explore the solution space in a way that combines elements of both deterministic and probabilistic search strategies, which should be beneficial for finding optimal solutions to continuous optimization problems like the Sphere function used in this example.