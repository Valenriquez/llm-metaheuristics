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
# The GravitationalSearchMetaheuristic is designed to solve optimization problems using a gravitational search algorithm. 
# It begins with the gravitational_search operator, which simulates the gravitational force between masses to explore the solution space.
# The parameters for this operator include 'gravity' set to 1.0 and 'alpha' set to 0.02. This configuration encourages diverse exploration of the search space.
# Additionally, a random_flight operator is included in the algorithm. It uses a scale factor of 1.0 with a distribution type set to 'levy'. 
# The parameter 'beta' is set to 1.5, which influences the direction and magnitude of the flight movements within the search space.
# This combination of operators allows for both structured exploration (gravitational_search) and more random exploration (random_flight), providing a balanced approach to problem-solving.
# The selector 'all' is used for both operators, indicating that these operators should be applied in every iteration of the metaheuristic process, ensuring comprehensive coverage of potential solutions across multiple iterations.