 # Name: CustomMetaheuristic
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
# This metaheuristic uses two main search operators: gravitational_search and random_flight. 
# The gravitational_search operator is configured with a gravity value of 1.0 and an alpha parameter set to 0.02, which are parameters related to the force applied during the search. 
# The selector for this operator is 'metropolis', indicating that it will use probabilistic selection criteria.
# The random_flight operator has its scale set to 1.0, with a distribution type of 'levy' and a beta parameter of 1.5. This operator uses a probabilistic selector as well ('probabilistic').
# These operators are chosen based on their ability to handle different types of search spaces and probabilities, which is suitable for exploring the solution space of the Sphere benchmark function with two dimensions.