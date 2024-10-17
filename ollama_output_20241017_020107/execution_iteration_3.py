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
# The chosen metaheuristic is named GravitationalSearchMetaheuristic, which incorporates the gravitational search algorithm. 
# This algorithm simulates the physical principle of gravity to guide its search process. 
# We use two main operators: 'gravitational_search' and 'random_flight'. 
# The 'gravitational_search' operator is set with a gravity parameter of 1.0 and an alpha value of 0.02, applied across all possible solutions ('all'). 
# This helps in exploring the solution space more effectively by mimicking gravitational forces.
# The 'random_flight' operator uses a scale factor of 1.0 and follows a levy distribution, with a beta parameter of 1.5. 
# It is configured to operate probabilistically ('probabilistic') which allows for random movements in the search space while still influenced by gravity-like forces from the 'gravitational_search' operator.
# Together, these operators leverage both deterministic and probabilistic elements to navigate the optimization landscape efficiently.