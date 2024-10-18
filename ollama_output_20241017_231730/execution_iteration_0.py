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
# The GravitationalSearchMetaheuristic is designed to optimize the Rastrigin function with two main operators: gravitational_search and random_flight. 
# Gravitational search uses parameters 'gravity' set to 1.0 and 'alpha' set to 0.02, while the selector for this operator is set to 'metropolis'. 
# Random flight employs 'scale' of 1.0, distribution type 'levy', and 'beta' set to 1.5, with a selector set to 'probabilistic'.
# These operators are chosen based on their ability to explore the solution space effectively through gravitational dynamics and probabilistic random flights, respectively, as supported by the parameters specified in the parameters_to_take.txt file.
