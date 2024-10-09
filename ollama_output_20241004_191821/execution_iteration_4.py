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
# This code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm for optimization. 
# The benchmark function used is Rastrigin with two dimensions (Rastrigin(2)). 
# Two main operators are employed: gravitational_search and random_flight. 
# The gravitational_search uses parameters 'gravity' set to 1.0 and 'alpha' to 0.02, while the selector is set to 'metropolis'. 
# The random_flight utilizes 'scale' of 1.0, 'distribution' as 'levy', and 'beta' set to 1.5, with the selector 'probabilistic'.
# Both operators are selected based on their probabilistic characteristics, which aligns with the purpose of exploring diverse solution spaces in a metaheuristic approach.