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
# The GravitationalSearchMetaheuristic is designed to solve optimization problems using a gravitational search algorithm inspired by the physics of gravity. 
# We start with a simple problem, Rastrigin function, which has two dimensions (x, y). 
# Heuristics are defined as described in parameters_to_take.txt: 
# The first operator is Gravitational Search with parameters 'gravity' set to 1.0 and 'alpha' set to 0.02, using a Metropolis selector for probabilistic selection.
# The second operator is Random Flight which uses a scale of 1.0 and distribution type 'levy', along with beta set to 1.5, selected probabilistically ('probabilistic').
# These operators are chosen based on the characteristics of their parameters that aim to explore and exploit the solution space for optimization.
# The algorithm is run for 100 iterations, and results including the best found solutions' coordinates (x_best) and its value (f_best) are printed at the end.