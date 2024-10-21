 # Name: GravitationalSearchAlgorithm
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
    ),
    ( # Search operator 3
    'local_random_walk',
    {
        'probability': 0.75,
        'scale': 1.0,
        'distribution': 'gaussian'
    },
    'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The GravitationalSearchAlgorithm is a metaheuristic inspired by the laws of gravity, where solutions are attracted to better regions based on their fitness values. We start with gravitational_search, setting gravity to 1.0 and alpha to 0.02, covering all possible solutions ('all'). Next, we introduce random_flight with scale set to 1.0 and distribution as 'levy', favoring probabilistic exploration. Finally, local_random_walk is used with a probability of 0.75 for mutation and 'gaussian' distribution for exploring the solution space more dynamically. These operators are chosen based on their ability to balance exploration and exploitation, which is crucial for optimizing complex functions like Rastrigin's function.
