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
# The code defines a metaheuristic named GravitationalSearchMetaheuristic using the gravitational search algorithm, which is characterized by its 'gravity' parameter controlling the strength of the gravitational force and 'alpha' affecting the scaling of the movement. Additionally, it incorporates random flight behavior with 'scale', 'distribution' set to 'levy', and 'beta'. The selector for these operators is set to 'metropolis' for probabilistic exploration based on a predefined probability distribution.
# This setup aims to leverage both deterministic gravitational search properties and stochastic random flight movements within the metaheuristic framework, providing a balanced approach between global and local explorations as defined by the parameters chosen.