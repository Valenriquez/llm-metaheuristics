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
# This implementation uses the Gravitational Search Algorithm (GSA), a bio-inspired algorithm inspired by the laws of gravity and mass interactions, to solve optimization problems. The GSA is designed with two main operators: gravitational search and random flight.
# Gravitational search involves simulating the force of attraction between masses proportional to their masses and inversely proportional to the square of the distance between them. This operator uses parameters 'gravity' (representing the strength of gravity) and 'alpha' which controls the scaling factor for the velocity updates. The selector used is 'all', meaning this operator will be applied to all particles in the search space during each iteration.
# Random flight simulates random movements influenced by a scaling factor and distribution type ('levy', 'uniform', or 'gaussian'). This helps in exploring diverse regions of the search space, with 'beta' controlling the intensity of these random flights. The selector used here is 'probabilistic', indicating that this operator will be applied probabilistically during each iteration to enhance exploration while preserving some elements of exploitation.
# These operators together aim to balance between global and local explorations, leveraging both gravitational attraction for convergence towards promising regions and random flight for escaping local minima, thus making the algorithm effective in finding optimal solutions across a range of complex optimization problems.