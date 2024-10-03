 # Name: PSO_GravitationalSearch
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
# The algorithm is designed to use a Gravitational Search (GSA) combined with a Random Flight operator for optimization. 
# Gravitational search mimics the gravitational force among particles in a multi-dimensional space, encouraging them towards higher fitness regions. 
# This approach aims to balance exploration (random flight) and exploitation (gravitational attraction).
# Parameters:
# - 'gravity': A constant that defines the strength of the gravitational pull between particles; set to 1.0 for full force.
# - 'alpha': A coefficient that controls the scaling of the forces, influencing the speed at which solutions converge. Set to 0.02, typical for balancing convergence and exploration.
# - 'scale': The magnitude of random flight steps; here set to 1.0 for standard deviation in search directions.
# - 'distribution': Defines the type of noise added during random flights, using Levy distribution for potential benefits in diverse landscapes.
# - 'beta': Exponent that scales the strength of the random flight component, useful when a non-uniform exploration is desired.
# Selector:
# - 'all' allows both probabilistic and greedy selection criteria to be applied across all operators, ensuring a comprehensive search strategy.
# This setup aims to leverage strengths of both methods for effective optimization in complex domains, especially those with rugged fitness landscapes.