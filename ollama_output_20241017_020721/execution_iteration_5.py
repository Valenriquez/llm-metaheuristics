 # Name: GravitationalSearchMetaheuristic
# Code:

import sys
sys.path.append('/Users/valeriaenriquezlimon/Documents/research-llm/llm-metaheuristics')
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ( # Gravitational Search Operator
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
# The GravitationalSearchMetaheuristic is designed to mimic the gravitational search mechanism, where "gravity" represents the strength of attraction between particles, controlling their movement towards better solutions. Parameters include 'gravity' set at 1.0 and 'alpha' set at 0.02. This encourages convergence toward optimal regions in the solution space.
# Additionally, a RandomFlight operator is introduced using the "levy" distribution to explore distant areas of the search space, with scale set to 1.0 and beta adjusted to 1.5 for probabilistic exploration. This combination aims to balance between local exploitation and global exploration effectively.
# The selectors are chosen as 'metropolis' for both operators. Metropolis selection helps in determining whether a move should be accepted or rejected based on the acceptance probability, which is crucial for maintaining diversity in search while allowing occasional jumps to explore new regions.