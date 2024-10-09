 # Name: GravitationalSearchAlgorithm
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
    ( # Local Random Walk Operator
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Gravitational Search Algorithm (GSA) is a metaheuristic inspired by the principles of gravitational force and mass interactions, which are used to model the behavior of celestial bodies. In this implementation, we use three main operators: 'gravitational_search' with parameters gravity set to 1.0 and alpha to 0.02, 'random_flight' with scale set to 1.0 and distribution as levy, beta set to 1.5, and 'local_random_walk' with probability of 0.75, scale also set to 1.0, and distribution as uniform. The selectors for these operators are chosen based on their probabilistic behavior ('probabilistic' for random flight and local random walk), while the gravitational search uses 'all' to explore the entire population. This combination aims to leverage both global exploration through gravitational forces and local refinement through random walks, suitable for a diverse range of optimization problems.
