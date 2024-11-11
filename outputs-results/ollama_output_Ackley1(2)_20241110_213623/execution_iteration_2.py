# Name: Ackley_1_Metaheuristic
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: genetic_mutation
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'gravitational_search', # Search operator 2: gravitational_search
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'metropolis'
    ),
    (
        'local_random_walk', # Search operator 3: local_random_walk
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {{}}, f_best = {{}}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines three different search operators: genetic_mutation, gravitational_search and local_random_walk. 
# The genetic_mutation operator is used as the initial population for the search process. 
# The gravitational_search operator is used to update the positions of the particles in the population. 
# The local_random_walk operator is used to explore the search space more efficiently by randomly changing the position of a subset of particles.
# The selection of the operators and their parameters may vary depending on the case, as demonstrated by this example where three different combinations are used.