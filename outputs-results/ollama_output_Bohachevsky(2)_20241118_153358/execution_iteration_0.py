# Name: My Metaheuristic
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.benchmark_function({6})
prob = fun.get_formatted_problem()

heur = [
    (
        'genetic_crossover',
        {
            'crossover': 'single',
            'mating_pool_factor': 0.4,
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'scale': 1.0,
            ' elite_rate': 0.1,
            'mutation_rate': 0.25,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Genetic Crossover and mutation are used together because genetic_crossover is only applicable when using genetic_mutation. 
# The selection of the first operator depends on the case in which a higher probability is given to be chosen.
# In this example we use 'greedy' as selector for all operators, but other selectors like 'metropolis', or 'probabilistic' can also be used.