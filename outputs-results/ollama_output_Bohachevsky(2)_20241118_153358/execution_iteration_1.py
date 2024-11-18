# Name: Custom Metaheuristic
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.sombrero({3})
prob = fun.get_formatted_problem()

heur = [
    (
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
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'metropolis'
    )
]

heur.append(
    (
        'genetic_crossover',
        {
            'pairing': 'random',
            'crossover': 'blend',
            'mating_pool_factor': 0.4
        },
        'probabilistic'
    )
)

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# This metaheuristic uses the genetic mutation and gravitational search operators together with genetic crossover. 
# The selection of each operator was done based on their general performance in solving the benchmark function.
# It is worth mentioning that each problem may have different optimal operators, so this selection might not be accurate for other problems.

# The use of 'greedy' as selector for 'genetic_mutation' is because genetic mutation has a higher probability to improve the f(x) value. 
# On the other hand, 'metropolis' was used as selector for 'gravitational_search' since it often performs better than 'greedy' when convergence is prioritized.
# The selection of 'probabilistic' as selector for 'genetic_crossover' was chosen because crossover has a lower probability to improve the f(x) value but also produces more diverse solutions, and this could be beneficial in exploration.