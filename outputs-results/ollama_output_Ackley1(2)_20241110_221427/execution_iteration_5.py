# Name: akley_meta
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2)
prob = fun.get_formatted_problem()

heur = [
    (
        'gravitational_search',
        {
            'gravity': 0.9,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.05,
            'mutation_rate': 0.2,
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))