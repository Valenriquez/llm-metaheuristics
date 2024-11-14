# Name: AlpineMeta
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Alpine1(2)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_flight',
        {
            'scale': 0.9,
            'distribution': 'levy',
            'beta': 4.5
        },
        'greedy'
    ),
    (
        'gravitational_search',
        {
            'gravity': 1.2,
            'alpha': 0.02
        },
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.8,
            'scale': 0.7,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))