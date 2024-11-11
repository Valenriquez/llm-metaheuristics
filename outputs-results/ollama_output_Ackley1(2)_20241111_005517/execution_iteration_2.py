# Name: spiral_silver
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
        'spiral_dynamic',
        {
            'radius': 0.95,
            'angle': 22.5,
            'sigma': 0.05
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.7,
            'scale': 1.2,
            'distribution': 'uniform'
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))