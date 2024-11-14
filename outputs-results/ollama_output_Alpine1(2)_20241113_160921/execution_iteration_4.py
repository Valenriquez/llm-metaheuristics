# Name: alpine_meta
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
        'random_sample',
        {
            # parameters for random_sample operator
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.95,
            'angle': 22.8,
            'sigma': 0.1
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(met.get_solution()[0], met.get_solution()[1]))