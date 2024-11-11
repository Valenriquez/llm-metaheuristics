# Name: ackleymeta
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
            'gravity': 10,
            'alpha': 0.01
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 1.2,
            'angle': 25,
            'sigma': 0.05
        },
        'probabilistic'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.8,
            'scale': 1.5,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))