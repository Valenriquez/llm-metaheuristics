# Name: Ackley's Metaheuristic
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
        'random_flight',
        {
            'scale': 0.01,
            'distribution': 'uniform',
            'beta': 3.5
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.99,
            'angle': 30.0,
            'sigma': 0.05
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))