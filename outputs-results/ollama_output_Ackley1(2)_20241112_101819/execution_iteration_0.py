# Name: ackley_modified
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'gravitational_search',
        {
            'gravity': 0.99,
            'alpha': 0.01
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.95,
            'angle': 25,
            'sigma': 0.05
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))