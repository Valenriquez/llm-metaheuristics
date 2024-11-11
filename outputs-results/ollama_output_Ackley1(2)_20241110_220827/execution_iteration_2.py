# Name: AckleyMeta
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
    (  # Search operator 1
        'random_flight',
        {
            'scale': 0.5,
            'distribution': 'levy',
            'beta': 2.0
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.8,
            'angle': 25.0,
            'sigma': 0.1
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Ackley function is a multimodal test problem used to assess the performance of optimization algorithms.
# The goal of this metaheuristic is to find a better solution than 0, i.e., x_best < 0, and a better fitness than 0, i.e., f_best < 0.
# We have selected two search operators: random_flight and spiral_dynamic, which are combined using the greedy and metropolis selectors.