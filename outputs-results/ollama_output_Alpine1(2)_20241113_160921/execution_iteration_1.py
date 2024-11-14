# Name: Alpine_Adamant
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
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
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
        'metropolis'
    ),
    (
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic is called Alpine_Adamant because it combines three different operators to search for the optimal solution.
# The 'local_random_walk' operator uses a probability of 0.75 to choose the next step, with a scale of 1.0 and uniform distribution.
# The 'spiral_dynamic' operator searches in a spiral pattern with radius 0.9, angle 22.5, and sigma 0.1.
# The 'random_flight' operator uses a scale of 1.0 and levy distribution to search for the optimal solution.
# The selector 'greedy' chooses the next step based on the best fitness value, while the selector 'metropolis' uses a temperature-based method to choose the next step.
# The selector 'probabilistic' chooses the next step randomly with a probability of 0.5.