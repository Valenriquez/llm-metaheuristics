# Name: ackley_optimiser
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.9,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'probabilistic'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.95,
            'angle': 22.5,
            'sigma': 0.05
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))