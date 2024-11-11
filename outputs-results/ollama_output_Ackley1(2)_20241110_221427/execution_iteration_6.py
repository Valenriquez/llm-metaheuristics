# Name: spir_mh
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
        'spiral_dynamic',
        {
            'sigma': 0.05,
            'radius': 0.99,
            'angle': 22.5
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'elite_rate': 0.1,
            'mutation_rate': 0.25,
            'scale': 0.9
        },
        'metropolis'
    ),
    (
        'random_flight',
        {
            'beta': 1.5,
            'scale': 0.99,
            'distribution': 'levy'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))