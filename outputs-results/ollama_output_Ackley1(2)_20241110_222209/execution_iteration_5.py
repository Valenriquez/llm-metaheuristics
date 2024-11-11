# Name: ackley_modified
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
        'local_random_walk',
        {
            'scale': 0.5,
            'probability': 0.8,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'mutation_rate': 0.2,
            'elite_rate': 0.1,
            'scale': 1.0
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))