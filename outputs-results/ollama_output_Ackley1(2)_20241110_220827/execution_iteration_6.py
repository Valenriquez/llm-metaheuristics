# Name: Ackley_1_Swarm
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
    (
        'random_flight',
        {
            'scale': 0.9,
            'distribution': 'levy',
            'beta': 3
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'elite_rate': 0.1,
            'mutation_rate': 0.2,
            'scale': 0.8
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))