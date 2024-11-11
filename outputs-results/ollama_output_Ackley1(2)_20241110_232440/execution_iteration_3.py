# Name: Ackley_Meta
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
            'gravity': 5.0,
            'alpha': 0.05
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'scale': 1.2,
            'elite_rate': 0.3,
            'mutation_rate': 0.45,
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))