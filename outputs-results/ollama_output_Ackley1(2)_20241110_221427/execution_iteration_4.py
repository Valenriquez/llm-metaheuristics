# Name: ackley_met
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
        'gravitational_search',
        {
            'gravity': 10.0,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'random_flight', 
        {
            'scale': 100.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'metropolis'
    ),
    ( 
        'genetic_mutation',  
        {
            'elite_rate': 0.05,
            'mutation_rate': 0.25,
            'scale': 100.0
        },
        'probabilistic'
    ),
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))