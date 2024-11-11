# Name: Ackley2Meta
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
        'random_flight',
        {
            'scale': 0.8,
            'distribution': 'gaussian',
            'beta': 1.5
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'scale': 1.0,
            'elite_rate': 0.2,
            'mutation_rate': 0.3,
            'distribution': 'uniform'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Ackley2Meta metaheuristic combines the random flight operator to explore new solutions with a more aggressive search radius,
# and the genetic mutation operator to modify existing solutions using uniform distribution for better local search.
# 
# Initially, we will only use the Ackley1 function as the benchmark function.