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
            'distribution': 'gaussian',
            'beta': 3.0
        },
        'greedy'
    ),
    (
        'genetic_mutation',
        {
            'elite_rate': 0.05,
            'mutation_rate': 0.1,
            'scale': 0.5,
            'distribution': 'gaussian'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The AckleyMeta metaheuristic uses a combination of random flight and genetic mutation operators.
# Random flight provides an initial exploration of the solution space, while genetic mutation introduces local search to improve convergence.
# We chose a greedy selector to balance between exploration and exploitation.