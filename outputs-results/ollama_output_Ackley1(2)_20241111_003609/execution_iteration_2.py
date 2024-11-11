# Name: ackley
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
        'gravitational_search',
        {
            'gravity': 0.01,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'random_flight',
        {
            'scale': 0.5,
            'distribution': 'uniform',
            'beta': 1.5
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic uses a combination of the gravitational search and random flight operators to search for the global minimum of the Ackley function.
# The gravitational search operator is used as the initial search operator, while the random flight operator is used to explore the search space.
# The selection of these operators is based on their performance in previous runs, where the gravitational search operator performed better.