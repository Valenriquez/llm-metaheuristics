# Name: Ackley1Meta
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
            'gravity': 0.01,
            'alpha': 0.01
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.9,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# I chose the gravitational search operator to guide the population towards the optimal solution.
# The local random walk operator is used in a probabilistic manner to explore new regions of the search space.
# This combination aims to find a better solution than the actual one, which is 0.