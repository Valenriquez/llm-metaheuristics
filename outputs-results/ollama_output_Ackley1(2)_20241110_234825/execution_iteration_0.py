# Name: akleh_meta
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
            'gravity': 1.0,
            'alpha': 0.02
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'random'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic "akleh_meta" uses a combination of gravitational search and local random walk to find the optimal solution.
#   *   Gravitational search is used as the primary search operator, with a gravity value of 1.0 and alpha value of 0.02.
#   *   Local random walk is used as the secondary search operator, with a probability of 0.75, scale of 1.0, and gaussian distribution.
# The metaheuristic runs for 100 iterations and prints the optimal solution found by the algorithm.