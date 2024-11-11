# Name: simple_example
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
            'gravity': 10,
            'alpha': 0.01
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.8,
            'scale': 1.2,
            'distribution': 'gaussian'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The metaheuristic uses a combination of gravitational search and local random walk operators to navigate the search space. 
# The gravitational search operator pulls the solution towards the global optimum, while the local random walk operator explores the surrounding region.
# This combination allows the algorithm to balance exploration and exploitation, leading to a smaller fitness solution than the initial one.