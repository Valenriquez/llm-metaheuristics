# Name: natural_selection
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
            'scale': 0.8,
            'distribution': 'levy',
            'beta': 2.5
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.9,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic uses a combination of the random flight operator to explore the search space and the local random walk operator 
# to exploit the best solutions found so far. The random flight operator is used as the primary search operator, while the local 
# random walk operator is used as the secondary search operator. The 'greedy' selector is used to choose between these two operators, 
# while the 'metropolis' selector is used for the local random walk operator.

# We use the Ackley function as our target function because it has a single global optimum and a high degree of complexity.
# Our goal is to find a solution that minimizes this function's value.