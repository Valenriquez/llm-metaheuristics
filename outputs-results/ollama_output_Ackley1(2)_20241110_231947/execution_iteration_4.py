# Name: ackley_metaheuristic
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
            'scale': 0.7,
            'distribution': 'gaussian',
            'beta': 3.0
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.8,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines the Random Flight and Local Random Walk operators to search for a good solution.
# The Random Flight operator is used as the primary search operator, with the parameters tuned to ensure a balance between exploration and exploitation.
# The Local Random Walk operator is used as a second search operator to explore local neighborhoods of the current solution.
# The greedy selector is used to select the next solution from the best operator in each iteration.