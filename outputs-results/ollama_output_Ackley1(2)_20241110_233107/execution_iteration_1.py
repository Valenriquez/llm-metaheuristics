# Name: ackley_mh
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
            'scale': 2.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic uses a combination of two search operators: 
# gravitational_search which uses gravity to attract the solution towards the global minimum,
# random_flight which uses randomness to explore the search space. Both are chosen for their ability to get smaller solutions.
# The greedy selector is used as it provides the best solution found so far at each iteration, 
# ensuring that we converge on a good local optimum rather than getting stuck in a local minimum.