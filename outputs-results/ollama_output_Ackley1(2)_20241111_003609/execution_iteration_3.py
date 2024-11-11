# Name: Ackley_1
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
        'local_random_walk',  # Changed the order of parameters and added another operator for comparison
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform',
            'beta': 1.5
        },
        'random_sample'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))
# Explanation and justification:
# The selected metaheuristic used two different search operators: 
# gravitational_search which uses gravity as a weighting factor for the fitness function,
# and local_random_walk, which allows the algorithm to make random changes in the space.
# These two operators are combined using 'greedy' and 'random_sample' selectors, respectively.