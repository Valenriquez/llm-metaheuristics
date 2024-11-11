# Name: ackley1_mh
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
        'local_random_walk',
        {
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Explanation and justification:
#
# The Ackley1 metaheuristic was chosen because it's a classic test problem in the field of natural computing.
# 
# The local_random_walk operator was used to guide the search towards better solutions by allowing the algorithm
# to take small steps around the current solution. This helps avoid getting stuck in local optima.
# 
# The spiral_dynamic operator was used to provide some diversity in the search, helping to find multiple good solutions.
# The metropolis selector was chosen because it's a simple and effective method for selecting operators.
#
# In this metaheuristic, the number of iterations is fixed at 100. This can be adjusted depending on the problem size
# or desired level of convergence. 
#