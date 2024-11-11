# Name: ackley_mh
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
    (  # Search operator 1
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
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
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# 
# The Ackley1 metaheuristic aims to find a global minimum of the Ackley function.
#
# The 'gravitational_search' operator is used as the search operator, which uses gravity to guide the search towards the better solutions.
# The 'spiral_dynamic' operator is used in combination with probabilistic selection to increase the diversity of the population and explore different regions of the search space.
#
# The goal is to get a smaller solution than 0 and a smaller fitness value than 0, which requires careful tuning of the parameters and operator selection.