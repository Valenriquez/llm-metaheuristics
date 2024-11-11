# Name: ackley_one_point
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
    (
        'gravitational_search',
        {
            'gravity': 0.5,
            'alpha': 0.01
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 1.0,
            'angle': 45.0,
            'sigma': 0.05
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The AckleyOnePoint metaheuristic is designed to get a smaller solution than 0 and a smaller fitness than 0. 
# It uses the Gravitational Search operator for local optimization and the Spiral Dynamic operator for global exploration.
# The Gravitational Search operator is used with a reduced gravity force to encourage convergence, while the Spiral Dynamic operator is used with a large sigma value to increase exploration.