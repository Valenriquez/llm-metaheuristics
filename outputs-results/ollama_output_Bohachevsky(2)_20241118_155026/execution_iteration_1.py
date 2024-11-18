# Name: bohaceva
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2)
prob = fun.get_formatted_problem()

heur = [
    (
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'all'
    ),
    (
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# The Bohachevsky function is a 3-dimensional minimization problem. 
# In this case we are using the 2-dim version as it was provided to us.
# Random Flight algorithm will be our first operator and Gravitational Search our second.
# They have been chosen because they both use vectors, 
# which makes them easier to implement on higher dimensional spaces.
# Also because they do not need to look at every point in the space,
# that can make a big difference when dealing with big amounts of points.