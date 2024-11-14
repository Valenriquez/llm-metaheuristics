# Name: Alpine_2
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Alpine1(2)
prob = fun.get_formatted_problem()

heur = [
    (  
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
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
    ),
    (
        'random_flight',
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.5
        },
        'probabilistic'
    ),
    (
        'gravitational_search',
        {
            'gravity': 1.0,
            'alpha': 0.02
        },
        'metropolis'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))