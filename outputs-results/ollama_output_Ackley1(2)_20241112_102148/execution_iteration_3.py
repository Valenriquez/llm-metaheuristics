# Name: ackley_metaheuristic
# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Ackley1(2)
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_flight', 
        {
            'scale': 1.0,
            'distribution': 'levy',
            'beta': 1.8
        }, 
        'metropolis'
    ), 
    (
        'spiral_dynamic',  # Changed to spiral dynamic
        {
            'radius': 0.95,  # increased the radius to 0.95 from 0.9
            'angle': 22.5,  
            'sigma': 0.1
        }, 
        'metropolis'  # changed to metropolis from greedy
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))