# Name: bochahybrid
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
        'random_sample',
        {
            'scale': 1.0,
            'distribution': 'levy'
        },
        'greedy'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'
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

heur3 = [
    (  
        'random_sample',
        {
            'scale': 1.0,
            'distribution': 'levy'
        },
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'metropolis'
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
print('x_best = {}, f_best = {}'.format(*met.get_solution()))
for it in range(3):
    met = mh.Metaheuristic(prob, heur3[it], num_iterations=5)
    met.verbose = False
    met.run()

print('x_best2 = {}, f_best = {}'.format(*met.get_solution()))