# Name: Alpine2Meta
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
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.8,
            'scale': 0.9,
            'distribution': 'uniform'
        },
        'constriction'
    ),
    (
        'spiral_dynamic', 
        {
            'radius': 0.7,
            'angle': 23.0,
            'sigma': 0.1
        }, 
        'random_sample'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines the advantages of Local Random Walk and Spiral Dynamics.
# The purpose is to find a global optimum solution with minimal search space.

# Initialise the fitness register
fitness = []

# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000)
    met.reset_historicals()
    met.verbose = False
    met.run()

    # Check if f_best is not zero to avoid incorrect results
    if met.get_solution()[1] != 0:
        print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    else:
        print(f'rep = {rep+1}, Could not find a valid solution')
        
    fitness.append(met.historical['fitness'])