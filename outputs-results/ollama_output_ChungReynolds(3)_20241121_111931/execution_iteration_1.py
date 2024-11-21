# Name: Spiral-Dynamic Swarm Hybrid Metaheuristic

# Code:

import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.ChungReynolds(3)  # Change this line based on the provided benchmark function and dimension.
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.4180393565758949,
            'angle': 25,
            'sigma': 0.3044470463928741
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Initialise the fitness register
fitness = []
# Run the metaheuristic with the same problem 30 times
for rep in range(30):
    met = mh.Metaheuristic(prob, heur, num_iterations=1000, num_agents=2)
    met.reset_historicals()
    met.verbose = False
    met.run()
    print('rep = {}, x_best = {}, f_best = {}'.format(rep+1, *met.get_solution()))
    
    fitness.append(met.historical['fitness'])
    
# Short explanation and justification:
# This metaheuristic combines the Spiral Dynamic algorithm with a variant of the Swarm Dynamic algorithm to improve exploration and exploitation capabilities. The Spiral Dynamic helps in fine-tuning solutions in promising regions, while the Swarm Dynamic allows for broader search. This hybrid approach is expected to enhance convergence towards optimal solutions.

# In case there was an error, please fix it.