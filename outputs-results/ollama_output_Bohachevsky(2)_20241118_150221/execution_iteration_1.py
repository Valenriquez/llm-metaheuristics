# Name: Hybrid Swarm Optimization Algorithm (HSA)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh


fun = bf.Bohachevsky(2)  # WRITE IT EXACTLY AS GIVEN, BE CAREFUL
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Particle Swarm Optimization (PSO)
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'  # Use a bigger selector for dimension 2
    ),
    (  
        'spiral_dynamic',  # Search operator 2: Spiral Dynamic
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'  # Use a bigger selector for dimension 2
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))


# Short explanation and justification:
# The Hybrid Swarm Optimization Algorithm combines two search operators to enhance the exploration and exploitation capabilities of the optimization process. 
# Particle Swarm Optimization (PSO) is used for global search, while Spiral Dynamic helps in fine-tuning near optimal solutions. Both are selected with appropriate parameters for a 2D problem.
# The 'all' selector allows both operators to cover the solution space effectively.