# Name: Enhanced Swarm Metaheuristic with Inertial Version

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(2)
prob = fun.get_formatted_problem()

heur = [
    ('swarm_dynamic',
     {'factor': 0.8,
      'self_conf': 2.4,
      'swarm_conf': 2.5,
      'version': 'inertial',
      'distribution': 'gaussian'},
     'probabilistic'),
    ('random_sample', {}, 'metropolis')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))