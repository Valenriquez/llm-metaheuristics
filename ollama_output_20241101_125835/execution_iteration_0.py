# Name: Swarm Metaheuristic with Inertial Version and Gaussian Distribution

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
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic uses the Swarm Dynamic operator with an inertial version and a Gaussian distribution for the swarm positions.
# The Gaussian distribution ensures that the swarm particles are spread out and explore different areas of the search space.
# The inertial version provides a balance between exploration and exploitation, encouraging the swarm to move around the search space while still attempting to find the best solution.