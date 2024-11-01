# Name: SwarmMetaheuristic

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
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'levy'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The SwarmMetaheuristic uses the swarm_dynamic operator with the following parameters:
# - factor: 0.7, which specifies the factor for the swarm's inertia.
# - self_conf: 2.54, which is the confidence threshold for individual solutions.
# - swarm_conf: 2.56, which is the confidence threshold for the swarm as a whole.
# - version: inertial, which specifies the version of the swarm algorithm.
# - distribution: levy, which specifies the distribution of the swarm's positions.