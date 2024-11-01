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
    ('swarm_dynamic',
     {
         'factor': 0.7,
         'self_conf': 2.54,
         'swarm_conf': 2.56,
         'version': 'inertial',
         'distribution': 'uniform'
     },
     'probabilistic'),
    ('random_sample',
     {},
     'metropolis')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The SwarmMetaheuristic uses the swarm_dynamic operator with the parameters factor=0.7, self_conf=2.54, swarm_conf=2.56, version=inertial, and distribution=uniform.
# The swarm_dynamic operator is a good choice for the Rastrigin(2) problem because it can explore the search space effectively and find good solutions.
# The random_sample operator is used as a backup operator in case the swarm_dynamic operator fails to find a good solution.