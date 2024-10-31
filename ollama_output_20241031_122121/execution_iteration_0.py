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
# We choose the swarm_dynamic operator because it is known to be effective for continuous optimization problems.
# The Gaussian distribution is used for the initial swarm positions.
# The factor of 0.7 is recommended for a balance between exploration and exploitation.
# The self-confidence and swarm-confidence parameters influence the influence of social and cognitive factors in the swarm dynamics.
# The inertial version of swarm dynamics is used in this metaheuristic.