# Name: HybridMetaHeuristic
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
    (  # Search operator 1: Random Sample for exploration
        'random_sample',
        {
        },
        'all'
    ),
    (
        'swarm_dynamic',  # Search operator 2: Particle Swarm Optimization for exploitation
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'greedy'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()
print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# HybridMetaHeuristic combines the exploration capabilities of Random Sample with the exploitation power of Particle Swarm Optimization.
# The random_sample operator helps in exploring the search space thoroughly, while swarm_dynamic provides a focus on promising regions through informed movement.