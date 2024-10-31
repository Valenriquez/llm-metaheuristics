# Name: Swarm Metaheuristic with Gaussian Self-Confidence and Inertial Version

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
# We chose the swarm_dynamic operator with the Gaussian distribution for self-confidence.
# The inertial version of the swarm is recommended for this problem due to its ability to explore diverse solutions.
# The probabilistic selector ensures that the swarm explores the search space with a focus on promising solutions.