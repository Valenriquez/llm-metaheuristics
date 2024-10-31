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
    (  # Search operator 1
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
# The SwarmMetaheuristic utilizes the swarm_dynamic operator with the Levy distribution for particle distribution. The probabilistic selector ensures that particles are selected based on their probability of success, promoting diversity and exploration. The factor parameter controls the speed of the swarm, while the self_conf, swarm_conf parameters influence the confidence levels for individual and group decisions. The inertial version of the swarm dynamics is chosen for its ability to navigate complex landscapes.