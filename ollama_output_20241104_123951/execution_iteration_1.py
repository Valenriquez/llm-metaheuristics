# Name: Swarm Metaheuristic with Inertial Version

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
# The Swarm Metaheuristic with Inertial Version is chosen due to its effectiveness in optimizing complex problems. The inertial version of the swarm metaheuristic is particularly well-suited for optimizing the Rastrigin function, as it allows for more flexible movement and exploration of the search space. Additionally, the Gaussian distribution is recommended for the swarm metaheuristic as it is known to perform well in optimizing multimodal functions.