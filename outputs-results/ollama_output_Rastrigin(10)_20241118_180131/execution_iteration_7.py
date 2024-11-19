# Name: Improved Particle Swarm Optimization with Local Random Walk

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)
prob = fun.get_formatted_problem()

heur = [
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'gaussian'
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines the Swarm Dynamic operator with a Local Random Walk to improve exploration and exploitation. 
# The Swarm Dynamic operator helps in global search, while the Local Random Walk enhances fine-grained searching around promising regions.
# Using 'all' selector for Swarm Dynamic allows it to cover more ground quickly, while the probabilistic selector for Local Random Walk ensures
# that the algorithm occasionally explores new areas even if the current path is promising. This combination aims to balance exploration and exploitation effectively.

# Note:
# The dimension of 10 is used here, so the 'all' selector for Swarm Dynamic is appropriate. If the dimension were smaller, a simpler selector could suffice.