# Name: CustomMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 2.0,
            'distribution': 'uniform'
        },
        'greedy'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.8,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'all'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=200)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines three different search operators: local_random_walk, swarm_dynamic, and spiral_dynamic. 
# The local_random_walk helps in exploring the neighborhood of the current solution, while the swarm_dynamic mimics the behavior of a flock of birds to explore the space more globally.
# The spiral_dynamic operator is added to facilitate exploration and exploitation by guiding the search in a spiral manner.
# This combination is chosen because it allows for a good balance between exploration and exploitation, which is crucial for effectively solving optimization problems like Rastrigin's function.