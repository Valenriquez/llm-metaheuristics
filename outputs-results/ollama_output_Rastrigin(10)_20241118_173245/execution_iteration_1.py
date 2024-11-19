# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'
    ),
    (
        'random_sample',
        {},
        'metropolis'
    ),
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
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The HybridMetaheuristic combines three different search operators: local_random_walk, random_sample, and swarm_dynamic.
# local_random_walk is used to explore the neighborhood of the current solution, while maintaining a balance between exploration and exploitation.
# random_sample allows for global exploration by randomly selecting solutions from the solution space, which can help escape local minima.
# swarm_dynamic simulates the behavior of a swarm of particles, where each particle moves towards better solutions based on its own experience and the collective experience of the swarm.
# By combining these operators in this manner, we aim to create a more robust search algorithm that can effectively tackle complex optimization problems.