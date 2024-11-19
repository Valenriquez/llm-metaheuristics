# Name: Hybrid Metaheuristic for Rastrigin Function
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
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
            'self_conf': 2.3,
            'swarm_conf': 2.4,
            'version': 'inertial',
            'distribution': 'gaussian'
        },
        'metropolis'
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=300)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines three different search operators to effectively explore the Rastrigin function landscape. 
# The `local_random_walk` operator helps in making small, informed steps to find a better local optimum.
# The `swarm_dynamic` operator mimics the behavior of particles in a flock, allowing for both exploration and exploitation phases.
# The `spiral_dynamic` operator encourages movement in spiral paths, which can help escape plateaus and find new areas of interest.
# Each operator is equipped with different parameters to optimize their performance on the Rastrigin function.