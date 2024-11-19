# Name: Advanced Hybrid Metaheuristic

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]  # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1: Local Random Walk
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        'all'  # Bigger selector for dimension 3 or more
    ),
    (  # Search operator 2: Swarm Dynamic
        'swarm_dynamic',
        {
            'factor': 0.7,
            'self_conf': 2.54,
            'swarm_conf': 2.56,
            'version': 'inertial',
            'distribution': 'uniform'
        },
        'all'  # Bigger selector for dimension 3 or more
    ),
    (  # Search operator 3: Spiral Dynamic
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        'probabilistic'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This metaheuristic combines three effective search operators: Local Random Walk, Swarm Dynamic, and Spiral Dynamic.
# The Local Random Walk operator is suitable for fine-grained exploration around the current best solution.
# The Swarm Dynamic operator provides broad search capability by mimicking social animal behavior.
# The Spiral Dynamic operator efficiently navigates through the solution landscape.

# Using 'all' as the selector ensures that both operators have access to all dimensions of the problem space, which is particularly important for high-dimensional problems like Rastrigin with 10 dimensions. This hybrid approach aims to balance exploitation and exploration, leading to improved convergence and solution quality.

# Additionally, the use of different selectors helps in balancing exploration and exploitation by providing diverse search behaviors during the optimization process.