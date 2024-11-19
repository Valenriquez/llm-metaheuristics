# Name: Multi-Operator Hybrid Metaheuristic (MOH)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10)  # This is the selected problem, fixed according to your instructions.
prob = fun.get_formatted_problem()

heur = [
    (  # Search operator 1
        'random_sample',
        {},
        'all'
    ),
    (
        'swarm_dynamic',
        {
            'factor': 0.7,
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
        'all'
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# # The Multi-Operator Hybrid Metaheuristic (MOH) combines three different search operators to explore the solution space more effectively.
# # It starts with a random sampling of solutions to initialize the population.
# # Then, it uses a swarm dynamic operator to simulate a social learning process, where particles move towards better positions based on their personal best and global best.
# # A spiral dynamic operator is also included to perform fine-grained exploration in the vicinity of promising regions.