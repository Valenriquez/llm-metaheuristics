# Name: Hybrid Metaheuristic with Dynamic Operator Selection

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2)
prob = fun.get_formatted_problem()

heur = [
    ('random_sample', {}, 'all'),
    ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'uniform'}, 'greedy'),
    ('swarm_dynamic', {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'gaussian'}, 'probabilistic'),
    ('spiral_dynamic', {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1}, 'all'),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This hybrid metaheuristic combines four different search operators: random sampling, local random walk, swarm dynamics, and spiral dynamic.
# The use of a greedy selector allows for a balance between exploration and exploitation in the search space.
# Local random walk helps in fine-tuning the solution while maintaining diversity due to its probabilistic nature.
# Swarm dynamics simulates the behavior of particles in a flock or school, promoting collective intelligence and avoiding premature convergence.
# Spiral dynamic offers a unique approach to exploring the landscape by moving along spiral paths, which can be particularly effective for certain types of problems.
# The hybrid approach aims to leverage the strengths of each operator while mitigating their weaknesses, resulting in more robust search capabilities.