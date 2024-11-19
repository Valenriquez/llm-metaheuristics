# Name: Enhanced Hybrid Metaheuristic with Parameter Tuning

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]  # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

heur = [
    ('random_sample', {}, 'all'), 
    ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'gaussian'}, 'greedy'),
    ('swarm_dynamic', {'factor': 0.65, 'self_conf': 2.4, 'swarm_conf': 2.6, 'version': 'inertial', 'distribution': 'uniform'}, 'probabilistic'), 
    ('spiral_dynamic', {'radius': 1.0, 'angle': 30.0, 'sigma': 0.05}, 'all')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This enhanced hybrid metaheuristic combines four different search operators: random sampling, local random walk, swarm dynamics with optimized parameters, and spiral dynamic. 
# The parameter tuning aims to balance the exploration and exploitation strategies more effectively.
# Random sampling helps in covering the entire search space initially, while the local random walk refines the solution by making small steps around the current best solution.
# Swarm dynamics simulates the behavior of particles in a flock or school, promoting collective intelligence and avoiding premature convergence with optimized parameters for the swarm's inertia factor and confidence weights.
# Spiral dynamic offers a unique approach to exploring the landscape by moving along spiral paths, which can be particularly effective for certain types of problems. The adjusted radius and angle ensure it traverses the space efficiently.
# By combining these operators with well-tuned parameters, this enhanced hybrid approach aims to leverage the strengths of each operator while mitigating their weaknesses, resulting in more robust search capabilities that are better suited for solving the Bohachevsky function effectively.