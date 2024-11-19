# Name: Hybrid Metaheuristic for Optimizing Benchmark Functions
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
    ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'gaussian'}, 'greedy'),
    ('spiral_dynamic', {'radius': 0.9, 'angle': 22.5, 'sigma': 0.1}, 'probabilistic'),
    ('swarm_dynamic', {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'}, 'all')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# This hybrid metaheuristic combines the strengths of different optimization techniques. 
# 'random_sample' provides a broad exploration of the search space, ensuring that the algorithm does not get stuck in local minima.
# 'local_random_walk' refines the solution by taking small steps randomly based on a given probability and scale, which helps in fine-tuning the solution.
# 'spiral_dynamic' uses a spiral pattern to guide the search process, potentially leading to faster convergence compared to linear patterns.
# 'swarm_dynamic' employs swarm intelligence principles, utilizing a group of agents (or particles) that move around and communicate with each other to find optimal solutions.