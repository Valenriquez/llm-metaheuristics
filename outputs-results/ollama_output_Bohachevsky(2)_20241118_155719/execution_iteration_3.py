# Name: HybridMetaheuristic
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2)  # Selected problem is Bohachevsky function with 2 dimensions.
prob = fun.get_formatted_problem()

heur = [
    ('random_sample', {}, 'all'),  # Initial random sampling to explore the search space.
    ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'uniform'}, 'greedy'),
    ('swarm_dynamic', {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'uniform'}, 'metropolis')
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The HybridMetaheuristic combines a random sampling operator to initialize the search space, followed by a local random walk to explore neighborhood solutions.
# A swarm dynamic operator is added for collective exploration and exploitation. 
# The 'all' selector ensures thorough exploration initially, while 'greedy' helps in faster convergence with promising neighborhoods.
# 'Metropolis' selection aids in diversifying search efforts by accepting slightly worse solutions based on probability.