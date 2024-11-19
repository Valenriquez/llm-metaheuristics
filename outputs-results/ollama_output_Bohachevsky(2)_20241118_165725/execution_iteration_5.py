# Name: Adaptive Hybrid Metaheuristic for Optimization

# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Bohachevsky(2) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Initialize search operators and selectors
operators = [
    ("local_random_walk", 
     {'probability': 0.75, 'scale': 1.0, 'distribution': 'uniform'}, 
     'greedy'),
    ("swarm_dynamic",
     {'factor': 0.7, 'self_conf': 2.54, 'swarm_conf': 2.56, 'version': 'inertial', 'distribution': 'gaussian'},
     'all')
]

# Create metaheuristic instance
met = mh.Metaheuristic(prob, operators, num_iterations=100)
met.verbose = True

# Run the metaheuristic
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# 1. The 'local_random_walk' operator is used to explore the local neighborhood of the current solution, enhancing fine-tuning.
# 2. The 'swarm_dynamic' operator introduces global exploration capabilities, which are crucial in handling complex landscapes.
# 3. The combination of both operators ensures a balance between exploitation and exploration, leading to more effective optimization.