# Name: Randomized Multi-Objective Optimization (RMO)
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(15)  # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

# Search operators with random sampling and local walk to explore different regions of the search space
heur = [
    ('random_sample', {}, 'greedy'),
    ('local_random_walk', {'probability': 0.75, 'scale': 1.0, 'distribution': 'uniform'}, 'all'),
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# RMO combines random sampling to explore the solution space globally and local random walk to exploit promising regions. 
# This hybrid approach helps in finding a good balance between exploration and exploitation, which is crucial for complex optimization problems like Rastrigin's function with 15 dimensions.
# The greedy selector ensures that each move improves the current solution at least as much as the next one according to the selected operator.
# By using both operators and selectors effectively, RMO aims to efficiently converge to a high-quality solution while exploring a wide range of potential solutions.