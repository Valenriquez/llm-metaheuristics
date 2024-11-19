# Name: Hybrid Metaheuristic with Variable Operators
# Code:
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2] # Remember to write well this line: 'project_dir = Path(__file__).resolve().parents[2]'
sys.path.insert(0, str(project_dir))
import benchmark_func as bf
import metaheuristic as mh

fun = bf.Rastrigin(10) # This is the selected problem, the problem may vary depending on the case.
prob = fun.get_formatted_problem()

if prob['dimensions'] > 3:
    selector = 'all'
else:
    selector = 'greedy'

heur = [
    (  # Search operator 1
        'local_random_walk',
        {
            'probability': 0.75,
            'scale': 1.0,
            'distribution': 'uniform'
        },
        selector
    ),
    (
        'spiral_dynamic',
        {
            'radius': 0.9,
            'angle': 22.5,
            'sigma': 0.1
        },
        selector
    )
]

met = mh.Metaheuristic(prob, heur, num_iterations=100)
met.verbose = True
met.run()

print('x_best = {}, f_best = {}'.format(*met.get_solution()))

# Short explanation and justification:
# The Hybrid Metaheuristic combines the 'local_random_walk' and 'spiral_dynamic' operators to explore the search space effectively.
# 'local_random_walk' helps in making small, local changes which are suitable for fine-tuning the solution.
# 'spiral_dynamic' provides a global exploration mechanism by moving along a spiral path.
# This combination allows for a balanced search that efficiently converges towards the optimal solution.